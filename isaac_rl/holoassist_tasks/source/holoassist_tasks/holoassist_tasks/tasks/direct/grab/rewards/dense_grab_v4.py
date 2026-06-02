# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""V4: v3 reward + anti-drag penalty + boosted grasp_activation.

Pulls from v3 (dense_grab_v3.py) unchanged:
    - Terms 1-6 are v0's proven 6 terms (reach, xy_align, orient_align,
      grasp_activation, lift_bonus, success_bonus) with the v0 condition
      definitions intact.
    - Term 7 is the elbow_up posture nudge introduced in v3.

Adds ONE new term and CHANGES one scale:
    - Term 8 (NEW): anti-drag penalty. -1.0/step when either finger tip
      Z-position is at or below the table surface. Designed to break the
      "finger drag" local optimum v3 fell into (gripper aligns over cube
      but fingers descend to table level where they cannot close).
    - `rew_scale_grasp_activation` boosted from v3's 1.0 -> 2.0. The
      higher scale makes the descent-and-close transition substantially
      more rewarding so PPO can find it via gradient (v3 had grasp_act
      too small to compete with the per-step accumulation of the dragging
      state).

Reward-balance math (the invariant: max per-step accumulation < success):

    Per-step max (cube + close + closing + lift held at 0.09m):
        grasp_act 2.0 + lift 10*0.09 + xy 0.5 + orient 0.3 + elbow 0.15 = 3.85/step
    Over 200 steps:                                                    = 770
    Success bonus:                                                     = 800
    Margin:                                                            = +30 (tight but safe)

    Anti-drag penalty applies independently:
        -1.0/step when fingers drag = -200/episode max if always dragging.
    Combined with v3-style dragging state:
        dragging_state_reward ~= v3's ~750 - 200 = +550
    vs successful completion ~= +1100 - 1300.

Same env class (HoloassistGrabEnv); switched via cfg fields.
Logs land in grab-r4-run1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from ..grab_env import HoloassistGrabEnv


_TARGET_DOWN = torch.tensor([0.0, 0.0, -1.0])


def compute(env: "HoloassistGrabEnv") -> torch.Tensor:
    """Compute per-env reward — 8 terms summed.

    Returns:
        Tensor of shape (num_envs,) on env.device.
    """
    n = env.num_envs

    # ---- Common state ----
    left_pos  = env._robot.data.body_link_state_w[:, env._left_finger_idx,  :3]
    right_pos = env._robot.data.body_link_state_w[:, env._right_finger_idx, :3]
    gripper_center = (left_pos + right_pos) * 0.5
    cube_pos = env._cube.data.root_pos_w[:, :3]

    delta = gripper_center - cube_pos
    dist = torch.linalg.norm(delta, dim=-1)
    xy_offset = torch.linalg.norm(delta[:, :2], dim=-1)
    z_offset = torch.abs(delta[:, 2])

    table_z = env.cfg.robot_base_height_m

    # ---- Term 1: reach distance (v0-identical) ----
    reach_reward = env.cfg.rew_scale_reach * dist

    # ---- Proximity gate (v0-identical) ----
    proximity_mask = (z_offset < env.cfg.alignment_z_gate).float()

    # ---- Term 2: XY alignment (v0-identical, proximity-gated) ----
    xy_strength = torch.clamp(
        1.0 - xy_offset / env.cfg.xy_alignment_threshold, min=0.0
    )
    xy_align_reward = env.cfg.rew_scale_xy_align * xy_strength * proximity_mask

    # ---- Term 3: Orientation alignment (v0-identical, proximity-gated) ----
    ee_quat = env._robot.data.body_link_state_w[:, env._left_finger_idx, 3:7]
    local_z = torch.zeros((n, 3), device=env.device)
    local_z[:, 2] = 1.0
    gripper_z_world = quat_apply(ee_quat, local_z)
    target_down = _TARGET_DOWN.to(env.device).expand(n, -1)
    orient_err = torch.linalg.norm(gripper_z_world - target_down, dim=-1)
    orient_strength = torch.clamp(
        1.0 - orient_err / env.cfg.orient_alignment_threshold, min=0.0
    )
    orient_align_reward = env.cfg.rew_scale_orient_align * orient_strength * proximity_mask

    # ---- Term 4: Grasp activation (v0 condition, scale boosted in v4 cfg) ----
    finger_width_for_grasp = env._robot.data.joint_pos[:, env._finger_width_idx]
    close_mask = (dist < env.cfg.grasp_distance).float()
    closing_mask = (
        (env.actions[:, 6] < 0.0) & (finger_width_for_grasp < 0.07)
    ).float()
    grasp_activation_reward = env.cfg.rew_scale_grasp_activation * close_mask * closing_mask

    # ---- Term 5: Lift bonus (v0-identical condition) ----
    finger_width = env._robot.data.joint_pos[:, env._finger_width_idx]
    grasped = (
        (cube_pos[:, 2] > table_z + 0.005) &
        (finger_width < env.cfg.grasped_gripper_width) &
        (dist < env.cfg.grasped_distance)
    ).float()
    lift_height = torch.clamp(cube_pos[:, 2] - table_z, min=0.0)
    lift_reward = env.cfg.rew_scale_lift * lift_height * grasped

    # ---- Term 6: Success bonus (v0-identical) ----
    success = (cube_pos[:, 2] > table_z + env.cfg.success_lift_height).float()
    success_reward = env.cfg.rew_scale_success * success

    # ---- Term 7: Elbow-up posture nudge (v3-identical) ----
    forearm_z = env._robot.data.body_link_state_w[:, env._forearm_idx, 2]
    elbow_up_strength = torch.clamp(
        forearm_z - env.cfg.elbow_up_threshold_z,
        min=0.0,
        max=env.cfg.elbow_up_clamp_max,
    )
    elbow_up_reward = env.cfg.rew_scale_elbow_up * elbow_up_strength

    # ---- Term 8: Anti-drag penalty (NEW in v4) ----
    # Penalises the policy when either finger's body link frame is within
    # `drag_threshold_above_table` of the table surface. The inner_finger
    # link frame is at the knuckle (joint anchor), which sits ~5 cm above
    # the actual finger tip — so to detect "finger tips on table" we
    # threshold the LINK frame at ~6 cm above the table top, not 5 mm.
    # If the threshold proves wrong, adjust the cfg field rather than
    # this file.
    min_finger_z = torch.minimum(left_pos[:, 2], right_pos[:, 2])
    dragging = (min_finger_z < table_z + env.cfg.drag_threshold_above_table).float()
    drag_penalty = env.cfg.rew_scale_drag_penalty * dragging

    return (
        reach_reward
        + xy_align_reward
        + orient_align_reward
        + grasp_activation_reward
        + lift_reward
        + success_reward
        + elbow_up_reward
        + drag_penalty
    )
