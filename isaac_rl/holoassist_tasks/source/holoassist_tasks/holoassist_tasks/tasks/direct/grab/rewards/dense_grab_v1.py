# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""V1: 7-term reward — overhead approach + higher lift.

Differences vs v0 (dense_grab.py):
    - Orient_align is now UNGATED (applies during entire trajectory, not just
      when EE is near the cube). v0 only rewarded gripper-down when proximity
      was satisfied — letting the policy approach side-on with any wrist
      orientation. v1 pulls toward "gripper-down" from step 0.
    - NEW Term 7: approach_height bonus rewards EE being ABOVE the cube
      during the far-approach phase (when dist > cfg.approach_far_threshold).
      Combined with the always-on orient pull, this incentivises the
      "fly-over-then-descend" trajectory rather than the side-sprawl the
      v0 policy learned.
    - Success threshold + lift scale tuned via cfg (10 cm + scale 80).

Keeps the v0 structural choices (positive bonuses, no penalty cages,
geometric grasped heuristic) — only the shape of the gradient changes.

Paired with HoloassistGrabEnvCfgV1 (which sets enable_self_collisions=True
on the articulation), so the policy can no longer exploit arm-through-arm
fold poses the v0 policy used.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from ..grab_env import HoloassistGrabEnv


_TARGET_DOWN = torch.tensor([0.0, 0.0, -1.0])


def compute(env: "HoloassistGrabEnv") -> torch.Tensor:
    """Compute per-env reward — 7 terms summed.

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

    # ---- Term 1: reach distance (always-on negative pull) ----
    reach_reward = env.cfg.rew_scale_reach * dist

    # ---- Proximity gate (used by xy_align only in v1; orient is now ungated) ----
    proximity_mask = (z_offset < env.cfg.alignment_z_gate).float()

    # ---- Term 2: XY alignment (proximity-gated) ----
    xy_strength = torch.clamp(
        1.0 - xy_offset / env.cfg.xy_alignment_threshold, min=0.0
    )
    xy_align_reward = env.cfg.rew_scale_xy_align * xy_strength * proximity_mask

    # ---- Term 3: Orientation alignment (UNGATED in v1) ----
    # Rewards gripper-down posture throughout the entire trajectory — not
    # only when already close. Primary driver of the overhead approach:
    # even far above the cube, the policy gets rewarded for keeping the
    # wrist pointed down.
    ee_quat = env._robot.data.body_link_state_w[:, env._left_finger_idx, 3:7]
    local_z = torch.zeros((n, 3), device=env.device)
    local_z[:, 2] = 1.0
    gripper_z_world = quat_apply(ee_quat, local_z)
    target_down = _TARGET_DOWN.to(env.device).expand(n, -1)
    orient_err = torch.linalg.norm(gripper_z_world - target_down, dim=-1)
    orient_strength = torch.clamp(
        1.0 - orient_err / env.cfg.orient_alignment_threshold, min=0.0
    )
    orient_align_reward = env.cfg.rew_scale_orient_align * orient_strength

    # ---- Term 4: Grasp activation (close AND actually closing) ----
    finger_width_for_grasp = env._robot.data.joint_pos[:, env._finger_width_idx]
    close_mask = (dist < env.cfg.grasp_distance).float()
    closing_mask = (
        (env.actions[:, 6] < 0.0) & (finger_width_for_grasp < 0.07)
    ).float()
    grasp_activation_reward = env.cfg.rew_scale_grasp_activation * close_mask * closing_mask

    # ---- Term 5: Lift bonus (gated on grasped state) ----
    finger_width = env._robot.data.joint_pos[:, env._finger_width_idx]
    grasped = (
        (cube_pos[:, 2] > table_z + 0.005) &
        (finger_width < env.cfg.grasped_gripper_width) &
        (dist < env.cfg.grasped_distance)
    ).float()
    lift_height = torch.clamp(cube_pos[:, 2] - table_z, min=0.0)
    lift_reward = env.cfg.rew_scale_lift * lift_height * grasped

    # ---- Term 6: Success bonus (terminal) ----
    success = (cube_pos[:, 2] > table_z + env.cfg.success_lift_height).float()
    success_reward = env.cfg.rew_scale_success * success

    # ---- Term 7: Approach height (NEW in v1) ----
    # Bonus for EE being above the cube during the far-approach phase.
    # Combined with always-on orient_align, shapes the trajectory into
    # "rise above cube → orient down → descend" rather than the v0 side-sprawl.
    far_mask = (dist > env.cfg.approach_far_threshold).float()
    ee_z = gripper_center[:, 2]
    height_above = torch.clamp(ee_z - cube_pos[:, 2], min=0.0, max=0.20)
    approach_height_reward = (
        env.cfg.rew_scale_approach_height * far_mask * height_above
    )

    return (
        reach_reward
        + xy_align_reward
        + orient_align_reward
        + grasp_activation_reward
        + lift_reward
        + success_reward
        + approach_height_reward
    )
