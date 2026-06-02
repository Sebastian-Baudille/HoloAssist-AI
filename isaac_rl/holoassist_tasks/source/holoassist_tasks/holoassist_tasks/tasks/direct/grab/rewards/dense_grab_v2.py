# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""V2: 8-term reward — overhead approach + time penalty (breaks hover trap).

Differences vs v1 (dense_grab_v1.py):
    - Standing-reward magnitudes slashed: orient_align 1.5 -> 0.3 (5x smaller),
      approach_height 2.0 -> 0.5 (4x smaller). v1's hover-trap math showed
      these terms could earn +360/episode for doing nothing useful; new
      scales cap that at ~+75/episode.
    - NEW Term 8: per-step time_penalty. Constant -0.5 / step makes hover
      net-negative (~-100/episode for 200 steps of doing nothing). Standard
      manipulation-RL technique — forces forward progress.
    - grasp_activation 1.0 -> 5.0: 5x stronger reward when EE actually
      closes near the cube. Pulls the policy through the descent-grasp
      transition where v1 stalled.
    - lift_bonus 80 -> 100, success_bonus 200 -> 300: stronger terminal
      signals. With time_penalty dragging hover down, the contrast with
      success becomes overwhelming.

Reward balance at v2 scales (rough math):
    - 200-step hover at perfect orient: -9 + 60 + 15 - 100 = -34 (NET NEGATIVE)
    - Successful grasp + lift to 10cm:   ~-50 + 250 + 250 + 300 = ~+750

Hover loses by ~+800. Policy should converge to grasping in 1500-2500 iters.

Keeps all v1 architectural choices:
    - Self-collision ON via HoloassistGrabEnvCfgV2's inherited robot_cfg
    - 10 cm lift target
    - Ungated orient_align (direction kept; only magnitude reduced)
    - Approach_height bonus for above-cube positioning during far approach
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

    # ---- Term 1: reach distance (always-on negative pull) ----
    reach_reward = env.cfg.rew_scale_reach * dist

    # ---- Proximity gate (xy_align only; orient stays ungated as in v1) ----
    proximity_mask = (z_offset < env.cfg.alignment_z_gate).float()

    # ---- Term 2: XY alignment (proximity-gated) ----
    xy_strength = torch.clamp(
        1.0 - xy_offset / env.cfg.xy_alignment_threshold, min=0.0
    )
    xy_align_reward = env.cfg.rew_scale_xy_align * xy_strength * proximity_mask

    # ---- Term 3: Orientation alignment (UNGATED, but small magnitude in v2) ----
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

    # ---- Term 7: Approach height (above cube during far approach) ----
    far_mask = (dist > env.cfg.approach_far_threshold).float()
    ee_z = gripper_center[:, 2]
    height_above = torch.clamp(ee_z - cube_pos[:, 2], min=0.0, max=0.20)
    approach_height_reward = (
        env.cfg.rew_scale_approach_height * far_mask * height_above
    )

    # ---- Term 8: Time penalty (NEW in v2 — breaks the hover trap) ----
    # Constant negative reward per step. Forces forward progress: hovering
    # for 200 steps costs ~+100 vs +0 cost for completing in 50 steps.
    # Combined with reduced standing bonuses, makes hover net-negative.
    time_penalty = env.cfg.rew_scale_time_penalty * torch.ones(n, device=env.device)

    return (
        reach_reward
        + xy_align_reward
        + orient_align_reward
        + grasp_activation_reward
        + lift_reward
        + success_reward
        + approach_height_reward
        + time_penalty
    )
