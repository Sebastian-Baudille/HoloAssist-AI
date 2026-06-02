# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""V5: v0's proven reward + elbow_up posture nudge.

Reward shape is identical to dense_grab_v3.py (also v0 + elbow_up). The
difference between v3 and v5 lives in the CFG layer:
    - v3 used 10cm threshold + aggressive scales (lift 10, success 800)
      to target a higher lift — failed three times (drag trap, etc.).
    - v5 uses v0's PROVEN scales (lift 50, success 200) at 5cm threshold
      so accidental closures can fire success and PPO can discover grasping.

Each rN reward gets its own file for explicit lineage clarity, even when
two reward shapes happen to coincide.

Terms:
    1. reach_distance       always-on negative gradient toward cube
    2. xy_alignment         proximity-gated
    3. orient_alignment     proximity-gated
    4. grasp_activation     close + actually-closing
    5. lift_bonus           grasped flag * lift_height
    6. success_bonus        terminal cube lift > threshold
    7. elbow_up             small bonus for forearm_link Z above threshold

DESIGN RULE: max per-step accumulation < success bonus. With v0 scales:
    max per-step (sustained held cube at 4cm) ~= 0.5+0.3+1.0+50*0.04+0.15 = ~3.95
    max episode  ~= 790
    success bonus  = 200
    790 > 200 — theoretical exploit possible, but v0 empirically never
    fell into it because the geometric grasped flag is strict (cube
    must actually be held + lifted, hard to maintain randomly). v5 trusts
    the same empirical observation.
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

    # ---- Term 1: reach distance (v0-identical) ----
    reach_reward = env.cfg.rew_scale_reach * dist

    # ---- Proximity gate (v0-identical: gates xy and orient) ----
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

    # ---- Term 4: Grasp activation (v0-identical: close + actually closing) ----
    finger_width_for_grasp = env._robot.data.joint_pos[:, env._finger_width_idx]
    close_mask = (dist < env.cfg.grasp_distance).float()
    closing_mask = (
        (env.actions[:, 6] < 0.0) & (finger_width_for_grasp < 0.07)
    ).float()
    grasp_activation_reward = env.cfg.rew_scale_grasp_activation * close_mask * closing_mask

    # ---- Term 5: Lift bonus (v0-identical condition + scale) ----
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

    # ---- Term 7: Elbow-up posture nudge (v3-identical, kept small) ----
    # Forearm_link Z position approximates the elbow joint. Higher = elbow
    # raised in overhead-reach posture; lower = elbow near table.
    # Designed-small: max 0.5 * 0.3 = 0.15/step -> 30/episode max.
    forearm_z = env._robot.data.body_link_state_w[:, env._forearm_idx, 2]
    elbow_up_strength = torch.clamp(
        forearm_z - env.cfg.elbow_up_threshold_z,
        min=0.0,
        max=env.cfg.elbow_up_clamp_max,
    )
    elbow_up_reward = env.cfg.rew_scale_elbow_up * elbow_up_strength

    return (
        reach_reward
        + xy_align_reward
        + orient_align_reward
        + grasp_activation_reward
        + lift_reward
        + success_reward
        + elbow_up_reward
    )
