# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""V3: v0's proven reward shape + small posture nudge — back to basics.

Pulls almost everything from v0 (dense_grab.py):
    - Term 1: reach_distance       always-on negative gradient toward cube
    - Term 2: xy_alignment         proximity-gated (NOT ungated like v1/v2)
    - Term 3: orient_alignment     proximity-gated (NOT ungated like v1/v2)
    - Term 4: grasp_activation     close + actually-closing (same definition as v0)
    - Term 5: lift_bonus           grasped flag * lift_height (same definition)
    - Term 6: success_bonus        terminal cube lift > threshold

Adds ONE new term:
    - Term 7: elbow_up             small bonus for forearm_link Z above 1.1 m

Why this design:
    v0 trained successfully (97% grasp + lift) with mean_reward 195/200. It only
    had two visible flaws: side-sprawl approach posture and arm-through-arm
    self-collision. v1 and v2 each tried to fix posture via richer reward
    shaping and broke v0's reward balance in different ways:
      - v1: ungated orient + approach_height -> hover trap (no descent)
      - v2: time penalty + boosted lift/grasp -> lateral misalignment exploit
            + hold-cube-below-threshold exploit

V3's strategy: keep v0's reward shape EXACTLY (proximity gates, same
condition definitions, modest scales), and address the two flaws via:
  1. Self-collision constraint at the physics level (PhysX, set in cfg)
  2. Small elbow_up posture reward to gently nudge overhead approach
  3. Higher success_lift_height (10 cm) with rebalanced lift/success scales
     to preserve v0's "per-step accumulation < terminal" invariant

DESIGN INVARIANT (lesson from v1/v2):
    max cumulative per-step reward across an episode MUST be less than
    rew_scale_success. Otherwise the policy will find an exploit that
    avoids termination to accumulate per-step rewards instead of
    completing the task.

    V3 numbers:
        max per-step ~= grasp_act(1) + lift(10*0.09) + xy(0.5) + orient(0.3)
                       + elbow_up(0.15) = ~2.85/step
        max episode  ~= 2.85 * 200 = 570
        success bonus  = 800
        margin         = 230 (success unambiguously dominates)
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

    # ---- Proximity gate (v0-identical: gates BOTH xy_align and orient_align) ----
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

    # ---- Term 5: Lift bonus (v0-identical condition, scale reduced for 10cm threshold) ----
    finger_width = env._robot.data.joint_pos[:, env._finger_width_idx]
    grasped = (
        (cube_pos[:, 2] > table_z + 0.005) &
        (finger_width < env.cfg.grasped_gripper_width) &
        (dist < env.cfg.grasped_distance)
    ).float()
    lift_height = torch.clamp(cube_pos[:, 2] - table_z, min=0.0)
    lift_reward = env.cfg.rew_scale_lift * lift_height * grasped

    # ---- Term 6: Success bonus (v0-identical, scale boosted to maintain dominance) ----
    success = (cube_pos[:, 2] > table_z + env.cfg.success_lift_height).float()
    success_reward = env.cfg.rew_scale_success * success

    # ---- Term 7: Elbow-up posture nudge (NEW in v3) ----
    # Forearm_link Z position approximates the elbow joint location. Higher
    # = elbow is up in an overhead-reach configuration. Lower = side-sprawl.
    # Small magnitude by design (max 30/episode) — only nudges, doesn't dictate.
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
