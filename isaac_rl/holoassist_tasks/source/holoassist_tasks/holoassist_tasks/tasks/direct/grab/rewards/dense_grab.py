# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""6-term gated dense reward for the grab task.

Terms (positive bonuses + one always-on dense pull):
    1. reach_distance       always-on negative gradient pulling EE toward cube
    2. xy_alignment         bonus when EE z within 10 cm of cube z AND XY centered
    3. orient_alignment     bonus when EE z within 10 cm of cube z AND gripper pointing down
    4. grasp_activation     bonus when EE close to cube AND gripper signal is "closing"
    5. lift_bonus           bonus when cube grasped, scales with lift height
    6. success_bonus        terminal bonus when cube lifted > success threshold

Design principle: all NEW terms (2-6) are bonuses, not penalties. Walking away
from the cube simply earns less reward — there's no penalty cage that could
encourage degenerate "do nothing" solutions (as v2/v3 of the reach task showed).
Gates on proximity prevent alignment bonuses from dominating the reach phase.

The grasped flag is a geometric heuristic (cube lifted + gripper closed + cube
near gripper). Matches what we can derive from AprilTag/encoder readings in
real-world deploy — no sim-only ground truth needed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from ..grab_env import HoloassistGrabEnv


# World-frame target direction for gripper Z-axis (straight down)
_TARGET_DOWN = torch.tensor([0.0, 0.0, -1.0])


def compute(env: "HoloassistGrabEnv") -> torch.Tensor:
    """Compute per-env reward — 6 terms summed.

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

    # ---- Proximity gate for alignment terms (vertical proximity only) ----
    proximity_mask = (z_offset < env.cfg.alignment_z_gate).float()

    # ---- Term 2: XY alignment ----
    # Linear ramp: +rew_scale at xy_offset=0, 0 at xy_offset=threshold
    xy_strength = torch.clamp(
        1.0 - xy_offset / env.cfg.xy_alignment_threshold, min=0.0
    )
    xy_align_reward = env.cfg.rew_scale_xy_align * xy_strength * proximity_mask

    # ---- Term 3: Orientation alignment ----
    # Gripper Z-axis in world frame: rotate local +Z by the body's quaternion.
    # Using left_inner_finger body's orientation as the gripper reference.
    ee_quat = env._robot.data.body_link_state_w[:, env._left_finger_idx, 3:7]    # (n, 4) wxyz
    local_z = torch.zeros((n, 3), device=env.device)
    local_z[:, 2] = 1.0
    gripper_z_world = quat_apply(ee_quat, local_z)
    target_down = _TARGET_DOWN.to(env.device).expand(n, -1)
    orient_err = torch.linalg.norm(gripper_z_world - target_down, dim=-1)
    orient_strength = torch.clamp(1.0 - orient_err / env.cfg.orient_alignment_threshold, min=0.0)
    orient_align_reward = env.cfg.rew_scale_orient_align * orient_strength * proximity_mask

    # ---- Term 4: Grasp activation ----
    # Bonus when EE is close to cube AND the gripper is ACTUALLY closing
    # (not just the action signal). Requires gripper_width < 0.07 in
    # addition to action[6] < 0 — prevents the "random gripper signal"
    # reward hack where the policy could earn this bonus by toggling the
    # signal without the gripper actually responding.
    finger_width_for_grasp = env._robot.data.joint_pos[:, env._finger_width_idx]
    close_mask = (dist < env.cfg.grasp_distance).float()
    closing_mask = (
        (env.actions[:, 6] < 0.0) & (finger_width_for_grasp < 0.07)
    ).float()
    grasp_activation_reward = env.cfg.rew_scale_grasp_activation * close_mask * closing_mask

    # ---- Term 5: Lift bonus (gated on grasped state) ----
    # Heuristic grasped flag: cube lifted slightly + gripper mostly closed + cube near gripper.
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

    return (
        reach_reward
        + xy_align_reward
        + orient_align_reward
        + grasp_activation_reward
        + lift_reward
        + success_reward
    )
