# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""16-D observation for the grab task.

Layout:
    [0:6]   arm joint positions (6 joints, raw radians)
    [6:9]   gripper_center world position (midpoint of left + right inner finger bodies)
    [9:12]  cube world position
    [12:15] gripper_center - cube position delta (explicit relative position)
    [15]    gripper_width normalized to [0, 1] (0 = closed, 1 = fully open)

The relative position (12:15) is redundant with absolute positions (6:9, 9:12) but
giving it explicitly saves the policy network from learning the subtraction. Cheap
to compute, much easier to learn from.

Sim-to-real compatibility:
    joint positions       <- robot encoders
    gripper_center        <- AprilTag on gripper, transformed to robot-base frame
    cube position         <- AprilTag on cube
    gripper_width         <- gripper controller readback

All five sources are available in the planned hardware setup. No sim-only info.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..grab_env import HoloassistGrabEnv


def build(env: "HoloassistGrabEnv") -> dict:
    """Assemble the 16-D observation vector for all envs in one shot."""
    n = env.num_envs
    obs = torch.zeros((n, 16), device=env.device)

    # [0:6] arm joint positions
    obs[:, 0:6] = env._robot.data.joint_pos[:, env._arm_joint_ids]

    # [6:9] gripper center = midpoint of left + right inner fingers
    left_pos  = env._robot.data.body_link_state_w[:, env._left_finger_idx,  :3]
    right_pos = env._robot.data.body_link_state_w[:, env._right_finger_idx, :3]
    gripper_center = (left_pos + right_pos) * 0.5
    obs[:, 6:9] = gripper_center

    # [9:12] cube world position
    cube_pos = env._cube.data.root_pos_w[:, :3]
    obs[:, 9:12] = cube_pos

    # [12:15] gripper-cube relative position
    obs[:, 12:15] = gripper_center - cube_pos

    # [15] gripper width normalized
    finger_width = env._robot.data.joint_pos[:, env._finger_width_idx]
    obs[:, 15] = (finger_width / env.cfg.gripper_max_width).clamp(0.0, 1.0)

    return {"policy": obs}
