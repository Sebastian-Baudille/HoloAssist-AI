# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""7-D joint-delta + gripper action for the grab task.

Action layout:
    action[0:6] in [-1, 1] : per-arm-joint delta, scaled by cfg.action_scale_rad.
                              Same joint-delta semantics as the reach task.
    action[6]   in [-1, 1] : gripper open/close signal.
                              +1.0 = fully open (open_fraction=1, finger_joint=0)
                              -1.0 = fully closed (open_fraction=0, finger_joint=cfg.gripper_closed_angle)

The 6 linkage joints (finger_joint + 5 mimic followers) get coupled targets per
the RG2 mimic relationship — the same sign convention used in grasp_test_v0.py
and gripper_coupling.py. PhysxMimicJointAPI is metadata-only in this Isaac Sim
5.1 build (per Phase 4b-mimic findings), so the coupling is enforced via
high-stiffness drives (linkage stiffness 2500, set by the env at init).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..grab_env import HoloassistGrabEnv


# Sign per linkage joint. Order matches env._gripper_linkage_ids.
# Index 0 is finger_joint (the master); the rest are followers.
_LINKAGE_SIGNS = (
    +1.0,   # finger_joint
    -1.0,   # left_inner_knuckle_joint
    +1.0,   # left_inner_finger_joint
    -1.0,   # right_outer_knuckle_joint
    -1.0,   # right_inner_knuckle_joint
    +1.0,   # right_inner_finger_joint
)


def process(env: "HoloassistGrabEnv", action: torch.Tensor) -> None:
    """Process the raw policy action: cache it, compute joint targets.

    Stores into:
        env.actions                       : the raw 7-D action (clamped)
        env._joint_pos_target             : (n, 6) arm joint targets after delta
        env._gripper_linkage_magnitude    : (n,) scalar linkage magnitude per env
    """
    action = action.clone().clamp_(-1.0, 1.0)
    env.actions = action

    # --- Arm (joint-delta semantics) ---
    arm_action = action[:, :6]
    arm_delta = arm_action * env.cfg.action_scale_rad
    current_arm = env._robot.data.joint_pos[:, env._arm_joint_ids]
    env._joint_pos_target = current_arm + arm_delta
    # NOTE: joint limits are enforced by PhysX at the drive level — no manual clamp here.

    # --- Gripper ---
    # action[6] in [-1, 1]  ->  open_fraction in [0, 1]
    open_fraction = (action[:, 6] + 1.0) * 0.5
    # linkage_magnitude in [0, gripper_closed_angle]
    env._gripper_linkage_magnitude = env.cfg.gripper_closed_angle * (1.0 - open_fraction)


def apply(env: "HoloassistGrabEnv") -> None:
    """Write joint position targets to the simulator."""
    n = env.num_envs

    # Build the full joint target vector for the joints we drive (arm + gripper).
    # We use two separate calls to keep it tidy.

    # Arm joints
    env._robot.set_joint_position_target(env._joint_pos_target, joint_ids=env._arm_joint_ids)

    # Gripper: 6 linkage joints + 1 finger_width prismatic
    linkage_mag = env._gripper_linkage_magnitude.unsqueeze(-1)   # (n, 1)
    signs = torch.tensor(_LINKAGE_SIGNS, device=env.device).unsqueeze(0)   # (1, 6)
    linkage_targets = linkage_mag * signs                                   # (n, 6)
    env._robot.set_joint_position_target(linkage_targets, joint_ids=env._gripper_linkage_ids_tensor)

    # finger_width follows linearly: 0 when closed, gripper_max_width when open
    # open_fraction = 1 - linkage_mag / gripper_closed_angle
    open_fraction = 1.0 - env._gripper_linkage_magnitude / env.cfg.gripper_closed_angle
    finger_width_target = (open_fraction * env.cfg.gripper_max_width).unsqueeze(-1)   # (n, 1)
    env._robot.set_joint_position_target(
        finger_width_target,
        joint_ids=torch.tensor([env._finger_width_idx], device=env.device, dtype=torch.long),
    )
