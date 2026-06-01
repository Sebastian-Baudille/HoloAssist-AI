# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""12-D ground-truth observation: 6 joint positions + 3 EE pos + 3 target pos.

Replaces the legacy 13-D normalised observation
(`ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/ros_interface.py::build_observation`).
See ISAAC_SIM_PLAN.md § Phase 4c (Mapping #3) for the design rationale —
notably, joint positions are ADDED (legacy didn't include them) and
gripper/bin/timestep fields are DROPPED (broken or irrelevant for reach).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # avoid runtime circular import; only needed for type hints
    from ..reach_env import HoloassistReachEnv


def build(env: "HoloassistReachEnv") -> dict:
    """Build the 12-D reach observation.

    Layout (all in world frame, raw units — no normalisation here; RSL-RL's
    `actor_obs_normalization=True` handles running mean/std at training time):
        [0:6]   arm joint positions (rad)
        [6:9]   end-effector world position (m)
        [9:12]  target world position (m)

    Args:
        env: the HoloassistReachEnv instance. Reads `_robot`, `_ee_body_idx`,
             `_target_pos`.

    Returns:
        `{"policy": tensor of shape (num_envs, 12) on env.device}`
    """
    # 6 arm joint positions (joint indices 0-5 in the articulation; the gripper
    # joints 6-12 are excluded — they're held closed by their drive targets
    # set in _reset_idx and never appear in the action space for reach v1).
    joint_pos = env._robot.data.joint_pos[:, :6]                            # (num_envs, 6)

    # End-effector world position. env._ee_body_idx is resolved once in
    # __init__ via env._robot.find_bodies("gripper_tcp")[0][0].
    ee_pos = env._robot.data.body_link_state_w[:, env._ee_body_idx, :3]     # (num_envs, 3)

    # Target world position — randomised per env per episode in _reset_idx.
    target_pos = env._target_pos                                            # (num_envs, 3)

    obs = torch.cat([joint_pos, ee_pos, target_pos], dim=-1)                # (num_envs, 12)
    return {"policy": obs}
