# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""True joint-delta action processor for the 6-DOF UR3e arm.

Replaces the legacy SafetyChecker.make_safe_target slew-limit
(`ur3e_rl_ws/src/ur3e_safety_layer/.../safety_checker.py`) but with cleaner
delta semantics (Q3 decision):

    legacy:   requested = action * joint_range + midpoint   (absolute target)
              safe      = make_safe_target(current, requested)   (slew-limit)
    isaac:    delta     = clip(action * scale, +/- scale)   (true delta)
              target    = current + delta                   (no slew indirection)

URDF joint limits are enforced by PhysX at the physics level (Mapping #4
discussion item — explicit clamp dropped to keep this lean). The cfg-level
scale (cfg.action_scale_rad = 0.08) IS the slew limit.

The action processor is split into two functions because Isaac Lab calls
them at different rates:
    process(env, action):  once per env step  (= once per `decimation` physics steps)
    apply(env):            once per physics step (= `decimation` times per env step)

This lets the target be computed once (smooth policy decision) and written
multiple times (PD drive smoothly tracks toward it).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..reach_env import HoloassistReachEnv


def process(env: "HoloassistReachEnv", action: torch.Tensor) -> None:
    """Translate policy action into a joint position target. Called once per env step.

    Stashes:
        env.actions              — for the action-penalty term in rewards.dense_reach
        env._joint_pos_target    — the per-env arm joint target tensor (num_envs, 6)

    Args:
        env: the HoloassistReachEnv instance.
        action: tensor of shape (num_envs, 6), nominal range [-1, 1].
    """
    # Defensive bound (PPO Tanh should already keep action in [-1, 1])
    action = torch.clip(action, -1.0, 1.0)

    # Stash for reward's action-penalty term
    env.actions = action.clone()

    # True joint-delta: target = current + scaled, clipped action
    scale = env.cfg.action_scale_rad
    delta = torch.clip(action * scale, -scale, scale)                       # (num_envs, 6)

    current = env._robot.data.joint_pos[:, :6]                              # (num_envs, 6)
    env._joint_pos_target = current + delta                                 # (num_envs, 6)
    # PhysX enforces URDF joint limits — no explicit clamp needed here.


def apply(env: "HoloassistReachEnv") -> None:
    """Write the cached joint position target to the sim. Called once per physics step.

    Args:
        env: the HoloassistReachEnv instance. Reads `_joint_pos_target` and
             `_arm_joint_ids` (both set up earlier — _joint_pos_target by
             process() above or by env init; _arm_joint_ids by env __init__).
    """
    env._robot.set_joint_position_target(
        env._joint_pos_target,
        joint_ids=env._arm_joint_ids,
    )
