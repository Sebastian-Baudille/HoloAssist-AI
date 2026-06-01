# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Dense reach reward — port of legacy reward.py (reach-only subset).

Replaces:
    ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/reward.py::compute_reward

Dropped from legacy:
    - grasp stipend (+5.0 while grasped)         — gripper not used in reach v1
    - transport gradient (-0.5 * dist(cube, bin)) — no bin in reach
    - collision penalty (-0.5)                   — no objects to collide with in reach v1
    - cube_in_bin success bonus                  — replaced with EE-near-target success

Kept from legacy (same scales, exposed as cfg fields):
    - dense reach gradient (-0.3 * dist(EE, target))
    - success impulse (+10.0)
    - time penalty (-0.001 * step_count)
    - action penalty (-0.01 * sum(action^2))

Added in Isaac port (not in legacy):
    - below-base-plane penalty (-10.0 * depth_below_threshold)
      Soft linear penalty when EE drops below the robot's mount height (with a
      small tolerance for surface contact). Keeps the policy from learning to
      "dip below the table" — a degenerate strategy where the EE goes under
      the workspace surface to chase targets on the other side.

Success semantic: Q2 chose option C (terminate on success). The success bonus
is therefore effectively one-shot — the episode terminates the step it fires
(termination handled in _get_dones, Mapping #5).
TODO: revisit when adding pick-and-place — pick-place success is cube-in-bin
(not EE-near-target) and may want non-terminating shaping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..reach_env import HoloassistReachEnv


def compute(env: "HoloassistReachEnv") -> torch.Tensor:
    """Compute the per-env scalar reward.

    Reads:
        env._robot.data.body_link_state_w  (EE world pose)
        env._ee_body_idx                    (which body is the EE)
        env._target_pos                     (per-env target, set in _reset_idx)
        env.actions                         (last action, stashed by joint_delta.process)
        env.episode_length_buf              (per-env step counter; maintained by DirectRLEnv)
        env.cfg.rew_scale_*                 (scale factors for each reward term)
        env.cfg.success_tolerance_m         (distance threshold for success)
        env.cfg.robot_base_height_m         (base plane z, for the below-plane penalty)
        env.cfg.base_plane_tolerance_m      (allowed dip below the plane)

    Returns:
        Tensor of shape (num_envs,) on env.device.
    """
    # Distance from EE to target (per env)
    ee_pos = env._robot.data.body_link_state_w[:, env._ee_body_idx, :3]     # (num_envs, 3)
    target_pos = env._target_pos                                            # (num_envs, 3)
    dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)                   # (num_envs,)

    # Term 1: dense reach gradient (always-on, negative)
    reach_reward = env.cfg.rew_scale_distance * dist

    # Term 2: success impulse (fires once on the step the EE arrives within
    # tolerance — episode terminates immediately after via _get_dones, so this
    # is effectively one-shot per episode)
    success = (dist <= env.cfg.success_tolerance_m).float()
    success_reward = env.cfg.rew_scale_success * success

    # Term 3: action penalty (always-on, negative; sums over 6 arm DOFs)
    action_penalty = env.cfg.rew_scale_action * torch.sum(env.actions ** 2, dim=-1)

    # Term 4: time penalty (always-on, negative; grows with episode step)
    time_penalty = env.cfg.rew_scale_time * env.episode_length_buf.float()

    # Term 5: below-base-plane penalty (soft, linear in depth below threshold).
    # Threshold = robot_base_height_m - base_plane_tolerance_m. Penalty is 0
    # when EE is above the threshold, scales linearly with how far below it
    # drops. Keeps the policy from learning to dip under the workspace surface.
    threshold_z = env.cfg.robot_base_height_m - env.cfg.base_plane_tolerance_m
    depth_below = torch.clamp(threshold_z - ee_pos[:, 2], min=0.0)          # (num_envs,)
    below_plane_penalty = env.cfg.rew_scale_below_base_plane * depth_below

    return reach_reward + success_reward + action_penalty + time_penalty + below_plane_penalty
