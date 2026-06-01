# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Dense reach reward - v2: stronger smoothness penalties + new jerk term.

WHY V2 EXISTS
-------------
V1 reduced twitchiness vs the default but residual motion noise remains at
inference time - visible in play.py as micro-corrections every frame. Two
diagnoses:

  1. V1's smoothness scales (action_rate -0.01, joint_vel -0.005) are tiny
     compared to the +10 success bonus and the -0.3 reach gradient, so the
     policy mostly ignores them once it has learned a half-decent reach.

  2. V1 penalises only the FIRST derivative of the action sequence
     (action_rate = sum((a_t - a_(t-1))^2)). The residual twitching lives in
     the SECOND derivative - rapid back-and-forth where consecutive deltas
     cancel out, so action_rate stays moderate while the visible motion is
     still jittery.

V2 makes two changes on top of v1:

  1. Cranks v1's smoothness scales 4-5x:
        action_rate : -0.01  -> -0.05  (5x stronger)
        joint_vel   : -0.005 -> -0.02  (4x stronger)

  2. Adds a 9th term, the jerk penalty (second action difference):
        jerk_t = a_t - 2*a_(t-1) + a_(t-2)
        jerk_penalty = -0.02 * sum(jerk_t ** 2)

     "Jerk" in the kinematic sense (rate-of-change of acceleration). Directly
     penalises the back-and-forth oscillation pattern that the first-derivative
     action_rate term doesn't catch.

DIFFERENCE FROM dense_reach_v1.py
---------------------------------
Same 5 base terms + same down_incentive (unchanged scales). action_rate and
joint_vel read from new *_v2 cfg fields with stronger weights. Plus one new
jerk term. V2 totals 9 terms.

DEPENDENCIES ON env
-------------------
- env.prev_actions       : same as v1; cached in _pre_physics_step before
                            the new action is written.
- env.prev_prev_actions  : NEW. Action from two steps ago. Cached one step
                            BEFORE prev_actions in _pre_physics_step (chain
                            roll). Zero on episode reset. Required for the
                            second-difference jerk term.
- env._robot.data.joint_vel : same as v1; live from articulation API.

HOW TO USE
----------
Register as a third gym task that subclasses HoloassistReachEnv and overrides
_get_rewards to call this module - keeps v0/v1/v2 as three distinct --task IDs
for parallel TensorBoard comparison. See reach/__init__.py for the pattern.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..reach_env import HoloassistReachEnv


def compute(env: "HoloassistReachEnv") -> torch.Tensor:
    """Compute per-env reward - v2 with stronger smoothness + jerk term.

    Terms (5 base + 1 down + 2 stronger smoothness + 1 jerk = 9 total):
        reach_reward         = cfg.rew_scale_distance         * dist(EE, target)
        success_reward       = cfg.rew_scale_success          * (dist <= cfg.success_tolerance_m)
        action_penalty       = cfg.rew_scale_action           * sum(action^2)
        time_penalty         = cfg.rew_scale_time             * episode_step
        below_plane_penalty  = cfg.rew_scale_below_base_plane * max(0, threshold_z - EE_z)
        down_incentive       = cfg.rew_scale_down_incentive   * max(0, EE_z - target_z)
        action_rate_penalty  = cfg.rew_scale_action_rate_v2   * sum((a_t - a_(t-1))^2)             - STRONGER (5x v1)
        joint_vel_penalty    = cfg.rew_scale_joint_vel_v2     * sum(joint_vel^2)                   - STRONGER (4x v1)
        jerk_penalty         = cfg.rew_scale_jerk_v2          * sum((a_t - 2*a_(t-1) + a_(t-2))^2) - NEW (v2)

    Returns:
        Tensor of shape (num_envs,) on env.device.
    """
    # Common state (same as v0/v1)
    ee_pos = env._robot.data.body_link_state_w[:, env._ee_body_idx, :3]     # (num_envs, 3)
    target_pos = env._target_pos                                            # (num_envs, 3)
    dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)                   # (num_envs,)

    # Term 1: dense reach gradient
    reach_reward = env.cfg.rew_scale_distance * dist

    # Term 2: success impulse (one-shot; episode terminates immediately after)
    success = (dist <= env.cfg.success_tolerance_m).float()
    success_reward = env.cfg.rew_scale_success * success

    # Term 3: action magnitude penalty
    action_penalty = env.cfg.rew_scale_action * torch.sum(env.actions ** 2, dim=-1)

    # Term 4: time penalty
    time_penalty = env.cfg.rew_scale_time * env.episode_length_buf.float()

    # Term 5: below-base-plane penalty
    threshold_z = env.cfg.robot_base_height_m - env.cfg.base_plane_tolerance_m
    depth_below = torch.clamp(threshold_z - ee_pos[:, 2], min=0.0)
    below_plane_penalty = env.cfg.rew_scale_below_base_plane * depth_below

    # Term 6: down-incentive (same as v1; encourages descending early)
    height_above_target = torch.clamp(ee_pos[:, 2] - target_pos[:, 2], min=0.0)
    down_incentive = env.cfg.rew_scale_down_incentive * height_above_target

    # Term 7: action rate penalty (first-difference smoothness, STRONGER in v2).
    # Same math as v1 but reads the v2 cfg field with a 5x larger weight, so v1
    # and v2 can coexist with different scales without conflict.
    action_delta = env.actions - env.prev_actions                           # (num_envs, action_space)
    action_rate_penalty = env.cfg.rew_scale_action_rate_v2 * torch.sum(action_delta ** 2, dim=-1)

    # Term 8: joint velocity penalty (STRONGER in v2; same math, 4x weight via v2 cfg).
    arm_joint_vel = env._robot.data.joint_vel[:, :6]                        # (num_envs, 6)
    joint_vel_penalty = env.cfg.rew_scale_joint_vel_v2 * torch.sum(arm_joint_vel ** 2, dim=-1)

    # Term 9: jerk penalty (NEW in v2).
    # Second-difference of the action sequence: a_t - 2*a_(t-1) + a_(t-2).
    # Mathematically the discrete equivalent of the second derivative
    # (kinematic jerk if the action is interpreted as a velocity command, or
    # snap if interpreted as a position command - either way, the residual
    # second-order character that the first-difference action_rate misses).
    # env.prev_prev_actions is zero on episode steps 0 and 1 (reset clears it);
    # the jerk term is therefore garbage on those two steps but the spurious
    # penalty is bounded and washes out across the rollout.
    jerk = env.actions - 2.0 * env.prev_actions + env.prev_prev_actions
    jerk_penalty = env.cfg.rew_scale_jerk_v2 * torch.sum(jerk ** 2, dim=-1)

    return (
        reach_reward
        + success_reward
        + action_penalty
        + time_penalty
        + below_plane_penalty
        + down_incentive
        + action_rate_penalty
        + joint_vel_penalty
        + jerk_penalty
    )
