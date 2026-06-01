# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Dense reach reward — v1: adds three smoothness/shaping terms on top of the
default `dense_reach` reward shape.

WHY V1 EXISTS
-------------
The default `dense_reach.py` reward only constrains *3D distance* to the
target. With 6-DOF kinematic redundancy + no smoothness constraints, the
policy can reach via contorted intermediate poses and twitchy motion.

V1 adds three terms layered on top:

    1. Down-incentive       — linear penalty for EE z above target z
    2. Action rate          — squared change between consecutive actions
                              (smoothness of policy output)
    3. Joint velocity       — squared joint velocities (smoothness of
                              joint-space motion in the sim)

All three are optional via their `cfg.rew_scale_*` fields — set any one to
0.0 to disable it individually. Set all three to 0.0 to make v1 behave
identically to the default `dense_reach.py`.

DIFFERENCE FROM dense_reach.py
------------------------------
Identical 5 terms (reach/success/action_penalty/time/below_plane) PLUS the
three above. v1 totals 8 terms.

DEPENDENCIES ON env
-------------------
- `env.prev_actions` — must be cached BEFORE `env.actions` is overwritten in
  `_pre_physics_step`. This is done in the env's delegator (not in the
  action strategy), so any reward strategy can read it.
- `env._robot.data.joint_vel` — live from the articulation API, no caching.

HOW TO USE
----------
In `reach_env.py`, swap the import:
    from .rewards import dense_reach   as reward_strategy   # default
    from .rewards import dense_reach_v1 as reward_strategy   # this file

Or register a second gym task that subclasses HoloassistReachEnv and overrides
`_get_rewards` to call this module — preserves both options as distinct
`--task` IDs (e.g., `Template-Holoassist-Reach-V1-Direct-v0`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..reach_env import HoloassistReachEnv


def compute(env: "HoloassistReachEnv") -> torch.Tensor:
    """Compute the per-env scalar reward — v1 with 3 smoothness terms added.

    Terms (5 from dense_reach + 3 new = 8 total):
        reach_reward         = cfg.rew_scale_distance * dist(EE, target)
        success_reward       = cfg.rew_scale_success  * (dist <= cfg.success_tolerance_m)
        action_penalty       = cfg.rew_scale_action   * sum(action^2)
        time_penalty         = cfg.rew_scale_time     * episode_step
        below_plane_penalty  = cfg.rew_scale_below_base_plane * max(0, threshold_z - EE_z)
        down_incentive       = cfg.rew_scale_down_incentive   * max(0, EE_z - target_z)   - NEW (v1)
        action_rate_penalty  = cfg.rew_scale_action_rate      * sum((aₜ - aₜ₋₁)^2)        - NEW (v1)
        joint_vel_penalty    = cfg.rew_scale_joint_vel        * sum(joint_vel^2)          - NEW (v1)

    Returns:
        Tensor of shape (num_envs,) on env.device.
    """
    # Common state — same as dense_reach
    ee_pos = env._robot.data.body_link_state_w[:, env._ee_body_idx, :3]     # (num_envs, 3)
    target_pos = env._target_pos                                            # (num_envs, 3)
    dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)                   # (num_envs,)

    # Term 1: dense reach gradient
    reach_reward = env.cfg.rew_scale_distance * dist

    # Term 2: success impulse (one-shot, episode terminates immediately after)
    success = (dist <= env.cfg.success_tolerance_m).float()
    success_reward = env.cfg.rew_scale_success * success

    # Term 3: action penalty (squared action magnitude — discourages large commands)
    action_penalty = env.cfg.rew_scale_action * torch.sum(env.actions ** 2, dim=-1)

    # Term 4: time penalty (grows with episode step)
    time_penalty = env.cfg.rew_scale_time * env.episode_length_buf.float()

    # Term 5: below-base-plane penalty (soft linear, no termination)
    threshold_z = env.cfg.robot_base_height_m - env.cfg.base_plane_tolerance_m
    depth_below = torch.clamp(threshold_z - ee_pos[:, 2], min=0.0)
    below_plane_penalty = env.cfg.rew_scale_below_base_plane * depth_below

    # -------- NEW IN V1 --------
    # Term 6: downward-reach incentive — penalty proportional to how far the
    # EE is ABOVE the target z. Cuts off at zero (no reward for being below).
    # Encourages descending toward target height early rather than exploring
    # laterally or upward first.
    height_above_target = torch.clamp(ee_pos[:, 2] - target_pos[:, 2], min=0.0)
    down_incentive = env.cfg.rew_scale_down_incentive * height_above_target

    # Term 7: action rate penalty — squared change between consecutive actions.
    # env.prev_actions is the previous step's action (cached by env._pre_physics_step
    # before the strategy overwrites env.actions). Zero on step 0 of each
    # episode (env._reset_idx clears it). Discourages twitchy policy output —
    # the standard "low-pass on the policy" smoothness term.
    action_delta = env.actions - env.prev_actions                           # (num_envs, action_space)
    action_rate_penalty = env.cfg.rew_scale_action_rate * torch.sum(action_delta ** 2, dim=-1)

    # Term 8: joint velocity penalty — squared joint velocities for the 6 arm
    # joints. Read directly from the articulation API (no caching needed —
    # data.joint_vel is updated by physics each step). Encourages slower, more
    # deliberate joint motion. Distinct from action_rate: action_rate
    # constrains the COMMANDED smoothness (under direct policy control);
    # joint_vel constrains the EXECUTED smoothness (affected by drives + dynamics).
    arm_joint_vel = env._robot.data.joint_vel[:, :6]                        # (num_envs, 6)
    joint_vel_penalty = env.cfg.rew_scale_joint_vel * torch.sum(arm_joint_vel ** 2, dim=-1)

    return (
        reach_reward
        + success_reward
        + action_penalty
        + time_penalty
        + below_plane_penalty
        + down_incentive
        + action_rate_penalty
        + joint_vel_penalty
    )
