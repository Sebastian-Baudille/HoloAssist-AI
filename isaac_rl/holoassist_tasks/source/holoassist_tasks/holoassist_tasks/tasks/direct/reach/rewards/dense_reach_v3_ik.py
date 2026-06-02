# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Dense reach reward - v3: IK-guided + soft joint-limit + action-rate.

WHY V3 EXISTS
-------------
V2 over-penalised motion: with smoothness scales 5x/4x stronger than v1
(action_rate -0.05, joint_vel -0.02, jerk -0.02), the policy learned that
folding the arm into a compact stationary pose maximises reward. The
arm hovered with EE near the base, never extending to reach the target.

The root cause: smoothness penalties dominated the +10 success bonus, so
"don't move" beat "move and reach" in the reward landscape.

V3 takes a different approach, inspired by Guy's ROS reward.py (merged in
from the parallel ROS stack). Instead of adding more smoothness penalties,
we add a **soft reference signal**: an IK solver computes a sensible
elbow-up top-down approach pose per target, and we reward the policy for
matching it. The reference is computed once per reset (via nearest-neighbour
lookup against a precomputed grid - see reach_env.py for the grid logic).

KEY INSIGHT
-----------
The IK reference replaces three v1/v2 reward terms at once:
  - down_incentive : IK reference encodes "TCP above target, pointing down"
  - action_rate    : pulling toward IK ref produces smooth trajectories
                     by construction (still kept here at a mild -0.02 as
                     belt-and-suspenders, per user decision)
  - joint_vel      : same reasoning
  - jerk           : same reasoning

DIFFERENCE FROM dense_reach_v2.py
---------------------------------
Dropped (v2 -> v3):
  - down_incentive (-0.3): redundant with IK reference
  - joint_vel (-0.02): redundant with IK reference
  - jerk (-0.02): redundant with IK reference

Adapted (v2 -> v3):
  - action_rate: -0.05 -> -0.02 (2x v1, 40% of v2; soft smoothness)

Added (NEW in v3, all inspired by Guy):
  - IK reference tracking (-0.10 * ||arm_joints - ik_ref||)
  - Elbow near-straight penalty (-0.20 * nearness when |elbow| < 0.4 rad)
  - Shoulder-lift soft safety (-2.0 * proximity when shoulder_lift > -0.5)

NOT included (deliberately diverging from Guy):
  - Gripper Z-down orientation term: Guy uses -0.15 *
    ||tcp_z_axis - [0,0,-1]||. We can't compute this directly because the
    URDF's `gripper_tcp` link got merged out during Isaac's "Merge Fixed
    Joints" import pass. Skipping it is fine because the IK reference
    itself was solved with the orientation constraint baked into its
    cost function - so matching the reference implicitly enforces
    top-down orientation.

V3 totals: 5 base terms + 4 new terms = 9 terms.

DEPENDENCIES ON env
-------------------
- env._ik_reference        : tensor (num_envs, 6), set by env._reset_idx
                              from a precomputed grid of IK solutions.
                              Required for the IK tracking term.
- env._arm_joint_ids       : indices into env._robot.data.joint_pos for
                              the 6 arm joints (already exists).
- env.prev_actions         : same as v1; cached in _pre_physics_step.
- env._robot.data.joint_pos : per-env joint positions, live from
                              articulation API.

JOINT INDEX CONVENTION
----------------------
After indexing via env._arm_joint_ids, the 6 arm joints are ordered:
    [0] shoulder_pan, [1] shoulder_lift, [2] elbow,
    [3] wrist_1,      [4] wrist_2,       [5] wrist_3
This matches Guy's ROS ordering, so elbow=joints[2] and
shoulder_lift=joints[1] index-compatibly.

HOW TO USE
----------
Registered as a separate gym task (`Template-Holoassist-Reach-V3-IK-Direct-v0`)
via subclass + override pattern in reach/__init__.py. Train with that --task ID.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from ..reach_env import HoloassistReachEnv


def compute(env: "HoloassistReachEnv") -> torch.Tensor:
    """Compute per-env reward - v3 with IK reference + soft joint limits.

    Term breakdown (9 total):
        reach_reward          = cfg.rew_scale_distance       * dist(EE, target)
        success_reward        = cfg.rew_scale_success        * (dist <= cfg.success_tolerance_m)
        action_penalty        = cfg.rew_scale_action         * sum(action^2)
        time_penalty          = cfg.rew_scale_time           * episode_step
        below_plane_penalty   = cfg.rew_scale_below_base_plane * max(0, threshold_z - EE_z)
        ik_track_penalty      = cfg.rew_scale_ik_track_v3    * ||arm_joints - ik_ref||           [NEW]
        elbow_singular_penalty= cfg.rew_scale_elbow_singular_v3 * nearness                       [NEW]
        shoulder_soft_penalty = cfg.rew_scale_shoulder_soft_v3 * max(0, shoulder_lift - thr)     [NEW]
        action_rate_penalty   = cfg.rew_scale_action_rate_v3 * sum((a_t - a_(t-1))^2)            [adapted]

    Returns:
        Tensor of shape (num_envs,) on env.device.
    """
    # Common state (same as v0/v1/v2)
    ee_pos = env._robot.data.body_link_state_w[:, env._ee_body_idx, :3]     # (num_envs, 3)
    target_pos = env._target_pos                                            # (num_envs, 3)
    dist = torch.linalg.norm(ee_pos - target_pos, dim=-1)                   # (num_envs,)

    # Term 1: dense reach gradient
    reach_reward = env.cfg.rew_scale_distance * dist

    # Term 2: success impulse (episode terminates immediately after)
    success = (dist <= env.cfg.success_tolerance_m).float()
    success_reward = env.cfg.rew_scale_success * success

    # Term 3: action magnitude penalty
    action_penalty = env.cfg.rew_scale_action * torch.sum(env.actions ** 2, dim=-1)

    # Term 4: time penalty
    time_penalty = env.cfg.rew_scale_time * env.episode_length_buf.float()

    # Term 5: below-base-plane penalty (table-crash guard)
    threshold_z = env.cfg.robot_base_height_m - env.cfg.base_plane_tolerance_m
    depth_below = torch.clamp(threshold_z - ee_pos[:, 2], min=0.0)
    below_plane_penalty = env.cfg.rew_scale_below_base_plane * depth_below

    # ---- NEW IN V3: IK-guided + soft joint-limit terms ----

    # Read current arm joint positions (6 joints, in the standard
    # shoulder_pan -> wrist_3 order via env._arm_joint_ids).
    arm_joints = env._robot.data.joint_pos[:, env._arm_joint_ids]           # (num_envs, 6)

    # Term 6: IK reference tracking. env._ik_reference holds the precomputed
    # IK solution for the nearest grid cell to each env's target (filled in
    # _reset_idx). Penalty is the L2 distance between current joints and
    # the reference - encourages the policy onto a sensible elbow-up
    # top-down-approach configuration without prescribing exact joint angles.
    ik_err = torch.linalg.norm(arm_joints - env._ik_reference, dim=-1)      # (num_envs,)
    ik_track_penalty = env.cfg.rew_scale_ik_track_v3 * ik_err

    # Term 7: elbow near-straight penalty (singularity / fold avoidance).
    # |elbow| < threshold means the arm is approaching a straight-elbow
    # config (elbow=0 is the singular point). Penalty ramps linearly from
    # 0 at the threshold to rew_scale_elbow_singular_v3 when elbow=0.
    # Pushes the policy toward a healthy bent-elbow configuration.
    elbow = arm_joints[:, 2]                                                # (num_envs,)
    threshold = env.cfg.elbow_singular_threshold_v3
    nearness = torch.clamp((threshold - torch.abs(elbow)) / threshold, min=0.0)
    elbow_singular_penalty = env.cfg.rew_scale_elbow_singular_v3 * nearness

    # Term 8: shoulder-lift soft safety. Penalty when shoulder_lift exceeds
    # the soft threshold (default -0.5 rad), ramping linearly. Provides a
    # gradient warning before any hard joint limit. Guy's ROS reward uses
    # this together with a tightened hard limit at -0.2; we keep USD's
    # looser hard limits and rely on this soft penalty for guidance.
    shoulder_lift = arm_joints[:, 1]                                        # (num_envs,)
    soft_thr = env.cfg.shoulder_soft_threshold_v3
    proximity = torch.clamp(shoulder_lift - soft_thr, min=0.0)
    shoulder_soft_penalty = env.cfg.rew_scale_shoulder_soft_v3 * proximity

    # Term 9: action rate (first-difference smoothness, ADAPTED from v1/v2).
    # Mild scale (-0.02) per user decision; the IK reference does most of
    # the smoothness work, this is belt-and-suspenders for residual jitter.
    action_delta = env.actions - env.prev_actions                           # (num_envs, action_space)
    action_rate_penalty = env.cfg.rew_scale_action_rate_v3 * torch.sum(action_delta ** 2, dim=-1)

    return (
        reach_reward
        + success_reward
        + action_penalty
        + time_penalty
        + below_plane_penalty
        + ik_track_penalty
        + elbow_singular_penalty
        + shoulder_soft_penalty
        + action_rate_penalty
    )
