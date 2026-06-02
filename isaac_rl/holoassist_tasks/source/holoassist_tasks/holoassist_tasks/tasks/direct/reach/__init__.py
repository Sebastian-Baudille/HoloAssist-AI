# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""HoloAssist UR3e reach task.

Registers four Gym envs:
  - Template-Holoassist-Reach-Direct-v0     : default 5-term reward (dense_reach.py)
  - Template-Holoassist-Reach-V1-Direct-v0  : 8-term reward (dense_reach_v1.py) with
                                              down-incentive + action-rate + joint-vel
                                              smoothness terms
  - Template-Holoassist-Reach-V2-Direct-v0  : 9-term reward (dense_reach_v2.py) with
                                              5x stronger action-rate, 4x stronger
                                              joint-vel, plus a new jerk term
                                              (second action difference) — OVER-PENALIZED:
                                              policy folds arm into stationary pose
  - Template-Holoassist-Reach-V3-IK-Direct-v0 : 9-term reward (dense_reach_v3_ik.py) with
                                              IK reference tracking + elbow + shoulder
                                              soft penalties + mild action_rate. Replaces
                                              v2's smoothness penalties with a soft
                                              reference signal computed via scipy IK
                                              (precomputed grid in env init). Adapts
                                              Guy's ROS reward design to Isaac scale.

All four share the same env class, env_cfg, and PPO config. V1/V2/V3 are thin
subclasses that override _get_rewards to call their respective reward modules.
Pattern lets us A/B/C/D compare the four reward shapes in the same TensorBoard.

NOTE on home pose: v3 introduced a change to cfg.home_joint_pos from vertical
zero [0,0,0,0,0,0] to ready pose [0,-π/2,0,-π/2,0,0]. All four tasks now reset
to the ready pose. Old v0/v1/v2 checkpoints trained against vertical zero will
behave oddly in play.py since they never saw the ready-pose initial state. For
an honest A/B comparison vs v3, retrain v0/v1/v2 with the new pose.
"""

import gymnasium as gym
import torch

from . import agents
from .reach_env import HoloassistReachEnv
from .rewards import dense_reach_v1 as _v1_reward
from .rewards import dense_reach_v2 as _v2_reward
from .rewards import dense_reach_v3_ik as _v3_reward


# -----------------------------------------------------------------------------
# Default task — 5-term reward shape (rewards/dense_reach.py)
# -----------------------------------------------------------------------------
gym.register(
    id="Template-Holoassist-Reach-Direct-v0",
    entry_point=f"{__name__}.reach_env:HoloassistReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:HoloassistReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# -----------------------------------------------------------------------------
# V1 task — same env + cfg + PPO, but uses dense_reach_v1.compute() for reward.
# Subclass + override pattern preserves v0 as the default while letting v1
# coexist as a separate gym ID. Train either with --task <id>; TensorBoard
# logs land under different experiment_name auto-prefixes so they don't
# clobber each other.
# -----------------------------------------------------------------------------
class HoloassistReachV1Env(HoloassistReachEnv):
    """Reach env with the v1 reward (smoothness extras)."""

    def _get_rewards(self) -> torch.Tensor:
        return _v1_reward.compute(self)


gym.register(
    id="Template-Holoassist-Reach-V1-Direct-v0",
    entry_point=f"{__name__}:HoloassistReachV1Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:HoloassistReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# -----------------------------------------------------------------------------
# V2 task — same env + cfg + PPO, uses dense_reach_v2.compute() for reward.
# V2 reads env.prev_prev_actions (the action from two steps ago) for its jerk
# term; that cache is allocated + rolled in reach_env.py alongside prev_actions,
# so V2 does NOT need to override __init__/_reset_idx/_pre_physics_step.
# -----------------------------------------------------------------------------
class HoloassistReachV2Env(HoloassistReachEnv):
    """Reach env with the v2 reward (stronger smoothness + jerk term)."""

    def _get_rewards(self) -> torch.Tensor:
        return _v2_reward.compute(self)


gym.register(
    id="Template-Holoassist-Reach-V2-Direct-v0",
    entry_point=f"{__name__}:HoloassistReachV2Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:HoloassistReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)


# -----------------------------------------------------------------------------
# V3 task — IK-guided reach with soft joint-limit penalties.
# Adapts Guy's ROS reward design (kinematics-based IK reference + elbow +
# shoulder soft terms) to Isaac's parallel-env scale (IK precomputed as a
# nearest-neighbour lookup grid at env init). Uses env._ik_reference, which is
# populated by reach_env._reset_idx for every env regardless of task variant.
# -----------------------------------------------------------------------------
class HoloassistReachV3Env(HoloassistReachEnv):
    """Reach env with the v3 IK-guided reward (replaces v1/v2 smoothness penalties)."""

    def _get_rewards(self) -> torch.Tensor:
        return _v3_reward.compute(self)


gym.register(
    id="Template-Holoassist-Reach-V3-IK-Direct-v0",
    entry_point=f"{__name__}:HoloassistReachV3Env",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_cfg:HoloassistReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)
