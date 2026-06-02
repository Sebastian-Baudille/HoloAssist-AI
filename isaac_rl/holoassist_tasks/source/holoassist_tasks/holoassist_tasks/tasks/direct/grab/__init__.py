# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""HoloAssist UR3e grab task.

Phase 4b: reach to a randomly placed cube, close the gripper around it, lift
it above the table. Successor to the reach task; uses the same robot
articulation but extends action space (+gripper), observation space (+cube
state), reward shape (+alignment + grasp + lift + success), and scene (+rigid
cube).

Registers two Gym envs:
  - Template-Holoassist-Grab-Direct-v0  : 6-term reward, no self-collision, 5 cm
                                          lift target. Original baseline.
  - Template-Holoassist-Grab-Direct-v1  : 7-term reward (ungated orient +
                                          approach-height bonus), PhysX
                                          self-collision ON, 10 cm lift target.
                                          Targets overhead-approach posture.

Both share the same env class (HoloassistGrabEnv); differences are entirely
in the cfg (HoloassistGrabEnvCfg vs HoloassistGrabEnvCfgV1). Reward module
is selected via cfg.reward_module, dispatched in grab_env._REWARD_MODULES.

Locked design decisions (see project-phase4b-grab-decisions memory):
  - Learned 7-D gripper (action[6] is the gripper open/close signal)
  - Random cube spawn in the reach task's target zone
  - Fixed ready home pose; --joint_noise_rad cfg hook for later robustness training
  - Combined single-model end-to-end (no separate stages)
  - No transport/bin (lift > threshold = success)
  - Drive stiffness 2500 + gripper_closed_angle 0.50 (validated empirically in
    grasp_test_v0.py — proper PhysxMimicJointAPI is metadata-only in Isaac Sim 5.1)
"""

import gymnasium as gym

from . import agents
from .grab_env import HoloassistGrabEnv
from .grab_env_cfg import (
    HoloassistGrabEnvCfg,
    HoloassistGrabEnvCfgV1,
    HoloassistGrabEnvCfgV2,
    HoloassistGrabEnvCfgV3,
    HoloassistGrabEnvCfgV4,
    HoloassistGrabEnvCfgV5,
)


gym.register(
    id="Template-Holoassist-Grab-Direct-v0",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfg",
    },
)

# V1: overhead-approach reward + PhysX self-collision + 10 cm lift target.
# Same env class as v0, swapped via cfg fields. Logs in grab-r1-run1.
# KNOWN HOVER TRAP — preserved as reward-tuning reference.
gym.register(
    id="Template-Holoassist-Grab-Direct-v1",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV1",
    },
)

# V2: v1 reward rebalanced + time penalty (-0.5/step) to break hover trap.
# Inherits all v1 architectural choices (self-collision, 10cm lift, ungated
# orient direction); only reward scales change + new time_penalty term.
# Logs in grab-r2-run1.
gym.register(
    id="Template-Holoassist-Grab-Direct-v2",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfgV2",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV2",
    },
)

# V3: strategic retreat — v0's proven 6-term reward + small elbow_up posture
# nudge, paired with V1's self-collision constraint, rebalanced scales for
# 10cm lift target. Maintains the invariant max(per-step) * 200 < success_bonus.
gym.register(
    id="Template-Holoassist-Grab-Direct-v3",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfgV3",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV3",
    },
)

# V4: v3 + anti-drag penalty + grasp_activation boost. Targets the v3
# finger-drag failure mode (gripper aligns but fingers descend to table
# where they cannot close). Anti-drag penalises the bad state; grasp_act
# boost makes the closing transition more rewarding to PPO.
gym.register(
    id="Template-Holoassist-Grab-Direct-v4",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfgV4",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV4",
    },
)

# V5: conservative return to v0. Same reward scales as v0 (proven 97% success),
# adds only PhysX self-collision + small elbow_up posture nudge. 5cm lift
# threshold preserves v0's accidental-grasp discovery path that v1/v3/v4 lost.
gym.register(
    id="Template-Holoassist-Grab-Direct-v5",
    entry_point=f"{__name__}.grab_env:HoloassistGrabEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.grab_env_cfg:HoloassistGrabEnvCfgV5",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:PPORunnerCfgV5",
    },
)
