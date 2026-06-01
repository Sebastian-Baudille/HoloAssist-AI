# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO hyperparams for HoloAssist UR3e reach task (Mapping #7).

Tuned for:
  - 6-DOF arm action space (joint-delta, per actions/joint_delta.py)
  - 12-D observation (6 joint pos + 3 EE pos + 3 target pos, per
    observations/ground_truth_12d.py)
  - Single reward (dense reach + success + small action/time penalties,
    per rewards/dense_reach.py)
  - Default num_envs=64 (smoke testing) — overridable via CLI for Phase 5
    full training (`--num_envs 4096`)

Deltas from the cartpole template (Phase 2 generator default):
  num_steps_per_env       16   - 24                  longer episodes need more transitions/iter
  max_iterations          150  - 200                 small default for dev; CLI override for Phase 5
  actor_hidden_dims     [32,32] - [256, 128, 64]     6-DOF arm needs more capacity
  critic_hidden_dims    [32,32] - [256, 128, 64]     symmetric with actor
  init_noise_std          1.0  - 0.5                 calmer initial exploration
  actor_obs_normalization False - True               Q5 decision; mixed-scale obs benefit from running mean/std
  critic_obs_normalization False - True              same
  entropy_coef            0.005 - 0.01               manipulation needs more exploration than cartpole

Phase 5 tuning notes (defer until first training shows behaviour):
  - learning_rate may drop to 5e-4 if KL spikes
  - entropy_coef may drop to 0.001-0.005 once policy is decent
  - desired_kl may tighten to 0.005 for slower LR adaptation
  - Network may grow to [512, 256, 128] if Phase 4b pick-place expands obs
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for the reach task."""

    # ---- Rollout / iteration ----
    num_steps_per_env = 24                     # 24 x num_envs transitions per iter
    max_iterations = 200                       # low default for dev — CLI override for Phase 5
    save_interval = 50                         # checkpoint every 50 iters
    experiment_name = "holoassist_reach_direct"

    # ---- Policy network ----
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,                    # calmer than cartpole's 1.0
        actor_obs_normalization=True,          # Q5: running mean/std for 12-D mixed-scale obs
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],      # standard manipulation network
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # ---- PPO algorithm ----
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,                     # higher than cartpole — manipulation exploration
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",                   # adaptive LR based on KL
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
