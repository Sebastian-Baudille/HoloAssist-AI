# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""PPO hyperparams for the HoloAssist UR3e grab task.

Starting point: copied from reach task's PPO config. Grab is a harder task
(multi-stage reward + extra action dim + cube physics), so we may need to:
  - Increase network size if the policy plateaus before learning to lift
  - Tune entropy_coef (higher early for exploration, lower late for refinement)
  - Increase num_steps_per_env if reward signal is too noisy per batch

These are starting values — adjust based on training metrics.
"""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner config for the grab task."""

    # ---- Rollout / iteration ----
    num_steps_per_env = 24
    max_iterations = 200                       # low default for smoke testing
    save_interval = 50
    # Naming scheme: <task>-r<reward_version>-run<N>. Bump runN manually per
    # training via --experiment_name (e.g., "grab-r0-run2"). See
    # reference-tensorboard-logdir memory for the full scheme.
    experiment_name = "grab-r0-run1"

    # ---- Policy network ----
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    # ---- PPO algorithm ----
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class PPORunnerCfgV1(PPORunnerCfg):
    """PPO runner for the v1 grab task.

    Same hyperparams as v0 — the reward and physics changed, not the
    learning algorithm. Only experiment_name differs so v1 logs land in
    a separate folder and don't mix with v0 in TensorBoard.
    """

    experiment_name = "grab-r1-run1"


@configclass
class PPORunnerCfgV2(PPORunnerCfg):
    """PPO runner for the v2 grab task.

    Same hyperparams as v0/v1 — reward design changed, not learning algo.
    Only experiment_name differs so v2 logs land in a separate folder
    (grab-r2-run1) and don't mix with v0 or v1 in TensorBoard.
    """

    experiment_name = "grab-r2-run1"


@configclass
class PPORunnerCfgV3(PPORunnerCfg):
    """PPO runner for the v3 grab task.

    Same hyperparams as v0/v1/v2 — only experiment_name differs. v3 logs
    land in grab-r3-run1, separate from v0 / v1 / v2.
    """

    experiment_name = "grab-r3-run1"


@configclass
class PPORunnerCfgV4(PPORunnerCfg):
    """PPO runner for the v4 grab task.

    Same hyperparams as previous versions — only experiment_name differs.
    v4 logs land in grab-r4-run1 (or grab-r4-pretest for wiring checks).
    """

    experiment_name = "grab-r4-run1"


@configclass
class PPORunnerCfgV5(PPORunnerCfg):
    """PPO runner for the v5 grab task.

    Same hyperparams as previous versions — only experiment_name differs.
    v5 logs land in grab-r5-run1 (or grab-r5-pretest for wiring checks).
    """

    experiment_name = "grab-r5-run1"


@configclass
class PPORunnerCfgV6(PPORunnerCfg):
    """PPO runner for the v6 grab task.

    Same hyperparams as previous versions — only experiment_name differs.
    v6 logs land in grab-r6-run1 (or grab-r6-pretest for wiring checks).
    """

    experiment_name = "grab-r6-run1"


@configclass
class PPORunnerCfgV0p5(PPORunnerCfg):
    """PPO runner for the v0.5 grab task.

    v0.5 = v0 reward + self-collision only. Pure baseline.
    Same hyperparams as previous versions — only experiment_name differs.
    Logs land in grab-r0p5-run1 (or grab-r0p5-pretest for wiring checks).
    """

    experiment_name = "grab-r0p5-run1"
