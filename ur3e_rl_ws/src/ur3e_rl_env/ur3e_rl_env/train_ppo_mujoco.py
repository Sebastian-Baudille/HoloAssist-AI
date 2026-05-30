"""
train_ppo_mujoco.py — Fast PPO training on the MuJoCo backend.

No ROS required. Run directly with python3.
~100-250× faster than Gazebo parallel workers.

Usage:
    python3 train_ppo_mujoco.py
    python3 train_ppo_mujoco.py --timesteps 1000000 --envs 16
    python3 train_ppo_mujoco.py --load /path/to/checkpoint.zip
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.ur3e_mujoco_env import UR3eMuJoCoEnv

# Paths relative to the workspace root
# __file__ = ur3e_rl_env/train_ppo_mujoco.py
# parents:  ur3e_rl_env/ → src/ur3e_rl_env/ → src/ → ur3e_rl_ws/ → HoloAssist-AI/
_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "mujoco_tb_logs"
CKPT_DIR  = MODEL_DIR / "mujoco_checkpoints"
BEST_DIR  = MODEL_DIR / "mujoco_best"


def make_env(rank: int, seed: int = 0):
    def _init() -> UR3eMuJoCoEnv:
        env = UR3eMuJoCoEnv()
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init


class ProgressCallback(BaseCallback):
    def __init__(self, print_freq: int = 10_000) -> None:
        super().__init__()
        self.print_freq = print_freq
        self._t0 = 0.0

    def _on_training_start(self) -> None:
        self._t0 = time.monotonic()

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            elapsed = time.monotonic() - self._t0
            steps_per_s = self.num_timesteps / max(elapsed, 1e-6)
            print(
                f"[MuJoCo PPO] {self.num_timesteps:,} steps | "
                f"{steps_per_s:.0f} steps/s | {elapsed:.0f}s elapsed",
                flush=True,
            )
        return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO on UR3e MuJoCo env")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--envs",      type=int, default=16)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--load",      type=str, default=None,
                        help="Checkpoint .zip to resume from")
    args = parser.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Creating {args.envs} parallel MuJoCo envs...")
    train_env = SubprocVecEnv(
        [make_env(i, args.seed) for i in range(args.envs)],
        start_method="fork",
    )
    train_env = VecMonitor(train_env, str(LOG_DIR / "monitor"))

    eval_env = SubprocVecEnv([make_env(999, args.seed)], start_method="fork")
    eval_env = VecMonitor(eval_env)

    callbacks = [
        ProgressCallback(print_freq=10_000),
        CheckpointCallback(
            save_freq=max(10_000 // args.envs, 1),
            save_path=str(CKPT_DIR),
            name_prefix="ppo_mujoco",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_DIR),
            eval_freq=max(20_000 // args.envs, 1),
            n_eval_episodes=20,
            verbose=1,
        ),
    ]

    reset_timesteps = True
    if args.load:
        print(f"Resuming from: {args.load}")
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
        reset_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.005,
            clip_range=0.2,
            max_grad_norm=0.5,
            device="cpu",
            verbose=1,
            seed=args.seed,
            tensorboard_log=str(LOG_DIR),
        )

    print(
        f"\nStarting MuJoCo PPO training:\n"
        f"  envs:        {args.envs}\n"
        f"  timesteps:   {args.timesteps:,}\n"
        f"  checkpoints: {CKPT_DIR}\n"
        f"  TensorBoard: tensorboard --logdir {LOG_DIR}\n"
    )

    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        reset_num_timesteps=reset_timesteps,
        progress_bar=True,
    )

    final_path = str(MODEL_DIR / "ppo_mujoco_final")
    model.save(final_path)
    print(f"\nTraining complete. Saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
