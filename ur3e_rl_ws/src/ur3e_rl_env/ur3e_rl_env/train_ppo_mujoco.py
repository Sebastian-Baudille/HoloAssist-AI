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

# Pin every subprocess to 1 BLAS/OpenMP thread BEFORE numpy is imported.
# Without this, 96 subprocesses each spawn 4+ threads = 384 threads on 32 cores,
# the scheduler thrashes and effective CPU drops to ~10%.
import os
os.environ.setdefault("OMP_NUM_THREADS",     "1")
os.environ.setdefault("MKL_NUM_THREADS",     "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import glob
import time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

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
        # Re-pin threads inside each subprocess (inherited via fork, but be explicit).
        for _k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
            os.environ[_k] = "1"
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
    parser.add_argument("--timesteps", type=int, default=5_000_000)
    parser.add_argument("--envs",      type=int, default=32)
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--load",      type=str, default=None,
                        help="Checkpoint .zip to resume from")
    parser.add_argument("--resume",    action="store_true",
                        help="Auto-resume from latest checkpoint")
    args = parser.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    print(f"Creating {args.envs} parallel MuJoCo envs...")
    train_env = SubprocVecEnv(
        [make_env(i, args.seed) for i in range(args.envs)],
        start_method="fork",
    )
    train_env = VecMonitor(train_env, str(LOG_DIR / "monitor"))
    # Normalise rewards so the critic sees consistent scale (~N(0,1)) regardless
    # of whether an episode earns a +50 success bonus or near-zero delta rewards.
    train_env = VecNormalize(train_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    eval_env = SubprocVecEnv([make_env(999, args.seed)], start_method="fork")
    eval_env = VecMonitor(eval_env)
    # Eval env shares normalisation stats but does NOT update them
    eval_env  = VecNormalize(eval_env, norm_obs=False, norm_reward=False, clip_reward=10.0,
                              training=False)

    callbacks = [
        ProgressCallback(print_freq=10_000),
        CheckpointCallback(
            save_freq=max(50_000 // args.envs, 1),
            save_path=str(CKPT_DIR),
            name_prefix="ppo_mujoco",
            verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_DIR),
            eval_freq=max(50_000 // args.envs, 1),
            n_eval_episodes=20,
            verbose=1,
        ),
    ]

    if args.resume and not args.load:
        zips = sorted(glob.glob(str(CKPT_DIR / "*.zip")), key=lambda p: Path(p).stat().st_mtime)
        if zips:
            args.load = zips[-1]
            print(f"Auto-resuming from: {args.load}")
        else:
            print("No checkpoint found — starting from scratch.")

    reset_timesteps = True
    if args.load:
        print(f"Resuming from: {args.load}")
        model = PPO.load(args.load, env=train_env, device="cpu", tensorboard_log=str(LOG_DIR))
        reset_timesteps = False
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            # Longer rollouts → workers stay busy; fewer gradient steps per cycle.
            # Old: 32 envs × 2048 steps / 512 batch × 10 epochs = 1280 mini-batches/cycle
            # New: 48 envs × 4096 steps / 2048 batch ×  5 epochs =  480 mini-batches/cycle
            # ~2.7× less time in update phase = workers idle far less often.
            n_steps=4096,
            batch_size=2048,
            n_epochs=5,
            gamma=0.99,
            learning_rate=3e-4,
            ent_coef=0.0001,
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
