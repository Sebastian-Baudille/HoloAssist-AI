"""
train_reach.py — Train Model 1: reach the nearest cube.

No ROS. Run directly with python3.

Usage:
    python3 train_reach.py
    python3 train_reach.py --timesteps 200000 --envs 16
    python3 train_reach.py --load rl_models/reach_best/best_model.zip
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.reach_env import UR3eReachEnv

_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "reach_tb_logs"
CKPT_DIR  = MODEL_DIR / "reach_checkpoints"
BEST_DIR  = MODEL_DIR / "reach_best"


def make_env(rank: int, seed: int = 0):
    def _init():
        e = UR3eReachEnv()
        e.reset(seed=seed + rank)
        return e
    set_random_seed(seed)
    return _init


class ProgressCB(BaseCallback):
    def __init__(self, freq: int = 10_000):
        super().__init__()
        self.freq = freq
        self._t0  = 0.0

    def _on_training_start(self):
        self._t0 = time.monotonic()

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            sps = self.num_timesteps / max(time.monotonic() - self._t0, 1e-6)
            print(f"[Reach] {self.num_timesteps:,} steps | {sps:.0f} sps", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--envs",      type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--load",      type=str, default=None)
    args = ap.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.envs)],
                               start_method="fork")
    train_env = VecMonitor(train_env, str(LOG_DIR / "monitor"))
    eval_env  = SubprocVecEnv([make_env(999, args.seed)], start_method="fork")
    eval_env  = VecMonitor(eval_env)

    callbacks = [
        ProgressCB(10_000),
        CheckpointCallback(
            save_freq=max(10_000 // args.envs, 1),
            save_path=str(CKPT_DIR), name_prefix="reach", verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_DIR),
            eval_freq=max(20_000 // args.envs, 1),
            n_eval_episodes=20, verbose=1,
        ),
    ]

    if args.load:
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
        reset_ts = False
    else:
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, learning_rate=3e-4,
            ent_coef=0.01,   # higher entropy → more exploration
            clip_range=0.2, max_grad_norm=0.5,
            device="cpu", verbose=1, seed=args.seed,
            tensorboard_log=str(LOG_DIR),
        )
        reset_ts = True

    print(f"\nTraining Reach model — {args.timesteps:,} steps, {args.envs} envs")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")
    model.learn(args.timesteps, callback=callbacks,
                reset_num_timesteps=reset_ts, progress_bar=True)
    model.save(str(MODEL_DIR / "reach_final"))
    print(f"Saved to {MODEL_DIR / 'reach_final'}.zip")
    train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()
