"""Train Sub-policy 2: Extend arm to reach cube (lift + elbow)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ur3e_rl_env before any ROS install
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.extend_env import UR3eExtendEnv

_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models" / "extend"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "extend_tb_logs"
CKPT_DIR  = MODEL_DIR / "checkpoints"
BEST_DIR  = MODEL_DIR / "best"


def make_env(rank, seed=0):
    def _init():
        e = UR3eExtendEnv()
        e.reset(seed=seed + rank)
        return e
    set_random_seed(seed)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=500_000)
    ap.add_argument("--envs",      type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--load",      type=str, default=None)
    args = ap.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_env = VecMonitor(SubprocVecEnv([make_env(i, args.seed) for i in range(args.envs)],
                                         start_method="spawn"), str(LOG_DIR / "monitor"))
    eval_env  = VecMonitor(SubprocVecEnv([make_env(999, args.seed)], start_method="spawn"))

    callbacks = [
        CheckpointCallback(save_freq=max(10_000//args.envs, 1),
                           save_path=str(CKPT_DIR), name_prefix="extend"),
        EvalCallback(eval_env, best_model_save_path=str(BEST_DIR),
                     eval_freq=max(20_000//args.envs, 1), n_eval_episodes=20, verbose=1),
    ]

    if args.load:
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
    else:
        model = PPO("MlpPolicy", train_env,
                    n_steps=2048, batch_size=256, n_epochs=10,
                    gamma=0.99, learning_rate=3e-4, ent_coef=0.01,
                    policy_kwargs=dict(net_arch=[128, 128]),
                    device="cpu", verbose=1, seed=args.seed,
                    tensorboard_log=str(LOG_DIR))

    print(f"\nTraining Extend model — {args.timesteps:,} steps, {args.envs} envs")
    try:
        model.learn(args.timesteps, callback=callbacks, reset_num_timesteps=not args.load,
                    progress_bar=True)
        model.save(str(MODEL_DIR / "extend_final"))
        print(f"Saved → {MODEL_DIR / 'extend_final'}.zip")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
