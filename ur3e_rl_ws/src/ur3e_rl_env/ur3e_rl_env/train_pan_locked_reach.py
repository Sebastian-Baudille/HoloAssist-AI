"""Train sub-policy 2: pan-locked reach (extend + descend combined)."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.pan_locked_reach_env import UR3ePanLockedReachEnv

_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models" / "pan_locked_reach"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "reach_tb_logs"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--envs",      type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--load",      type=str, default=None)
    args = ap.parse_args()

    CKPT_DIR = MODEL_DIR / "checkpoints"
    BEST_DIR = MODEL_DIR / "best"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    BEST_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    def make_env(rank):
        def _init():
            env = UR3ePanLockedReachEnv()
            env.reset(seed=args.seed + rank)
            return env
        return _init

    train_env = VecMonitor(SubprocVecEnv([make_env(i) for i in range(args.envs)]))
    eval_env  = VecMonitor(SubprocVecEnv([make_env(args.envs)]))

    callbacks = [
        CheckpointCallback(save_freq=max(10_000 // args.envs, 1),
                           save_path=str(CKPT_DIR), name_prefix="pan_locked_reach"),
        EvalCallback(eval_env, best_model_save_path=str(BEST_DIR),
                     eval_freq=max(20_000 // args.envs, 1),
                     n_eval_episodes=20, verbose=1),
    ]

    if args.load:
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
    else:
        model = PPO("MlpPolicy", train_env, verbose=1,
                    learning_rate=3e-4, n_steps=2048, batch_size=256,
                    n_epochs=10, gamma=0.99,
                    policy_kwargs=dict(net_arch=[256, 256]),
                    tensorboard_log=str(LOG_DIR))

    model.learn(total_timesteps=args.timesteps, callback=callbacks,
                reset_num_timesteps=args.load is None, progress_bar=True)
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
