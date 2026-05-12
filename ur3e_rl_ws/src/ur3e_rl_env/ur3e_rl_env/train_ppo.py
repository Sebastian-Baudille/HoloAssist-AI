from __future__ import annotations

import os
import time

from ur3e_rl_env.envs.ur3e_pick_place_env import UR3ePickPlaceEnv


MODEL_DIR = "./rl_models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_ur3e_reach_object")
PRETRAINED_PATH = os.path.join(MODEL_DIR, "ppo_ur3e_pretrained")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
TOTAL_TIMESTEPS = int(os.getenv("UR3E_RL_TOTAL_TIMESTEPS", "200000"))
TORCH_NUM_THREADS = int(os.getenv("UR3E_RL_TORCH_THREADS", "8"))
TORCH_NUM_INTEROP_THREADS = int(os.getenv("UR3E_RL_TORCH_INTEROP_THREADS", "1"))
PROGRESS_PRINT_EVERY_STEPS = int(os.getenv("UR3E_RL_PROGRESS_EVERY_STEPS", "1000"))


def format_duration(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def main() -> None:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.env_checker import check_env
        from stable_baselines3.common.monitor import Monitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 and gymnasium are required for PPO training. "
            "Install them with: python3 -m pip install stable-baselines3 gymnasium"
        ) from exc

    class TrainingProgressCallback(BaseCallback):
        def __init__(self, total_timesteps: int, print_every_steps: int) -> None:
            super().__init__()
            self.total_timesteps = total_timesteps
            self.print_every_steps = print_every_steps
            self.started_at = 0.0
            self.last_printed_step = 0

        def _on_training_start(self) -> None:
            self.started_at = time.monotonic()
            print(
                f"Training PPO for {self.total_timesteps:,} timesteps. "
                f"Progress prints every {self.print_every_steps:,} steps.",
                flush=True,
            )

        def _on_step(self) -> bool:
            current_step = min(int(self.num_timesteps), self.total_timesteps)
            should_print = (
                current_step - self.last_printed_step >= self.print_every_steps
                or current_step >= self.total_timesteps
            )
            if not should_print:
                return True

            elapsed = max(time.monotonic() - self.started_at, 1e-6)
            steps_per_second = current_step / elapsed
            remaining_steps = max(self.total_timesteps - current_step, 0)
            eta_seconds = remaining_steps / max(steps_per_second, 1e-6)
            percent = 100.0 * current_step / max(self.total_timesteps, 1)

            print(
                f"[PPO] {current_step:,}/{self.total_timesteps:,} steps "
                f"({percent:.1f}%) | elapsed {format_duration(elapsed)} "
                f"| ETA {format_duration(eta_seconds)} "
                f"| {steps_per_second:.2f} steps/s",
                flush=True,
            )
            self.last_printed_step = current_step
            return True

    torch.set_num_threads(TORCH_NUM_THREADS)
    torch.set_num_interop_threads(TORCH_NUM_INTEROP_THREADS)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = UR3ePickPlaceEnv()
    try:
        if not env.wait_until_ready(timeout_sec=20.0):
            raise RuntimeError(
                "ROS state is not ready. Start Gazebo, the joint controller, TCP pose, "
                "cube pose, and optional goal/collision publishers before training."
            )

        check_env(env, warn=True)
        monitored_env = Monitor(env)
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_ur3e_reach_object",
        )
        progress_callback = TrainingProgressCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            print_every_steps=PROGRESS_PRINT_EVERY_STEPS,
        )

        pretrained_zip = PRETRAINED_PATH + ".zip"
        if os.path.isfile(pretrained_zip):
            print(f"Loading pretrained model from {PRETRAINED_PATH}")
            model = PPO.load(PRETRAINED_PATH, env=monitored_env, tensorboard_log="./tb_logs")
        else:
            print("No pretrained model found — training from scratch.")
            model = PPO(
                policy="MlpPolicy",
                env=monitored_env,
                verbose=1,
                ent_coef=0.01,
                tensorboard_log="./tb_logs",
            )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=CallbackList([checkpoint_callback, progress_callback]),
            log_interval=1,
        )
        model.save(MODEL_PATH)
        print(f"Saved PPO model to {MODEL_PATH}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
