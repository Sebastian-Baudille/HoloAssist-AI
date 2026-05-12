from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ur3e_rl_env.envs.ur3e_pick_place_env import UR3ePickPlaceEnv
from ur3e_rl_env.train_ppo import format_duration


MODEL_DIR = "./rl_models"
MODEL_PATH = os.getenv(
    "UR3E_RL_MODEL_PATH",
    os.path.join(MODEL_DIR, "ppo_ur3e_reach_object"),
)
PRETRAINED_PATH = os.path.join(MODEL_DIR, "ppo_ur3e_pretrained")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints_parallel")
GAZEBO_LOG_DIR = os.path.join(MODEL_DIR, "gazebo_parallel_logs")

NUM_ENVS = int(os.getenv("UR3E_RL_NUM_ENVS", "4"))
BASE_ROS_DOMAIN_ID = int(os.getenv("UR3E_RL_BASE_ROS_DOMAIN_ID", "30"))
BASE_GAZEBO_MASTER_PORT = int(os.getenv("UR3E_RL_BASE_GAZEBO_MASTER_PORT", "11400"))
LAUNCH_TIMEOUT_SEC = float(os.getenv("UR3E_RL_PARALLEL_LAUNCH_TIMEOUT", "90.0"))

TOTAL_TIMESTEPS = int(os.getenv("UR3E_RL_TOTAL_TIMESTEPS", "200000"))
CHECKPOINT_EVERY_STEPS = int(os.getenv("UR3E_RL_CHECKPOINT_EVERY_STEPS", "10000"))
PROGRESS_PRINT_EVERY_STEPS = int(os.getenv("UR3E_RL_PROGRESS_EVERY_STEPS", "1000"))

TORCH_NUM_THREADS = int(os.getenv("UR3E_RL_TORCH_THREADS", "2"))
TORCH_NUM_INTEROP_THREADS = int(os.getenv("UR3E_RL_TORCH_INTEROP_THREADS", "1"))
PPO_N_STEPS = int(os.getenv("UR3E_RL_PPO_N_STEPS", "512"))
PPO_BATCH_SIZE = int(os.getenv("UR3E_RL_PPO_BATCH_SIZE", "256"))

CONTROL_DT = float(os.getenv("UR3E_RL_CONTROL_DT", "0.1"))
RESET_DURATION = float(os.getenv("UR3E_RL_RESET_DURATION", "0.4"))
MAX_EPISODE_STEPS = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))
USE_MOVEIT_COLLISION_CHECKER = os.getenv(
    "UR3E_RL_USE_MOVEIT_COLLISION_CHECKER",
    "0",
).lower() in {"1", "true", "yes"}
MOVEIT_GROUP_NAME = os.getenv("UR3E_RL_MOVEIT_GROUP_NAME", "ur_onrobot_manipulator")
MOVEIT_STATE_VALIDITY_SERVICE = os.getenv(
    "UR3E_RL_MOVEIT_STATE_VALIDITY_SERVICE",
    "/check_state_validity",
)
MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE = os.getenv(
    "UR3E_RL_MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE",
    "0",
).lower() in {"1", "true", "yes"}


class ParallelGazeboUR3eEnv(UR3ePickPlaceEnv):
    """A UR3e env that owns one isolated Gazebo launch process."""

    def __init__(
        self,
        worker_id: int,
        ros_domain_id: int,
        gazebo_master_port: int,
        launch_timeout_sec: float,
        **env_kwargs: Any,
    ) -> None:
        self.worker_id = int(worker_id)
        self.ros_domain_id = int(ros_domain_id)
        self.gazebo_master_port = int(gazebo_master_port)
        self.gazebo_master_uri = f"http://127.0.0.1:{self.gazebo_master_port}"
        self.launch_timeout_sec = float(launch_timeout_sec)
        self.launch_process: subprocess.Popen | None = None
        self.log_file_handle = None

        os.environ["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        os.environ["IGN_PARTITION"] = str(self.ros_domain_id)
        os.environ["GAZEBO_MASTER_URI"] = self.gazebo_master_uri
        os.environ.setdefault("ROS_LOCALHOST_ONLY", "1")

        self._start_gazebo()
        try:
            super().__init__(**env_kwargs)

            if not self.wait_until_ready(timeout_sec=self.launch_timeout_sec):
                raise RuntimeError(
                    f"Worker {self.worker_id} did not become ready. "
                    f"Check {self._log_path()} for Gazebo launch details."
                )
        except Exception:
            self._stop_gazebo()
            raise

    def close(self) -> None:
        try:
            super().close()
        finally:
            self._stop_gazebo()

    def _start_gazebo(self) -> None:
        log_path = self._log_path()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_file_handle = log_path.open("w", encoding="utf-8")

        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(self.ros_domain_id)
        env["IGN_PARTITION"] = str(self.ros_domain_id)
        env["GAZEBO_MASTER_URI"] = self.gazebo_master_uri
        env.setdefault("ROS_LOCALHOST_ONLY", "1")

        command = [
            "ros2",
            "launch",
            "ur3e_gazebo_sim",
            "ur3e_pick_place_world.launch.py",
            "paused:=false",
            "gui:=false",
            f"use_moveit_collision_checker:={'true' if USE_MOVEIT_COLLISION_CHECKER else 'false'}",
            f"moveit_group_name:={MOVEIT_GROUP_NAME}",
            f"moveit_state_validity_service:={MOVEIT_STATE_VALIDITY_SERVICE}",
            "moveit_fail_closed_when_unavailable:="
            + ("true" if MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE else "false"),
        ]
        print(
            f"[worker {self.worker_id}] launching Gazebo "
            f"ROS_DOMAIN_ID={self.ros_domain_id} "
            f"IGN_PARTITION={self.ros_domain_id} "
            f"log={log_path}",
            flush=True,
        )
        self.launch_process = subprocess.Popen(
            command,
            env=env,
            stdout=self.log_file_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _stop_gazebo(self) -> None:
        if self.launch_process is not None and self.launch_process.poll() is None:
            try:
                os.killpg(os.getpgid(self.launch_process.pid), signal.SIGINT)
                self.launch_process.wait(timeout=15.0)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                if self.launch_process.poll() is None:
                    os.killpg(os.getpgid(self.launch_process.pid), signal.SIGTERM)
                    try:
                        self.launch_process.wait(timeout=5.0)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(self.launch_process.pid), signal.SIGKILL)
                        self.launch_process.wait(timeout=5.0)

        if self.log_file_handle is not None:
            self.log_file_handle.close()
            self.log_file_handle = None

    def _log_path(self) -> Path:
        return Path(GAZEBO_LOG_DIR) / f"worker_{self.worker_id}.log"


def make_env(worker_id: int):
    def _init() -> ParallelGazeboUR3eEnv:
        return ParallelGazeboUR3eEnv(
            worker_id=worker_id,
            ros_domain_id=BASE_ROS_DOMAIN_ID + worker_id,
            gazebo_master_port=BASE_GAZEBO_MASTER_PORT + worker_id,
            launch_timeout_sec=LAUNCH_TIMEOUT_SEC,
            max_episode_steps=MAX_EPISODE_STEPS,
            control_dt=CONTROL_DT,
            reset_duration=RESET_DURATION,
            ready_timeout_sec=30.0,
        )

    return _init


def main() -> None:
    try:
        import torch
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
        from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3, gymnasium, and torch are required for parallel PPO. "
            "Install them with: python3 -m pip install stable-baselines3 gymnasium"
        ) from exc

    class ParallelTrainingProgressCallback(BaseCallback):
        def __init__(
            self,
            total_timesteps: int,
            print_every_steps: int,
            num_envs: int,
        ) -> None:
            super().__init__()
            self.total_timesteps = total_timesteps
            self.print_every_steps = print_every_steps
            self.num_envs = num_envs
            self.started_at = 0.0
            self.last_printed_step = 0

        def _on_training_start(self) -> None:
            self.started_at = time.monotonic()
            print(
                f"Training PPO with {self.num_envs} Gazebo envs for "
                f"{self.total_timesteps:,} timesteps.",
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
                f"[PPO parallel] {current_step:,}/{self.total_timesteps:,} steps "
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
    os.makedirs(GAZEBO_LOG_DIR, exist_ok=True)

    print(
        "Starting parallel Gazebo PPO. Stop any old Gazebo launches before running this.\n"
        f"  envs: {NUM_ENVS}\n"
        f"  ROS_DOMAIN_IDs: {BASE_ROS_DOMAIN_ID}..{BASE_ROS_DOMAIN_ID + NUM_ENVS - 1}\n"
        f"  Gazebo ports: {BASE_GAZEBO_MASTER_PORT}..{BASE_GAZEBO_MASTER_PORT + NUM_ENVS - 1}\n"
        f"  control_dt: {CONTROL_DT}s\n"
        f"  torch threads: {TORCH_NUM_THREADS}",
        flush=True,
    )

    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)], start_method="spawn")
    env = VecMonitor(env)
    try:
        checkpoint_callback = CheckpointCallback(
            save_freq=max(1, CHECKPOINT_EVERY_STEPS // NUM_ENVS),
            save_path=CHECKPOINT_DIR,
            name_prefix="ppo_ur3e_reach_object_parallel",
        )
        progress_callback = ParallelTrainingProgressCallback(
            total_timesteps=TOTAL_TIMESTEPS,
            print_every_steps=PROGRESS_PRINT_EVERY_STEPS,
            num_envs=NUM_ENVS,
        )

        pretrained_zip = PRETRAINED_PATH + ".zip"
        if os.path.isfile(pretrained_zip):
            print(f"Loading pretrained model from {PRETRAINED_PATH}")
            model = PPO.load(
                PRETRAINED_PATH,
                env=env,
                tensorboard_log="./tb_logs",
                n_steps=PPO_N_STEPS,
                batch_size=PPO_BATCH_SIZE,
                device="cpu",
            )
        else:
            print("No pretrained model found — training from scratch.")
            model = PPO(
                policy="MlpPolicy",
                env=env,
                verbose=1,
                ent_coef=0.01,
                tensorboard_log="./tb_logs",
                n_steps=PPO_N_STEPS,
                batch_size=PPO_BATCH_SIZE,
                device="cpu",
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
