from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import rclpy

try:
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - exercised only when deps are missing.
    raise ModuleNotFoundError(
        "gymnasium is required for ur3e_rl_env. Install it with: "
        "python3 -m pip install gymnasium stable-baselines3"
    ) from exc

from ur3e_rl_env.constants import (
    JOINT_TARGET_DURATION_SEC,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
)
from ur3e_rl_env.reward import check_failure, check_success, compute_reward
from ur3e_rl_env.ros_interface import OBSERVATION_SIZE, RosInterfaceNode, build_observation
from ur3e_safety_layer.safety_checker import SafetyChecker


HOME_JOINTS = np.array(
    [0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0],
    dtype=np.float32,
)
DEFAULT_MAX_EPISODE_STEPS = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))
DEFAULT_CONTROL_DT = float(os.getenv("UR3E_RL_CONTROL_DT", "0.2"))
DEFAULT_RESET_DURATION = float(os.getenv("UR3E_RL_RESET_DURATION", "1.0"))

RANDOMIZE_CUBE = os.getenv("UR3E_RL_RANDOMIZE_CUBE", "1").lower() in {"1", "true", "yes"}
CUBE_X_RANGE = (float(os.getenv("UR3E_RL_CUBE_X_MIN", "-0.20")), float(os.getenv("UR3E_RL_CUBE_X_MAX", "0.20")))
CUBE_Y_RANGE = (float(os.getenv("UR3E_RL_CUBE_Y_MIN", "-0.45")), float(os.getenv("UR3E_RL_CUBE_Y_MAX", "-0.10")))
CUBE_Z = float(os.getenv("UR3E_RL_CUBE_Z", "1.11"))


def _ensure_rclpy_initialized() -> bool:
    if rclpy.ok():
        return False
    rclpy.init(args=None)
    return True


class UR3ePickPlaceEnv(gym.Env):
    """Joint-delta PPO environment for reaching the cube with the UR3e TCP."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
        control_dt: float = DEFAULT_CONTROL_DT,
        reset_duration: float = DEFAULT_RESET_DURATION,
        ready_timeout_sec: float = 10.0,
    ) -> None:
        super().__init__()
        self._owns_rclpy = _ensure_rclpy_initialized()
        self.ros = RosInterfaceNode()
        self.safety_checker = SafetyChecker()
        self.max_episode_steps = int(max_episode_steps)
        self.control_dt = float(control_dt)
        self.reset_duration = float(reset_duration)
        self.ready_timeout_sec = float(ready_timeout_sec)
        self.step_count = 0
        self.joint_lower = np.asarray(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float32)
        self.joint_upper = np.asarray(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float32)
        self.joint_midpoint = 0.5 * (self.joint_lower + self.joint_upper)
        self.joint_range_rad = 0.5 * (self.joint_upper - self.joint_lower)

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(6,),  # 6 joint deltas only — no gripper in Phase A
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(OBSERVATION_SIZE,),  # 14D Phase A
            dtype=np.float32,
        )

    def wait_until_ready(self, timeout_sec: float | None = None) -> bool:
        return self.ros.wait_until_ready(timeout_sec or self.ready_timeout_sec)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        super().reset(seed=seed)
        self.step_count = 0

        self.ros.send_joint_target(HOME_JOINTS, duration_sec=self.reset_duration)
        if RANDOMIZE_CUBE:
            self.ros.reset_cube_position(self._random_cube_xyz())
        self._spin_for(self.reset_duration)

        ready = self.ros.wait_until_ready(self.ready_timeout_sec)
        state = self.ros.get_state() if ready else None
        if state is None:
            return self._zero_observation(), {
                "ready": False,
                "missing": self.ros.missing_state_fields(),
            }

        return build_observation(state, step_count=self.step_count, max_episode_steps=self.max_episode_steps), {
            "ready": True
        }

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        action_array = np.clip(np.asarray(action, dtype=np.float32).reshape(6), -1.0, 1.0)
        normalized = action_array[:6]
        requested_target = normalized * self.joint_range_rad + self.joint_midpoint

        state = self.ros.get_state()
        if state is None:
            return self._missing_state_step("missing state before command")

        state_safety = self.safety_checker.check_state(state)
        if not state_safety.safe:
            unsafe_obs = build_observation(
                state,
                step_count=self.step_count,
                max_episode_steps=self.max_episode_steps,
            )
            unsafe_info = {
                "collision": bool(state.get("collision_flag", False)),
                "reached":   False,
            }
            reward = compute_reward(state, action_array, self.step_count, unsafe_info)
            return (
                unsafe_obs,
                reward,
                True,
                False,
                {
                    "is_success":   False,
                    "failure_reason": state_safety.reason,
                },
            )

        current_joints = np.asarray(state["joint_positions"], dtype=np.float32).reshape(6)
        target_joints = self.safety_checker.make_safe_target(
            current_joints=current_joints,
            requested_target_joints=requested_target,
        )
        self.ros.send_joint_target(target_joints, duration_sec=JOINT_TARGET_DURATION_SEC)
        self._spin_for(max(self.control_dt, JOINT_TARGET_DURATION_SEC))

        new_state = self.ros.get_state()
        if new_state is None:
            return self._missing_state_step("missing state after command")

        observation = build_observation(
            new_state,
            step_count=self.step_count,
            max_episode_steps=self.max_episode_steps,
        )

        success = check_success(new_state)
        failure = check_failure(new_state)

        distance = float(
            np.linalg.norm(
                np.asarray(new_state["end_effector_position"], dtype=np.float32)
                - np.asarray(new_state["object_position"], dtype=np.float32)
            )
        )
        info = {
            "collision":        bool(new_state.get("collision_flag", False)),
            "reached":          success,
            "is_success":       success,
            "distance_to_cube": distance,
        }
        reward = compute_reward(new_state, action_array, self.step_count, info)
        terminated = success or failure
        truncated  = self.step_count >= self.max_episode_steps
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.ros.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

    def _random_cube_xyz(self) -> tuple[float, float, float]:
        x = float(self.np_random.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1]))
        y = float(self.np_random.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1]))
        return (x, y, CUBE_Z)

    def _spin_for(self, duration_sec: float) -> None:
        deadline = time.monotonic() + duration_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self.ros, timeout_sec=0.02)

    def _missing_state_step(
        self,
        reason: str,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return (
            self._zero_observation(),
            -1.0,
            False,
            True,
            {
                "is_success": False,
                "failure_reason": reason,
                "missing": self.ros.missing_state_fields(),
            },
        )

    def _zero_observation(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)
