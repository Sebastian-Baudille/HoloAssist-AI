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

from ur3e_rl_env.reward import check_failure, check_success, compute_reward
from ur3e_rl_env.ros_interface import OBSERVATION_SIZE, RosInterfaceNode, build_observation
from ur3e_safety_layer.safety_checker import SafetyChecker


HOME_JOINTS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    dtype=np.float32,
)
DEFAULT_MAX_EPISODE_STEPS = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))
DEFAULT_CONTROL_DT = float(os.getenv("UR3E_RL_CONTROL_DT", "0.2"))
DEFAULT_RESET_DURATION = float(os.getenv("UR3E_RL_RESET_DURATION", "1.0"))


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

        self.action_space = spaces.Box(
            low=-0.03,
            high=0.03,
            shape=(6,),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(OBSERVATION_SIZE,),
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
        self._spin_for(self.reset_duration)

        ready = self.ros.wait_until_ready(self.ready_timeout_sec)
        state = self.ros.get_state() if ready else None
        if state is None:
            return self._zero_observation(), {
                "ready": False,
                "missing": self.ros.missing_state_fields(),
            }

        return build_observation(state), {"ready": True}

    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self.step_count += 1
        clipped_action = np.clip(
            np.asarray(action, dtype=np.float32).reshape(6),
            self.action_space.low,
            self.action_space.high,
        )

        state = self.ros.get_state()
        if state is None:
            return self._missing_state_step("missing state before command")

        state_safety = self.safety_checker.check_state(state)
        if not state_safety.safe:
            reward = compute_reward(state, clipped_action, self.step_count)
            return (
                build_observation(state),
                reward,
                True,
                False,
                {
                    "is_success": False,
                    "failure_reason": state_safety.reason,
                },
            )

        target_joints = self.safety_checker.make_safe_target(
            state["joint_positions"],
            clipped_action,
        )
        self.ros.send_joint_target(target_joints, duration_sec=self.control_dt)
        self._spin_for(self.control_dt)

        new_state = self.ros.get_state()
        if new_state is None:
            return self._missing_state_step("missing state after command")

        observation = build_observation(new_state)
        reward = compute_reward(new_state, clipped_action, self.step_count)
        success = check_success(new_state)
        failure = check_failure(new_state)
        terminated = success or failure
        truncated = self.step_count >= self.max_episode_steps

        distance = float(
            np.linalg.norm(
                np.asarray(new_state["end_effector_position"], dtype=np.float32)
                - np.asarray(new_state["object_position"], dtype=np.float32)
            )
        )
        info = {
            "is_success": success,
            "distance_to_cube": distance,
            "collision": bool(new_state.get("collision_flag", False)),
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self.ros.destroy_node()
        if self._owns_rclpy and rclpy.ok():
            rclpy.shutdown()

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
        return np.zeros((OBSERVATION_SIZE,), dtype=np.float32)
