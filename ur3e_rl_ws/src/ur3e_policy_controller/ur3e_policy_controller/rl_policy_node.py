from __future__ import annotations

import os
import time

import numpy as np
import rclpy

from ur3e_rl_env.constants import (
    JOINT_TARGET_DURATION_SEC,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
)
from ur3e_rl_env.ros_interface import RosInterfaceNode, build_observation
from ur3e_safety_layer.safety_checker import SafetyChecker


MODEL_PATH_DEFAULT = "./rl_models/ppo_ur3e_reach_object"
CONTROL_PERIOD_SEC = 0.2


class RLPolicyRunner:
    """Loads a trained PPO policy and publishes safe joint-delta commands."""

    def __init__(self, model_path: str = MODEL_PATH_DEFAULT) -> None:
        try:
            from stable_baselines3 import PPO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "stable-baselines3 is required for rl_policy_node. Install it with: "
                "python3 -m pip install stable-baselines3 gymnasium"
            ) from exc

        if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained PPO model not found at {model_path}")

        self.ros = RosInterfaceNode("ur3e_rl_policy_node")
        self.safety_checker = SafetyChecker()
        self.model = PPO.load(model_path)
        self.joint_lower = np.asarray(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float32)
        self.joint_upper = np.asarray(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float32)
        self.joint_midpoint = 0.5 * (self.joint_lower + self.joint_upper)
        self.joint_range_rad = 0.5 * (self.joint_upper - self.joint_lower)

    def run(self) -> None:
        if not self.ros.wait_until_ready(timeout_sec=20.0):
            raise RuntimeError("ROS state is not ready. Start Gazebo and the required RL topics first.")

        next_tick = time.monotonic()
        while rclpy.ok():
            rclpy.spin_once(self.ros, timeout_sec=0.01)
            now = time.monotonic()
            if now < next_tick:
                continue
            next_tick = now + CONTROL_PERIOD_SEC

            state = self.ros.get_state()
            if state is None:
                self.ros.get_logger().warning(
                    f"Skipping policy step; missing {self.ros.missing_state_fields()}"
                )
                continue

            safety_result = self.safety_checker.check_state(state)
            if not safety_result.safe:
                self.ros.get_logger().warning(f"Skipping unsafe policy step: {safety_result.reason}")
                continue

            observation = build_observation(state)
            action, _ = self.model.predict(observation, deterministic=True)
            action_array = np.asarray(action, dtype=np.float32).reshape(-1)
            normalized = np.clip(action_array[:6], -1.0, 1.0)
            gripper_command = float(action_array[6]) if action_array.size > 6 else 0.0
            requested_target = normalized * self.joint_range_rad + self.joint_midpoint
            current_joints = np.asarray(state["joint_positions"], dtype=np.float32).reshape(6)
            safe_target = self.safety_checker.make_safe_target(
                current_joints=current_joints,
                requested_target_joints=requested_target,
            )
            if gripper_command > 0.5:
                self.ros.close_gripper()
            elif gripper_command < -0.5:
                self.ros.open_gripper()
            self.ros.send_joint_target(safe_target, duration_sec=JOINT_TARGET_DURATION_SEC)

    def close(self) -> None:
        self.ros.destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    model_path = os.environ.get("UR3E_RL_MODEL_PATH", MODEL_PATH_DEFAULT)
    runner = RLPolicyRunner(model_path=model_path)
    try:
        runner.run()
    finally:
        runner.close()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
