#!/usr/bin/env python3
"""Activate UR3e controllers after Ignition Fortress has spawned the robot.

Waits for the controller_manager to load the required controllers, then
activates them and publishes a home trajectory.
"""
from __future__ import annotations

import time

import rclpy
from controller_manager_msgs.srv import (
    ConfigureController,
    ListControllers,
    LoadController,
    SwitchController,
)
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


CONTROLLERS = [
    "joint_state_broadcaster",
    "scaled_joint_trajectory_controller",
]
JOINT_TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
UR3E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class ControllerSetup(Node):
    def __init__(self) -> None:
        super().__init__("setup_ur3e_gazebo_controllers")
        self.list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.configure_client = self.create_client(
            ConfigureController,
            "/controller_manager/configure_controller",
        )
        self.load_client = self.create_client(
            LoadController,
            "/controller_manager/load_controller",
        )
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            JOINT_TRAJECTORY_TOPIC,
            10,
        )

    def run(self) -> bool:
        if not self._wait_for_clients():
            return False
        deadline = time.monotonic() + 60.0
        while rclpy.ok() and time.monotonic() < deadline:
            if not self._ensure_controllers_loaded():
                time.sleep(0.5)
                continue
            if not self._wait_for_loaded_controllers(timeout_sec=2.0):
                continue
            if self._activate_controllers():
                self._publish_home()
                self.get_logger().info("Controllers active; UR3e at home position.")
                return True
            time.sleep(0.5)
        self.get_logger().error("Timed out bringing controllers to active state.")
        return False

    def _wait_for_clients(self, timeout_sec: float = 30.0) -> bool:
        clients = [
            (self.list_client, "/controller_manager/list_controllers"),
            (self.load_client, "/controller_manager/load_controller"),
            (self.configure_client, "/controller_manager/configure_controller"),
            (self.switch_client, "/controller_manager/switch_controller"),
        ]
        deadline = time.monotonic() + timeout_sec
        for client, name in clients:
            remaining = max(0.1, deadline - time.monotonic())
            if not client.wait_for_service(timeout_sec=remaining):
                self.get_logger().error(f"Timed out waiting for service {name}")
                return False
        return True

    def _wait_for_loaded_controllers(self, timeout_sec: float = 30.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            controllers = self._list_controllers(timeout_sec=5.0)
            loaded = {c.name for c in controllers}
            if all(n in loaded for n in CONTROLLERS):
                return True
            self.get_logger().info(f"Waiting for controllers. Loaded: {sorted(loaded)}")
            time.sleep(0.5)
        return False

    def _ensure_controllers_loaded(self) -> bool:
        loaded = {c.name for c in self._list_controllers(timeout_sec=5.0)}
        missing = [name for name in CONTROLLERS if name not in loaded]
        for name in missing:
            request = LoadController.Request()
            request.name = name
            future = self.load_client.call_async(request)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
            if not future.done() or future.result() is None:
                self.get_logger().error(f"Timed out loading controller '{name}'.")
                return False
            if not future.result().ok:
                self.get_logger().error(f"Failed loading controller '{name}'.")
                return False
            self.get_logger().info(f"Loaded controller '{name}'.")
        return True

    def _list_controllers(self, timeout_sec: float) -> list:
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.done() and future.result() is not None:
            return list(future.result().controller)
        return []

    def _activate_controllers(self) -> bool:
        controllers = self._list_controllers(timeout_sec=5.0)
        states = {c.name: c.state for c in controllers}

        for name in CONTROLLERS:
            if states.get(name) == "unconfigured":
                if not self._configure_controller(name):
                    return False

        controllers = self._list_controllers(timeout_sec=5.0)
        states = {c.name: c.state for c in controllers}
        if all(states.get(name) == "active" for name in CONTROLLERS):
            self.get_logger().info("Controllers already active.")
            return True

        to_activate = [name for name in CONTROLLERS if states.get(name) == "inactive"]
        if not to_activate:
            self.get_logger().info(f"Controllers not ready to activate yet: {states}")
            return False

        request = SwitchController.Request()
        request.activate_controllers = to_activate
        request.deactivate_controllers = []
        request.strictness = SwitchController.Request.STRICT
        request.activate_asap = True
        request.timeout.sec = 10

        future = self.switch_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=20.0)
        if not future.done() or future.result() is None:
            self.get_logger().error("Timed out activating controllers.")
            return False
        if not future.result().ok:
            self.get_logger().error("Controller manager rejected activation.")
            return False
        self.get_logger().info("Activated UR3e controllers.")
        return True

    def _configure_controller(self, name: str) -> bool:
        request = ConfigureController.Request()
        request.name = name
        future = self.configure_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
        if not future.done() or future.result() is None:
            self.get_logger().error(f"Timed out configuring controller '{name}'.")
            return False
        if not future.result().ok:
            self.get_logger().error(f"Failed configuring controller '{name}'.")
            return False
        self.get_logger().info(f"Configured controller '{name}'.")
        return True

    def _publish_home(self) -> None:
        deadline = time.monotonic() + 5.0
        while rclpy.ok() and time.monotonic() < deadline:
            if self.trajectory_pub.get_subscription_count() > 0:
                break
            rclpy.spin_once(self, timeout_sec=0.05)

        msg = JointTrajectory()
        msg.joint_names = UR3E_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = HOME_JOINTS
        point.velocities = [0.0] * len(HOME_JOINTS)
        point.time_from_start.sec = 1
        msg.points.append(point)

        for _ in range(5):
            self.trajectory_pub.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ControllerSetup()
    try:
        ok = node.run()
        if not ok:
            raise SystemExit(1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
