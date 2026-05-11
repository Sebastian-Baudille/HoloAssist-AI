#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from controller_manager_msgs.srv import ListControllers, SwitchController
from rclpy.node import Node
from std_srvs.srv import Empty
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
        self.declare_parameter("pause_after_setup", True)
        self.pause_after_setup = bool(self.get_parameter("pause_after_setup").value)

        self.list_client = self.create_client(
            ListControllers,
            "/controller_manager/list_controllers",
        )
        self.switch_client = self.create_client(
            SwitchController,
            "/controller_manager/switch_controller",
        )
        self.unpause_client = self.create_client(Empty, "/unpause_physics")
        self.pause_client = self.create_client(Empty, "/pause_physics")
        self.trajectory_pub = self.create_publisher(
            JointTrajectory,
            JOINT_TRAJECTORY_TOPIC,
            10,
        )

    def run(self) -> bool:
        if not self._wait_for_clients():
            return False

        if not self._wait_for_loaded_controllers():
            return False

        # Controller activation in gazebo_ros2_control needs Gazebo's update loop.
        if not self._call_empty(self.unpause_client, "unpause Gazebo physics"):
            return False
        time.sleep(0.5)

        if not self._activate_controllers():
            return False

        self._publish_home()
        time.sleep(0.5)

        if self.pause_after_setup:
            if not self._call_empty(self.pause_client, "pause Gazebo physics"):
                return False
            self.get_logger().info("Controllers active; Gazebo paused for inspection.")
        else:
            self.get_logger().info("Controllers active; Gazebo left unpaused.")
        return True

    def _wait_for_clients(self, timeout_sec: float = 30.0) -> bool:
        clients = [
            (self.list_client, "/controller_manager/list_controllers"),
            (self.switch_client, "/controller_manager/switch_controller"),
            (self.unpause_client, "/unpause_physics"),
            (self.pause_client, "/pause_physics"),
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
            loaded_names = {controller.name for controller in controllers}
            if all(name in loaded_names for name in CONTROLLERS):
                return True
            self.get_logger().info(
                f"Waiting for controllers to load. Loaded: {sorted(loaded_names)}"
            )
            time.sleep(0.5)
        self.get_logger().error(f"Timed out waiting for controllers: {CONTROLLERS}")
        return False

    def _list_controllers(self, timeout_sec: float) -> list:
        future = self.list_client.call_async(ListControllers.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.done() and future.result() is not None:
            return list(future.result().controller)
        return []

    def _activate_controllers(self) -> bool:
        controllers = self._list_controllers(timeout_sec=5.0)
        active = {controller.name for controller in controllers if controller.state == "active"}
        if all(name in active for name in CONTROLLERS):
            self.get_logger().info("Controllers are already active.")
            return True

        request = SwitchController.Request()
        request.activate_controllers = CONTROLLERS
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
            self.get_logger().error("Controller manager rejected controller activation.")
            return False

        self.get_logger().info("Activated UR3e Gazebo controllers.")
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
        self.get_logger().info("Published stable home trajectory.")

    def _call_empty(self, client, description: str, timeout_sec: float = 10.0) -> bool:
        future = client.call_async(Empty.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        if future.done() and future.result() is not None:
            return True
        self.get_logger().error(f"Timed out trying to {description}.")
        return False


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

