#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
UR3E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Match Gazebo's default spawned joint pose. The RL environment can move away
# from this after startup; this first command just gives the controller a stable
# reference before physics is unpaused.
HOME_JOINTS = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


class HomeTrajectoryPublisher(Node):
    def __init__(self) -> None:
        super().__init__("send_home_trajectory")
        self.publisher = self.create_publisher(
            JointTrajectory,
            JOINT_TRAJECTORY_TOPIC,
            10,
        )

    def wait_for_controller(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.publisher.get_subscription_count() > 0:
                return True
        return False

    def publish_home(self) -> None:
        msg = JointTrajectory()
        msg.joint_names = UR3E_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = HOME_JOINTS
        point.velocities = [0.0] * len(HOME_JOINTS)
        point.time_from_start.sec = 1
        msg.points.append(point)

        for _ in range(5):
            self.publisher.publish(msg)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HomeTrajectoryPublisher()
    try:
        if not node.wait_for_controller():
            node.get_logger().warning(
                f"No subscriber on {JOINT_TRAJECTORY_TOPIC}; skipping initial home command."
            )
            return
        node.publish_home()
        node.get_logger().info("Published initial home trajectory before unpausing Gazebo.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

