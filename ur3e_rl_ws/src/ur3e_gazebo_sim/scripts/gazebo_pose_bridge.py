#!/usr/bin/env python3
"""Publish the simple pose topics the RL environment expects.

For Ignition Fortress the cube and goal positions are static (set from
parameters), and the TCP pose is read from the TF tree published by
robot_state_publisher.  No Gazebo-specific messages are needed.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


CUBE_POSE_TOPIC = "/cube/pose"
GOAL_POSE_TOPIC = "/goal/pose"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"

DEFAULT_CUBE_XYZ = (0.1, -0.40, 1.11)
DEFAULT_GOAL_XYZ = (0.28, 0.0, 1.078)


class GazeboPoseBridge(Node):

    def __init__(self) -> None:
        super().__init__("gazebo_pose_bridge")
        self.declare_parameter("tcp_frame_names", ["gripper_tcp", "tool0", "wrist_3_link"])
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("cube_xyz", list(DEFAULT_CUBE_XYZ))
        self.declare_parameter("goal_xyz", list(DEFAULT_GOAL_XYZ))
        self.declare_parameter("publish_rate_hz", 20.0)

        self.tcp_frame_names = list(self.get_parameter("tcp_frame_names").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))

        cube_xyz = self._xyz_param("cube_xyz", DEFAULT_CUBE_XYZ)
        goal_xyz = self._xyz_param("goal_xyz", DEFAULT_GOAL_XYZ)

        self.cube_pub = self.create_publisher(PoseStamped, CUBE_POSE_TOPIC, 10)
        self.goal_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)
        self.tcp_pub = self.create_publisher(PoseStamped, TCP_POSE_TOPIC, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose = self._pose_from_xyz(cube_xyz)
        self.goal_pose = self._pose_from_xyz(goal_xyz)
        self.tcp_pose: PoseStamped | None = None
        self._warned_tf = False

        self.create_subscription(PoseStamped, "~/cube_reset_pose", self._cube_reset_cb, 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish)
        self.get_logger().info(
            f"Publishing RL pose topics: {TCP_POSE_TOPIC}, {CUBE_POSE_TOPIC}, {GOAL_POSE_TOPIC}"
        )

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()

        self.cube_pose.header.stamp = now
        self.goal_pose.header.stamp = now
        self.cube_pub.publish(self.cube_pose)
        self.goal_pub.publish(self.goal_pose)

        tcp = self._tcp_from_tf()
        if tcp is not None:
            self.tcp_pose = tcp
        if self.tcp_pose is not None:
            self.tcp_pose.header.stamp = now
            self.tcp_pub.publish(self.tcp_pose)

    def _tcp_from_tf(self) -> PoseStamped | None:
        for frame in self.tcp_frame_names:
            try:
                tf = self.tf_buffer.lookup_transform(self.world_frame, frame, Time())
            except TransformException:
                continue
            stamped = PoseStamped()
            stamped.header = tf.header
            stamped.pose.position.x = tf.transform.translation.x
            stamped.pose.position.y = tf.transform.translation.y
            stamped.pose.position.z = tf.transform.translation.z
            stamped.pose.orientation = tf.transform.rotation
            return stamped

        if not self._warned_tf:
            self.get_logger().warning(
                f"TF {self.world_frame} → {self.tcp_frame_names} not available yet."
            )
            self._warned_tf = True
        return None

    def _cube_reset_cb(self, msg: PoseStamped) -> None:
        self.cube_pose = msg

    def _pose_from_xyz(self, xyz: tuple[float, float, float]) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = self.world_frame
        stamped.pose.position.x = xyz[0]
        stamped.pose.position.y = xyz[1]
        stamped.pose.position.z = xyz[2]
        stamped.pose.orientation.w = 1.0
        return stamped

    def _xyz_param(
        self, name: str, fallback: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        val = list(self.get_parameter(name).value)
        if len(val) != 3:
            self.get_logger().warning(f"{name} must have 3 values; using {fallback}")
            return fallback
        return (float(val[0]), float(val[1]), float(val[2]))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = GazeboPoseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
