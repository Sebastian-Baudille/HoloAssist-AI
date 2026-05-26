#!/usr/bin/env python3
"""Publish the simple pose topics the RL environment expects.

TCP pose is read from TF, while cube poses are read live from Gazebo's
`/world/<name>/dynamic_pose/info` topic bridged through ros_gz_bridge.
"""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener


GOAL_POSE_TOPIC = "/goal/pose"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
LEGACY_CUBE_POSE_TOPIC = "/cube/pose"

DEFAULT_WORLD_NAME = "ur3e_pick_place_world"
DEFAULT_CUBE_FRAME_NAMES = ("cube_1", "cube_2", "cube_3", "cube_4")
DEFAULT_CUBE_XYZS = (
    (0.1, -0.40, 1.11),
    (0.1, -0.25, 1.11),
    (-0.1, -0.25, 1.11),
    (-0.1, -0.40, 1.11),
)
DEFAULT_GOAL_XYZ = (0.28, 0.0, 1.078)


class GazeboPoseBridge(Node):

    def __init__(self) -> None:
        super().__init__("gazebo_pose_bridge")
        self.declare_parameter("world_name", DEFAULT_WORLD_NAME)
        self.declare_parameter("tcp_frame_names", ["gripper_tcp", "tool0", "wrist_3_link"])
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("cube_model_names", list(DEFAULT_CUBE_FRAME_NAMES))
        self.declare_parameter("cube_0_xyz", list(DEFAULT_CUBE_XYZS[0]))
        self.declare_parameter("cube_1_xyz", list(DEFAULT_CUBE_XYZS[1]))
        self.declare_parameter("cube_2_xyz", list(DEFAULT_CUBE_XYZS[2]))
        self.declare_parameter("cube_3_xyz", list(DEFAULT_CUBE_XYZS[3]))
        self.declare_parameter("goal_xyz", list(DEFAULT_GOAL_XYZ))
        self.declare_parameter("publish_rate_hz", 20.0)

        world_name = str(self.get_parameter("world_name").value)
        self.tcp_frame_names = list(self.get_parameter("tcp_frame_names").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.dynamic_pose_topic = f"/world/{world_name}/dynamic_pose/info"

        frame_names_param = list(self.get_parameter("cube_model_names").value)
        self.cube_frame_names = tuple(str(name) for name in frame_names_param[:4])
        if len(self.cube_frame_names) != 4:
            self.get_logger().warning(
                "cube_model_names must define 4 names; using defaults."
            )
            self.cube_frame_names = DEFAULT_CUBE_FRAME_NAMES

        # Explicit, stable mapping from Gazebo TF child_frame_id to published
        # topics /cube_0..3/pose:
        # cube_1 -> /cube_0/pose, cube_2 -> /cube_1/pose, ...
        self.cube_index_by_frame = {
            self.cube_frame_names[0]: 0,
            self.cube_frame_names[1]: 1,
            self.cube_frame_names[2]: 2,
            self.cube_frame_names[3]: 3,
        }

        cube_fallbacks = (
            self._xyz_param("cube_0_xyz", DEFAULT_CUBE_XYZS[0]),
            self._xyz_param("cube_1_xyz", DEFAULT_CUBE_XYZS[1]),
            self._xyz_param("cube_2_xyz", DEFAULT_CUBE_XYZS[2]),
            self._xyz_param("cube_3_xyz", DEFAULT_CUBE_XYZS[3]),
        )
        goal_xyz = self._xyz_param("goal_xyz", DEFAULT_GOAL_XYZ)

        self.cube_pubs = [
            self.create_publisher(PoseStamped, f"/cube_{index}/pose", 10)
            for index in range(4)
        ]
        self.legacy_cube_pub = self.create_publisher(PoseStamped, LEGACY_CUBE_POSE_TOPIC, 10)
        self.goal_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)
        self.tcp_pub = self.create_publisher(PoseStamped, TCP_POSE_TOPIC, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_poses_by_index: dict[int, PoseStamped] = {}
        for index in range(4):
            self.cube_poses_by_index[index] = self._pose_from_xyz(cube_fallbacks[index])

        self.goal_pose = self._pose_from_xyz(goal_xyz)
        self.tcp_pose: PoseStamped | None = None
        self._warned_tf = False

        self.create_subscription(TFMessage, self.dynamic_pose_topic, self._dynamic_pose_cb, 10)
        self.create_subscription(PoseStamped, "~/cube_reset_pose", self._cube_reset_cb, 10)
        self.create_timer(1.0 / publish_rate_hz, self._publish)
        self.get_logger().info(
            "Publishing RL pose topics: "
            f"{TCP_POSE_TOPIC}, /cube_0..3/pose, {GOAL_POSE_TOPIC} "
            f"(dynamic source: {self.dynamic_pose_topic})"
        )

    def _publish(self) -> None:
        now = self.get_clock().now().to_msg()

        self.goal_pose.header.stamp = now
        self.goal_pub.publish(self.goal_pose)

        closest_cube: PoseStamped | None = None
        closest_distance = float("inf")
        tcp_xyz = None
        if self.tcp_pose is not None:
            tcp_xyz = (
                self.tcp_pose.pose.position.x,
                self.tcp_pose.pose.position.y,
                self.tcp_pose.pose.position.z,
            )

        for index in range(4):
            cube_pose = self.cube_poses_by_index.get(index)
            if cube_pose is None:
                continue
            cube_pose.header.stamp = now
            if not cube_pose.header.frame_id:
                cube_pose.header.frame_id = self.world_frame
            self.cube_pubs[index].publish(cube_pose)

            if tcp_xyz is not None:
                dx = cube_pose.pose.position.x - tcp_xyz[0]
                dy = cube_pose.pose.position.y - tcp_xyz[1]
                dz = cube_pose.pose.position.z - tcp_xyz[2]
                dist = dx * dx + dy * dy + dz * dz
                if dist < closest_distance:
                    closest_distance = dist
                    closest_cube = cube_pose
            elif index == 0:
                closest_cube = cube_pose

        if closest_cube is not None:
            self.legacy_cube_pub.publish(closest_cube)

        tcp = self._tcp_from_tf()
        if tcp is not None:
            self.tcp_pose = tcp
        if self.tcp_pose is not None:
            self.tcp_pose.header.stamp = now
            self.tcp_pub.publish(self.tcp_pose)

    def _dynamic_pose_cb(self, msg: TFMessage) -> None:
        for transform in msg.transforms:
            cube_index = self._cube_index_from_frame(transform.child_frame_id)
            if cube_index is None:
                continue

            stamped = PoseStamped()
            stamped.header.stamp = transform.header.stamp
            stamped.header.frame_id = transform.header.frame_id or self.world_frame
            stamped.pose.position.x = transform.transform.translation.x
            stamped.pose.position.y = transform.transform.translation.y
            stamped.pose.position.z = transform.transform.translation.z
            stamped.pose.orientation = transform.transform.rotation
            self.cube_poses_by_index[cube_index] = stamped

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
        # Fallback only: explicit reset writes the primary cube pose
        # until live /dynamic_pose/info updates arrive.
        self.cube_poses_by_index[0] = msg

    def _cube_index_from_frame(self, frame_id: str) -> int | None:
        if not frame_id:
            return None
        frame = frame_id.strip("/")
        if frame in self.cube_index_by_frame:
            return self.cube_index_by_frame[frame]

        # Gazebo may use scoped names like cube_1::link.
        scoped_root = frame.split("::", 1)[0]
        if scoped_root in self.cube_index_by_frame:
            return self.cube_index_by_frame[scoped_root]

        # Handle nested namespaced forms like /world/.../cube_1.
        for token in frame.split("/"):
            if token in self.cube_index_by_frame:
                return self.cube_index_by_frame[token]
        return None

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
