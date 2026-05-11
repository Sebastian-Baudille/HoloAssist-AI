#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from gazebo_msgs.msg import LinkStates, ModelStates
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


MODEL_STATES_TOPICS = ("/model_states", "/gazebo/model_states")
LINK_STATES_TOPICS = ("/link_states", "/gazebo/link_states")

CUBE_POSE_TOPIC = "/cube/pose"
GOAL_POSE_TOPIC = "/goal/pose"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"

DEFAULT_CUBE_XYZ = (0.1, -0.40, 1.11)
DEFAULT_GOAL_XYZ = (0.28, 0.0, 1.078)


class GazeboPoseBridge(Node):
    """Publish the simple pose topics used by the RL environment.

    Some Gazebo Classic setups publish /model_states and /link_states; others
    expose only a smaller set of Gazebo services. For the first PPO reach task,
    the cube and goal are static, so this bridge publishes configured fallback
    poses and uses TF for the robot TCP pose.
    """

    def __init__(self) -> None:
        super().__init__("gazebo_pose_bridge")
        self.declare_parameter("cube_model_name", "cube_1")
        self.declare_parameter("goal_model_name", "bin_1")
        self.declare_parameter(
            "tcp_link_names",
            [
                "ur3e_rg2::gripper_tcp",
                "ur3e_rg2::tool0",
                "ur3e_rg2::wrist_3_link",
            ],
        )
        self.declare_parameter("tcp_frame_names", ["gripper_tcp", "tool0", "wrist_3_link"])
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("cube_fallback_xyz", list(DEFAULT_CUBE_XYZ))
        self.declare_parameter("goal_fallback_xyz", list(DEFAULT_GOAL_XYZ))
        self.declare_parameter("publish_rate_hz", 20.0)

        self.cube_model_name = str(self.get_parameter("cube_model_name").value)
        self.goal_model_name = str(self.get_parameter("goal_model_name").value)
        self.tcp_link_names = list(self.get_parameter("tcp_link_names").value)
        self.tcp_frame_names = list(self.get_parameter("tcp_frame_names").value)
        self.world_frame = str(self.get_parameter("world_frame").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))

        cube_fallback_xyz = self._xyz_parameter("cube_fallback_xyz", DEFAULT_CUBE_XYZ)
        goal_fallback_xyz = self._xyz_parameter("goal_fallback_xyz", DEFAULT_GOAL_XYZ)

        self.cube_pub = self.create_publisher(PoseStamped, CUBE_POSE_TOPIC, 10)
        self.goal_pub = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)
        self.tcp_pub = self.create_publisher(PoseStamped, TCP_POSE_TOPIC, 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.cube_pose = self._pose_from_xyz(cube_fallback_xyz, self.world_frame)
        self.goal_pose = self._pose_from_xyz(goal_fallback_xyz, self.world_frame)
        self.tcp_pose: PoseStamped | None = None

        for topic in MODEL_STATES_TOPICS:
            self.create_subscription(ModelStates, topic, self._model_states_cb, 10)
        for topic in LINK_STATES_TOPICS:
            self.create_subscription(LinkStates, topic, self._link_states_cb, 10)

        self._warned_missing_cube = False
        self._warned_missing_goal = False
        self._warned_missing_tcp = False
        self._warned_missing_tf = False

        self.create_timer(1.0 / publish_rate_hz, self._publish_timer_cb)
        self.get_logger().info(
            "Publishing RL pose topics: "
            f"{TCP_POSE_TOPIC}, {CUBE_POSE_TOPIC}, {GOAL_POSE_TOPIC}"
        )

    def _model_states_cb(self, msg: ModelStates) -> None:
        self._publish_model_pose(
            msg,
            self.cube_model_name,
            self.cube_pub,
            "cube",
        )
        self._publish_model_pose(
            msg,
            self.goal_model_name,
            self.goal_pub,
            "goal",
        )

    def _link_states_cb(self, msg: LinkStates) -> None:
        name_to_pose = dict(zip(msg.name, msg.pose))
        for link_name in self.tcp_link_names:
            pose = name_to_pose.get(link_name)
            if pose is not None:
                self.tcp_pose = self._pose_stamped(pose, self.world_frame)
                return

        if not self._warned_missing_tcp:
            self.get_logger().warning(
                f"Could not find TCP link in Gazebo link states. Tried: {self.tcp_link_names}"
            )
            self._warned_missing_tcp = True

    def _publish_model_pose(
        self,
        msg: ModelStates,
        model_name: str,
        publisher,
        label: str,
    ) -> None:
        try:
            index = msg.name.index(model_name)
        except ValueError:
            if label == "cube" and not self._warned_missing_cube:
                self.get_logger().warning(f"Could not find cube model [{model_name}]")
                self._warned_missing_cube = True
            elif label == "goal" and not self._warned_missing_goal:
                self.get_logger().warning(f"Could not find goal model [{model_name}]")
                self._warned_missing_goal = True
            return

        pose = self._pose_stamped(msg.pose[index], self.world_frame)
        if label == "cube":
            self.cube_pose = pose
        else:
            self.goal_pose = pose
        publisher.publish(pose)

    def _publish_timer_cb(self) -> None:
        now = self.get_clock().now().to_msg()
        self.cube_pose.header.stamp = now
        self.goal_pose.header.stamp = now
        self.cube_pub.publish(self.cube_pose)
        self.goal_pub.publish(self.goal_pose)

        tcp_pose = self._tcp_pose_from_tf()
        if tcp_pose is not None:
            self.tcp_pose = tcp_pose

        if self.tcp_pose is not None:
            self.tcp_pose.header.stamp = now
            self.tcp_pub.publish(self.tcp_pose)

    def _tcp_pose_from_tf(self) -> PoseStamped | None:
        for frame_name in self.tcp_frame_names:
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.world_frame,
                    str(frame_name),
                    Time(),
                )
            except TransformException:
                continue

            stamped = PoseStamped()
            stamped.header = transform.header
            stamped.pose.position.x = transform.transform.translation.x
            stamped.pose.position.y = transform.transform.translation.y
            stamped.pose.position.z = transform.transform.translation.z
            stamped.pose.orientation = transform.transform.rotation
            return stamped

        if not self._warned_missing_tf:
            self.get_logger().warning(
                f"Could not find TCP TF from {self.world_frame} to any of "
                f"{self.tcp_frame_names} yet."
            )
            self._warned_missing_tf = True
        return None

    def _pose_stamped(self, pose, frame_id: str) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = frame_id
        stamped.pose = pose
        return stamped

    def _pose_from_xyz(self, xyz: tuple[float, float, float], frame_id: str) -> PoseStamped:
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = frame_id
        stamped.pose.position.x = xyz[0]
        stamped.pose.position.y = xyz[1]
        stamped.pose.position.z = xyz[2]
        stamped.pose.orientation.w = 1.0
        return stamped

    def _xyz_parameter(
        self,
        parameter_name: str,
        fallback: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        value = list(self.get_parameter(parameter_name).value)
        if len(value) != 3:
            self.get_logger().warning(
                f"Parameter {parameter_name} must have three values; using {fallback}."
            )
            return fallback
        return (float(value[0]), float(value[1]), float(value[2]))


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
