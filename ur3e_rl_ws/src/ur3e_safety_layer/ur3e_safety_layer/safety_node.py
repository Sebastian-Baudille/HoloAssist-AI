from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool

from ur3e_safety_layer.safety_checker import SafetyChecker


COLLISION_FLAG_TOPIC = "/collision_flag"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
SAFETY_STATUS_TOPIC = "/safety_layer/is_safe"


class SafetyNode(Node):
    """Publishes a simple safety status from collision and TCP height checks."""

    def __init__(self) -> None:
        super().__init__("ur3e_safety_node")
        self.checker = SafetyChecker()
        self.collision_flag = False
        self.end_effector_position: tuple[float, float, float] | None = None

        self.create_subscription(Bool, COLLISION_FLAG_TOPIC, self._collision_cb, 10)
        self.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._tcp_pose_cb, 10)
        self.publisher = self.create_publisher(Bool, SAFETY_STATUS_TOPIC, 10)
        self.create_timer(0.1, self._publish_status)

    def _collision_cb(self, msg: Bool) -> None:
        self.collision_flag = bool(msg.data)

    def _tcp_pose_cb(self, msg: PoseStamped) -> None:
        self.end_effector_position = (
            float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
        )

    def _publish_status(self) -> None:
        if self.end_effector_position is None:
            safe = False
        else:
            state = {
                "collision_flag": self.collision_flag,
                "end_effector_position": self.end_effector_position,
            }
            safe = self.checker.check_state(state).safe

        msg = Bool()
        msg.data = safe
        self.publisher.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SafetyNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

