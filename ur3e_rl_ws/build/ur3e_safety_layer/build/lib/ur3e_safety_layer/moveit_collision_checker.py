from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


COLLISION_FLAG_TOPIC = "/collision_flag"


class MoveItCollisionChecker(Node):
    """Placeholder collision publisher until a real MoveIt checker is connected.

    The first PPO reach task only needs a collision flag in the observation. This
    node publishes ``False`` at 10 Hz so the rest of the RL stack can run while
    the real MoveIt collision checker is added later.
    """

    def __init__(self) -> None:
        super().__init__("moveit_collision_checker")
        self.publisher = self.create_publisher(Bool, COLLISION_FLAG_TOPIC, 10)
        self.create_timer(0.1, self._publish_no_collision)
        self.get_logger().warn(
            "Publishing collision_flag=False placeholder. Replace this with a real "
            "MoveIt planning-scene collision checker before using the policy on hardware."
        )

    def _publish_no_collision(self) -> None:
        msg = Bool()
        msg.data = False
        self.publisher.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveItCollisionChecker()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

