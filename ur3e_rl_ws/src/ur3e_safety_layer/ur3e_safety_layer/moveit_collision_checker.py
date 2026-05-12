from __future__ import annotations

import time

import rclpy
from moveit_msgs.srv import GetStateValidity
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool


COLLISION_FLAG_TOPIC = "/collision_flag"
JOINT_STATES_TOPIC = "/joint_states"
CHECK_STATE_VALIDITY_SERVICE = "/check_state_validity"
DEFAULT_GROUP_NAME = "ur_onrobot_manipulator"


class MoveItCollisionChecker(Node):
    """Publish collision flags by querying MoveIt's state validity service."""

    def __init__(self) -> None:
        super().__init__("moveit_collision_checker")

        self.declare_parameter("collision_flag_topic", COLLISION_FLAG_TOPIC)
        self.declare_parameter("joint_states_topic", JOINT_STATES_TOPIC)
        self.declare_parameter("check_state_validity_service", CHECK_STATE_VALIDITY_SERVICE)
        self.declare_parameter("move_group_name", DEFAULT_GROUP_NAME)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("joint_state_stale_timeout_sec", 0.5)
        self.declare_parameter("request_timeout_sec", 0.25)
        self.declare_parameter("fail_closed_when_unavailable", False)
        self.declare_parameter("fail_closed_without_joint_state", False)

        collision_flag_topic = str(self.get_parameter("collision_flag_topic").value)
        joint_states_topic = str(self.get_parameter("joint_states_topic").value)
        service_name = str(self.get_parameter("check_state_validity_service").value)
        self.group_name = str(self.get_parameter("move_group_name").value)
        publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self.joint_state_stale_timeout_sec = float(
            self.get_parameter("joint_state_stale_timeout_sec").value
        )
        self.request_timeout_sec = float(self.get_parameter("request_timeout_sec").value)
        self.fail_closed_when_unavailable = bool(
            self.get_parameter("fail_closed_when_unavailable").value
        )
        self.fail_closed_without_joint_state = bool(
            self.get_parameter("fail_closed_without_joint_state").value
        )

        self.publisher = self.create_publisher(Bool, collision_flag_topic, 10)
        self.create_subscription(JointState, joint_states_topic, self._joint_state_cb, 20)
        self.client = self.create_client(GetStateValidity, service_name)
        self.timer = self.create_timer(1.0 / publish_rate_hz, self._on_timer)

        self.latest_joint_state: JointState | None = None
        self.latest_joint_state_time: float = 0.0
        self._pending_future = None
        self._pending_requested_at: float = 0.0
        self._last_warn_ts: dict[str, float] = {}

        self.get_logger().info(
            "MoveIt collision checker active: "
            f"topic={collision_flag_topic}, joints={joint_states_topic}, "
            f"service={service_name}, group={self.group_name}"
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        self.latest_joint_state = msg
        self.latest_joint_state_time = time.monotonic()

    def _warn_throttled(self, key: str, text: str, min_period_sec: float = 5.0) -> None:
        now = time.monotonic()
        prev = self._last_warn_ts.get(key, 0.0)
        if now - prev >= min_period_sec:
            self._last_warn_ts[key] = now
            self.get_logger().warn(text)

    def _publish_collision(self, in_collision: bool) -> None:
        msg = Bool()
        msg.data = bool(in_collision)
        self.publisher.publish(msg)

    def _make_request(self, joint_state: JointState) -> GetStateValidity.Request:
        req = GetStateValidity.Request()
        req.robot_state.joint_state = joint_state
        req.group_name = self.group_name
        return req

    def _handle_pending_future(self) -> bool:
        if self._pending_future is None:
            return False

        if self._pending_future.done():
            try:
                response = self._pending_future.result()
                if response is None:
                    self._publish_collision(self.fail_closed_when_unavailable)
                    self._warn_throttled(
                        "null_response",
                        "MoveIt returned an empty response for /check_state_validity.",
                    )
                else:
                    self._publish_collision(not bool(response.valid))
            except Exception as exc:
                self._publish_collision(self.fail_closed_when_unavailable)
                self._warn_throttled(
                    "service_exception",
                    f"/check_state_validity call failed: {exc}",
                )
            finally:
                self._pending_future = None
            return True

        elapsed = time.monotonic() - self._pending_requested_at
        if elapsed > self.request_timeout_sec:
            try:
                self._pending_future.cancel()
            except Exception:
                pass
            self._pending_future = None
            self._publish_collision(self.fail_closed_when_unavailable)
            self._warn_throttled(
                "request_timeout",
                f"/check_state_validity request timed out after {elapsed:.2f}s.",
            )
            return True

        return True

    def _on_timer(self) -> None:
        if self._handle_pending_future():
            return

        if not self.client.service_is_ready():
            self._publish_collision(self.fail_closed_when_unavailable)
            self._warn_throttled(
                "service_not_ready",
                "MoveIt /check_state_validity service not available; "
                "publishing fallback collision flag.",
            )
            return

        if self.latest_joint_state is None:
            self._publish_collision(self.fail_closed_without_joint_state)
            self._warn_throttled(
                "no_joint_state",
                "No /joint_states received yet; publishing fallback collision flag.",
            )
            return

        age = time.monotonic() - self.latest_joint_state_time
        if age > self.joint_state_stale_timeout_sec:
            self._publish_collision(self.fail_closed_without_joint_state)
            self._warn_throttled(
                "stale_joint_state",
                f"/joint_states is stale ({age:.2f}s old); publishing fallback collision flag.",
            )
            return

        request = self._make_request(self.latest_joint_state)
        self._pending_future = self.client.call_async(request)
        self._pending_requested_at = time.monotonic()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MoveItCollisionChecker()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
