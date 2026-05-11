from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


JOINT_STATE_TOPIC = "/joint_states"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
CUBE_POSE_TOPIC = "/cube/pose"
GOAL_POSE_TOPIC = "/goal/pose"
COLLISION_FLAG_TOPIC = "/collision_flag"
JOINT_TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"

UR3E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

OBSERVATION_SIZE = 29


def pose_to_array(msg: PoseStamped) -> np.ndarray:
    return np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=np.float32,
    )


def build_observation(state: Mapping[str, Any]) -> np.ndarray:
    """Builds the 29-value PPO observation vector from the latest ROS state."""

    joint_positions = np.asarray(state["joint_positions"], dtype=np.float32).reshape(6)
    joint_velocities = np.asarray(state["joint_velocities"], dtype=np.float32).reshape(6)
    ee_position = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    object_position = np.asarray(state["object_position"], dtype=np.float32).reshape(3)
    goal_position = np.asarray(state["goal_position"], dtype=np.float32).reshape(3)
    object_minus_ee = object_position - ee_position
    goal_minus_object = goal_position - object_position
    grasped_flag = np.array([float(state.get("grasped", 0.0))], dtype=np.float32)
    collision_flag = np.array([float(bool(state.get("collision_flag", False)))], dtype=np.float32)

    observation = np.concatenate(
        [
            joint_positions,
            joint_velocities,
            ee_position,
            object_position,
            goal_position,
            object_minus_ee,
            goal_minus_object,
            grasped_flag,
            collision_flag,
        ]
    ).astype(np.float32)

    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Expected observation shape {(OBSERVATION_SIZE,)}, got {observation.shape}")
    return observation


class RosInterfaceNode(Node):
    """ROS 2 interface used by the Gymnasium env, smoke tests, and policy node."""

    def __init__(self, node_name: str = "ur3e_rl_ros_interface") -> None:
        super().__init__(node_name)

        self.joint_positions_by_name: dict[str, float] = {}
        self.joint_velocities_by_name: dict[str, float] = {}
        self.end_effector_position: np.ndarray | None = None
        self.object_position: np.ndarray | None = None
        self.goal_position: np.ndarray | None = None
        self.collision_flag = False
        self._collision_received = False
        self._warned_no_traj_subscriber = False

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._tcp_pose_cb, 10)
        self.create_subscription(PoseStamped, CUBE_POSE_TOPIC, self._cube_pose_cb, 10)
        self.create_subscription(PoseStamped, GOAL_POSE_TOPIC, self._goal_pose_cb, 10)
        self.create_subscription(Bool, COLLISION_FLAG_TOPIC, self._collision_cb, 10)
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            JOINT_TRAJECTORY_TOPIC,
            10,
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if name not in UR3E_JOINT_NAMES:
                continue
            if index < len(msg.position):
                self.joint_positions_by_name[name] = float(msg.position[index])
            if index < len(msg.velocity):
                self.joint_velocities_by_name[name] = float(msg.velocity[index])
            else:
                self.joint_velocities_by_name.setdefault(name, 0.0)

    def _tcp_pose_cb(self, msg: PoseStamped) -> None:
        self.end_effector_position = pose_to_array(msg)

    def _cube_pose_cb(self, msg: PoseStamped) -> None:
        self.object_position = pose_to_array(msg)

    def _goal_pose_cb(self, msg: PoseStamped) -> None:
        self.goal_position = pose_to_array(msg)

    def _collision_cb(self, msg: Bool) -> None:
        self.collision_flag = bool(msg.data)
        self._collision_received = True

    def wait_for_joint_state(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._has_joint_state():
                return True
        self.get_logger().warning(
            f"Timed out waiting for all UR3e joints on {JOINT_STATE_TOPIC}."
        )
        return False

    def wait_until_ready(self, timeout_sec: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self._has_full_state():
                if not self._collision_received:
                    self.get_logger().warning(
                        f"No {COLLISION_FLAG_TOPIC} message yet; using collision_flag=False."
                    )
                if self.goal_position is None:
                    self.get_logger().warning(
                        f"No {GOAL_POSE_TOPIC} message yet; using cube pose as the reach goal."
                    )
                return True

        missing = ", ".join(self.missing_state_fields()) or "unknown"
        self.get_logger().warning(f"Timed out waiting for RL state. Missing: {missing}")
        return False

    def send_joint_target(
        self,
        target_joints: Sequence[float],
        duration_sec: float = 0.2,
    ) -> None:
        target_array = np.asarray(target_joints, dtype=np.float32).reshape(6)

        if (
            self.trajectory_publisher.get_subscription_count() == 0
            and not self._warned_no_traj_subscriber
        ):
            self.get_logger().warning(
                f"No subscribers on {JOINT_TRAJECTORY_TOPIC}. Is the joint trajectory "
                "controller running?"
            )
            self._warned_no_traj_subscriber = True

        msg = JointTrajectory()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = list(UR3E_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in target_array]
        point.velocities = [0.0] * 6
        point.time_from_start = Duration(seconds=float(duration_sec)).to_msg()
        msg.points.append(point)

        self.trajectory_publisher.publish(msg)

    def get_state(self) -> dict[str, Any] | None:
        if not self._has_full_state():
            return None

        joint_positions = np.array(
            [self.joint_positions_by_name[name] for name in UR3E_JOINT_NAMES],
            dtype=np.float32,
        )
        joint_velocities = np.array(
            [self.joint_velocities_by_name.get(name, 0.0) for name in UR3E_JOINT_NAMES],
            dtype=np.float32,
        )

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": self.end_effector_position.copy(),
            "object_position": self.object_position.copy(),
            "goal_position": self._current_goal_position().copy(),
            "grasped": 0.0,
            "collision_flag": self.collision_flag,
        }

    def get_observation(self) -> np.ndarray | None:
        state = self.get_state()
        if state is None:
            return None
        return build_observation(state)

    def missing_state_fields(self) -> list[str]:
        missing: list[str] = []
        if not self._has_joint_state():
            missing.append(JOINT_STATE_TOPIC)
        if self.end_effector_position is None:
            missing.append(TCP_POSE_TOPIC)
        if self.object_position is None:
            missing.append(CUBE_POSE_TOPIC)
        if self.goal_position is None:
            missing.append(f"{GOAL_POSE_TOPIC} (optional; falls back to cube pose)")
        return missing

    def _has_joint_state(self) -> bool:
        return all(name in self.joint_positions_by_name for name in UR3E_JOINT_NAMES)

    def _has_full_state(self) -> bool:
        return (
            self._has_joint_state()
            and self.end_effector_position is not None
            and self.object_position is not None
        )

    def _current_goal_position(self) -> np.ndarray:
        if self.goal_position is not None:
            return self.goal_position
        if self.object_position is not None:
            return self.object_position
        raise RuntimeError("Cannot build goal position before cube pose has been received.")
