from __future__ import annotations

import os
import subprocess
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

from ur3e_rl_env.constants import (
    BIN_POSITION_X,
    BIN_POSITION_Y,
    BIN_POSITION_Z,
    OBSERVATION_SIZE_13D,
    WORKSPACE_HEIGHT_M,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)


GAZEBO_WORLD_NAME = os.getenv("UR3E_RL_GAZEBO_WORLD_NAME", "ur3e_pick_place_world")
CUBE_MODEL_NAME = os.getenv("UR3E_RL_CUBE_MODEL_NAME", "cube_1")
CUBE_RESET_POSE_TOPIC = "/gazebo_pose_bridge/cube_reset_pose"

JOINT_STATE_TOPIC = "/joint_states"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
CUBE_POSE_TOPICS = (
    "/cube_0/pose",
    "/cube_1/pose",
    "/cube_2/pose",
    "/cube_3/pose",
)
LEGACY_CUBE_POSE_TOPIC = "/cube/pose"
GOAL_POSE_TOPIC = "/goal/pose"
COLLISION_FLAG_TOPIC = "/collision_flag"
JOINT_TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
GRIPPER_TRAJECTORY_TOPIC = "/finger_width_trajectory_controller/joint_trajectory"
GRIPPER_JOINT_NAME = "finger_width"
GRIPPER_OPEN_WIDTH_MM = 85.0
GRIPPER_CLOSED_WIDTH_MM = 0.0
GRIPPER_CLOSED_THRESHOLD_MM = 30.0
GRASP_PROXIMITY_THRESHOLD_M = 0.03

UR3E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

OBSERVATION_SIZE = OBSERVATION_SIZE_13D
BIN_POSITION = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z], dtype=np.float32)


def pose_to_array(msg: PoseStamped) -> np.ndarray:
    return np.array(
        [
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ],
        dtype=np.float32,
    )


def _normalize_axis(value: float, min_value: float, max_value: float) -> float:
    span = max(max_value - min_value, 1e-6)
    normalized = 2.0 * ((value - min_value) / span) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    return np.array(
        [
            _normalize_axis(float(xyz[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
            _normalize_axis(float(xyz[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
            _normalize_axis(float(xyz[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
        ],
        dtype=np.float32,
    )


def build_observation(
    state: Mapping[str, Any],
    step_count: int = 0,
    max_episode_steps: int = 200,
) -> np.ndarray:
    """Builds the normalized 13D PPO observation vector."""

    ee_position = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    object_position = np.asarray(state["object_position"], dtype=np.float32).reshape(3)
    bin_position = np.asarray(
        state.get("goal_position", BIN_POSITION),
        dtype=np.float32,
    ).reshape(3)

    grasped = 1.0 if float(state.get("grasped", 0.0)) > 0.5 else 0.0
    gripper_state = 1.0 if float(state.get("gripper_state", 0.0)) > 0.5 else 0.0
    ee_height_norm = float(np.clip(float(ee_position[2]) / WORKSPACE_HEIGHT_M, -1.0, 1.0))
    timestep_norm = float(
        np.clip(float(step_count) / max(float(max_episode_steps), 1.0), 0.0, 1.0)
    )

    observation = np.array(
        [
            *_normalize_xyz(ee_position),
            *_normalize_xyz(object_position),
            *_normalize_xyz(bin_position),
            grasped,
            gripper_state,
            ee_height_norm,
            timestep_norm,
        ],
        dtype=np.float32,
    )
    observation = np.clip(observation, -1.0, 1.0).astype(np.float32)

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
        self.cube_positions_by_topic: dict[str, np.ndarray] = {}
        self.goal_position: np.ndarray | None = None
        self.collision_flag = False
        self.gripper_width_mm = GRIPPER_OPEN_WIDTH_MM
        self._collision_received = False
        self._warned_no_traj_subscriber = False

        self.create_subscription(JointState, JOINT_STATE_TOPIC, self._joint_state_cb, 10)
        self.create_subscription(PoseStamped, TCP_POSE_TOPIC, self._tcp_pose_cb, 10)
        for topic in CUBE_POSE_TOPICS:
            self.create_subscription(
                PoseStamped,
                topic,
                lambda msg, topic_name=topic: self._cube_pose_cb(topic_name, msg),
                10,
            )
        self.create_subscription(
            PoseStamped,
            LEGACY_CUBE_POSE_TOPIC,
            lambda msg: self._cube_pose_cb(LEGACY_CUBE_POSE_TOPIC, msg),
            10,
        )
        self.create_subscription(PoseStamped, GOAL_POSE_TOPIC, self._goal_pose_cb, 10)
        self.create_subscription(Bool, COLLISION_FLAG_TOPIC, self._collision_cb, 10)
        self.trajectory_publisher = self.create_publisher(
            JointTrajectory,
            JOINT_TRAJECTORY_TOPIC,
            10,
        )
        self.cube_reset_publisher = self.create_publisher(
            PoseStamped, CUBE_RESET_POSE_TOPIC, 10
        )
        self.gripper_trajectory_publisher = self.create_publisher(
            JointTrajectory,
            GRIPPER_TRAJECTORY_TOPIC,
            10,
        )

    def _joint_state_cb(self, msg: JointState) -> None:
        for index, name in enumerate(msg.name):
            if name not in UR3E_JOINT_NAMES:
                if name == GRIPPER_JOINT_NAME and index < len(msg.position):
                    width_value = float(msg.position[index])
                    # In simulation this is typically meters. Convert to mm.
                    self.gripper_width_mm = width_value * 1000.0 if width_value <= 0.2 else width_value
                continue
            if index < len(msg.position):
                self.joint_positions_by_name[name] = float(msg.position[index])
            if index < len(msg.velocity):
                self.joint_velocities_by_name[name] = float(msg.velocity[index])
            else:
                self.joint_velocities_by_name.setdefault(name, 0.0)

    def _tcp_pose_cb(self, msg: PoseStamped) -> None:
        self.end_effector_position = pose_to_array(msg)

    def _cube_pose_cb(self, topic_name: str, msg: PoseStamped) -> None:
        self.cube_positions_by_topic[topic_name] = pose_to_array(msg)

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
        msg.joint_names = list(UR3E_JOINT_NAMES)

        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in target_array]
        point.time_from_start = Duration(seconds=float(duration_sec)).to_msg()
        msg.points.append(point)

        self.trajectory_publisher.publish(msg)

    def open_gripper(self) -> None:
        self._publish_gripper_width_mm(GRIPPER_OPEN_WIDTH_MM)

    def close_gripper(self) -> None:
        self._publish_gripper_width_mm(GRIPPER_CLOSED_WIDTH_MM)

    def get_gripper_width(self) -> float:
        return float(self.gripper_width_mm)

    def _publish_gripper_width_mm(self, width_mm: float) -> None:
        width_mm = float(np.clip(width_mm, GRIPPER_CLOSED_WIDTH_MM, GRIPPER_OPEN_WIDTH_MM))
        width_m = width_mm / 1000.0

        msg = JointTrajectory()
        msg.joint_names = [GRIPPER_JOINT_NAME]
        point = JointTrajectoryPoint()
        point.positions = [width_m]
        point.time_from_start = Duration(seconds=0.3).to_msg()
        msg.points.append(point)
        self.gripper_trajectory_publisher.publish(msg)

    def reset_cube_position(self, xyz: tuple[float, float, float]) -> None:
        stamped = PoseStamped()
        stamped.header.stamp = self.get_clock().now().to_msg()
        stamped.header.frame_id = "world"
        stamped.pose.position.x = float(xyz[0])
        stamped.pose.position.y = float(xyz[1])
        stamped.pose.position.z = float(xyz[2])
        stamped.pose.orientation.w = 1.0
        self.cube_reset_publisher.publish(stamped)

        req = (
            f'name: "{CUBE_MODEL_NAME}" '
            f'position: {{x: {xyz[0]:.4f}, y: {xyz[1]:.4f}, z: {xyz[2]:.4f}}} '
            f'orientation: {{w: 1.0}}'
        )
        try:
            subprocess.run(
                [
                    "ign", "service",
                    "-s", f"/world/{GAZEBO_WORLD_NAME}/set_pose",
                    "--reqtype", "ignition.msgs.Pose",
                    "--reptype", "ignition.msgs.Boolean",
                    "--timeout", "2000",
                    "--req", req,
                ],
                timeout=3.0,
                capture_output=True,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            self.get_logger().warning(f"Cube Gazebo teleport failed: {exc}")

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

        object_position = self._current_object_position().copy()
        gripper_width_mm = self.get_gripper_width()
        gripper_closed = gripper_width_mm < GRIPPER_CLOSED_THRESHOLD_MM
        grasped = (
            float(np.linalg.norm(self.end_effector_position - object_position))
            < GRASP_PROXIMITY_THRESHOLD_M
            and gripper_closed
        )

        return {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "end_effector_position": self.end_effector_position.copy(),
            "object_position": object_position,
            "goal_position": self._current_goal_position().copy(),
            "grasped": 1.0 if grasped else 0.0,
            "gripper_state": 1.0 if gripper_closed else 0.0,
            "gripper_width_mm": gripper_width_mm,
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
        if not self.cube_positions_by_topic:
            missing.extend(CUBE_POSE_TOPICS)
        if self.goal_position is None:
            missing.append(f"{GOAL_POSE_TOPIC} (optional; falls back to cube pose)")
        return missing

    def _has_joint_state(self) -> bool:
        return all(name in self.joint_positions_by_name for name in UR3E_JOINT_NAMES)

    def _has_full_state(self) -> bool:
        return (
            self._has_joint_state()
            and self.end_effector_position is not None
            and bool(self.cube_positions_by_topic)
        )

    def _current_goal_position(self) -> np.ndarray:
        if self.goal_position is not None:
            return self.goal_position
        if self.cube_positions_by_topic:
            return self._current_object_position()
        raise RuntimeError("Cannot build goal position before cube pose has been received.")

    def _current_object_position(self) -> np.ndarray:
        ee = self.end_effector_position
        if ee is None:
            raise RuntimeError("Cannot choose target cube before end effector pose is available.")

        # Prefer the four explicit cube topics, then fall back to legacy /cube/pose.
        candidates: list[np.ndarray] = []
        for topic in CUBE_POSE_TOPICS:
            cube = self.cube_positions_by_topic.get(topic)
            if cube is not None:
                candidates.append(cube)
        legacy = self.cube_positions_by_topic.get(LEGACY_CUBE_POSE_TOPIC)
        if not candidates and legacy is not None:
            candidates.append(legacy)
        if not candidates:
            raise RuntimeError("No cube poses available.")

        return min(
            candidates,
            key=lambda cube: float(np.linalg.norm(cube - ee)),
        )
