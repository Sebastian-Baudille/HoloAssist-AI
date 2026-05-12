"""Record teleoperation demonstrations for seeding RL training.

Run alongside a teleop node (e.g. Unity XR on Quest 3). The recorder
passively observes robot state at a fixed rate and saves (obs, action,
reward, next_obs, done) transitions compatible with the PPO pipeline.

TCP pose is read from TF (base_link -> tool0), so it works with any UR
driver stack without extra publishers. Object/goal positions can come
from ROS topics (e.g. perception) or static params.

Usage:
    ros2 run ur3e_rl_env record_demo

    # Custom object position (no perception running):
    ros2 run ur3e_rl_env record_demo --ros-args \
        -p object_xyz:="[0.3, 0.0, 0.05]"

    # Custom TF frames and sample rate:
    ros2 run ur3e_rl_env record_demo --ros-args \
        -p base_frame:=base -p tcp_frame:=tool0 -p hz:=10.0

Environment variables:
    UR3E_DEMO_OUTPUT_DIR    Output directory (default: ./demo_data)
    UR3E_DEMO_HZ            Sampling rate (default: 5.0)
    UR3E_DEMO_MAX_STEPS     Max steps per episode (default: 200)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from tf2_ros import Buffer, TransformException, TransformListener

from ur3e_rl_env.reward import check_failure, check_success, compute_reward
from ur3e_rl_env.ros_interface import (
    OBSERVATION_SIZE,
    UR3E_JOINT_NAMES,
    build_observation,
)


OUTPUT_DIR = os.getenv("UR3E_DEMO_OUTPUT_DIR", "./demo_data")
SAMPLE_HZ = float(os.getenv("UR3E_DEMO_HZ", "5.0"))
MAX_EPISODE_STEPS = int(os.getenv("UR3E_DEMO_MAX_STEPS", "200"))
ACTION_LOW = -0.03
ACTION_HIGH = 0.03


def _next_episode_number(output_dir: Path) -> int:
    existing = sorted(output_dir.glob("demo_episode_*.npz"))
    if not existing:
        return 1
    return int(existing[-1].stem.rsplit("_", 1)[-1]) + 1


class DemoRecorderNode(Node):

    def __init__(self) -> None:
        super().__init__("demo_recorder")

        self.declare_parameter("output_dir", OUTPUT_DIR)
        self.declare_parameter("hz", SAMPLE_HZ)
        self.declare_parameter("max_episode_steps", MAX_EPISODE_STEPS)
        self.declare_parameter("base_frame", "base_link")
        self.declare_parameter("tcp_frame", "tool0")
        self.declare_parameter("tcp_pose_topic", "/tcp_pose_broadcaster/pose")
        self.declare_parameter("object_topic", "/cube/pose")
        self.declare_parameter("object_xyz", [0.3, 0.0, 0.05])
        self.declare_parameter("goal_topic", "/goal/pose")
        self.declare_parameter("collision_topic", "/collision_flag")
        self.declare_parameter("velocity_topic", "/forward_velocity_controller/commands")

        self.output_dir = Path(str(self.get_parameter("output_dir").value))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        hz = float(self.get_parameter("hz").value)
        self.dt = 1.0 / hz
        self.max_steps = int(self.get_parameter("max_episode_steps").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.tcp_frame = str(self.get_parameter("tcp_frame").value)

        object_xyz = list(self.get_parameter("object_xyz").value)
        self.static_object_pos = np.array(object_xyz, dtype=np.float32)

        # --- live state ---
        self.joint_positions: dict[str, float] = {}
        self.joint_velocities: dict[str, float] = {}
        self.tcp_position: np.ndarray | None = None
        self._tcp_from_topic = False
        self.object_position: np.ndarray | None = None
        self.goal_position: np.ndarray | None = None
        self.collision_flag = False
        self.last_velocity_cmd: np.ndarray | None = None

        # --- TF (fallback for TCP when topic is not available) ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._warned_tf = False

        # --- subscriptions ---
        self.create_subscription(JointState, "/joint_states", self._joint_state_cb, 10)

        tcp_pose_topic = str(self.get_parameter("tcp_pose_topic").value)
        self.create_subscription(PoseStamped, tcp_pose_topic, self._tcp_pose_cb, 10)

        object_topic = str(self.get_parameter("object_topic").value)
        self.create_subscription(PoseStamped, object_topic, self._object_pose_cb, 10)

        goal_topic = str(self.get_parameter("goal_topic").value)
        self.create_subscription(PoseStamped, goal_topic, self._goal_pose_cb, 10)

        collision_topic = str(self.get_parameter("collision_topic").value)
        self.create_subscription(Bool, collision_topic, self._collision_cb, 10)

        velocity_topic = str(self.get_parameter("velocity_topic").value)
        self.create_subscription(
            Float64MultiArray, velocity_topic, self._velocity_cmd_cb, 10
        )

        # --- episode buffers ---
        self._obs_buf: list[np.ndarray] = []
        self._act_buf: list[np.ndarray] = []
        self._rew_buf: list[float] = []
        self._next_obs_buf: list[np.ndarray] = []
        self._term_buf: list[bool] = []
        self._vel_cmd_buf: list[np.ndarray] = []

        self._prev_joints: np.ndarray | None = None
        self._prev_obs: np.ndarray | None = None
        self._step_count = 0
        self._total_transitions = 0
        self._episodes_saved = 0
        self._clipped_count = 0
        self._total_actions = 0
        self._idle_ticks = 0
        self.episode_num = _next_episode_number(self.output_dir)
        self._ready_logged = False

        self.create_timer(self.dt, self._tick)

        self.get_logger().info(
            f"Recording demos at {hz} Hz → {self.output_dir}/\n"
            f"  TCP: {tcp_pose_topic} (fallback: TF {self.base_frame} → {self.tcp_frame})\n"
            f"  Object topic: {object_topic} (fallback: {object_xyz})\n"
            f"  Velocity topic: {velocity_topic}\n"
            f"  Max steps/episode: {self.max_steps}\n"
            f"  Press Ctrl+C to save and quit."
        )

    # ------------------------------------------------------------------ #
    #  Callbacks                                                          #
    # ------------------------------------------------------------------ #

    def _joint_state_cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name not in UR3E_JOINT_NAMES:
                continue
            if i < len(msg.position):
                self.joint_positions[name] = float(msg.position[i])
            if i < len(msg.velocity):
                self.joint_velocities[name] = float(msg.velocity[i])
            else:
                self.joint_velocities.setdefault(name, 0.0)

    def _tcp_pose_cb(self, msg: PoseStamped) -> None:
        self.tcp_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )
        self._tcp_from_topic = True

    def _object_pose_cb(self, msg: PoseStamped) -> None:
        self.object_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

    def _goal_pose_cb(self, msg: PoseStamped) -> None:
        self.goal_position = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            dtype=np.float32,
        )

    def _collision_cb(self, msg: Bool) -> None:
        self.collision_flag = bool(msg.data)

    def _velocity_cmd_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) >= 6:
            self.last_velocity_cmd = np.array(msg.data[:6], dtype=np.float32)

    # ------------------------------------------------------------------ #
    #  TF lookup                                                          #
    # ------------------------------------------------------------------ #

    def _lookup_tcp(self) -> np.ndarray | None:
        try:
            tf = self.tf_buffer.lookup_transform(
                self.base_frame, self.tcp_frame, Time()
            )
            return np.array(
                [
                    tf.transform.translation.x,
                    tf.transform.translation.y,
                    tf.transform.translation.z,
                ],
                dtype=np.float32,
            )
        except TransformException:
            if not self._warned_tf:
                self.get_logger().warning(
                    f"TF {self.base_frame} → {self.tcp_frame} not available yet."
                )
                self._warned_tf = True
            return None

    # ------------------------------------------------------------------ #
    #  State assembly (matches RL env's build_observation format)          #
    # ------------------------------------------------------------------ #

    def _has_joints(self) -> bool:
        return all(name in self.joint_positions for name in UR3E_JOINT_NAMES)

    def _get_state(self) -> dict[str, Any] | None:
        if not self._has_joints():
            return None

        if not self._tcp_from_topic:
            tcp = self._lookup_tcp()
            if tcp is not None:
                self.tcp_position = tcp
        if self.tcp_position is None:
            return None

        object_pos = (
            self.object_position
            if self.object_position is not None
            else self.static_object_pos
        )
        goal_pos = (
            self.goal_position if self.goal_position is not None else object_pos
        )

        return {
            "joint_positions": np.array(
                [self.joint_positions[n] for n in UR3E_JOINT_NAMES],
                dtype=np.float32,
            ),
            "joint_velocities": np.array(
                [self.joint_velocities.get(n, 0.0) for n in UR3E_JOINT_NAMES],
                dtype=np.float32,
            ),
            "end_effector_position": self.tcp_position.copy(),
            "object_position": object_pos.copy(),
            "goal_position": goal_pos.copy(),
            "grasped": 0.0,
            "collision_flag": self.collision_flag,
        }

    # ------------------------------------------------------------------ #
    #  Recording timer                                                    #
    # ------------------------------------------------------------------ #

    def _tick(self) -> None:
        state = self._get_state()
        if state is None:
            return

        if not self._ready_logged:
            self.get_logger().info("State available — recording started.")
            self._ready_logged = True

        joints = state["joint_positions"].copy()
        obs = build_observation(state)

        if self._prev_joints is None or self._prev_obs is None:
            self._prev_joints = joints
            self._prev_obs = obs
            return

        raw_delta = joints - self._prev_joints
        action = np.clip(raw_delta, ACTION_LOW, ACTION_HIGH).astype(np.float32)

        self._total_actions += 1
        if not np.allclose(raw_delta, action):
            self._clipped_count += 1

        if np.allclose(raw_delta, 0.0, atol=1e-5):
            self._idle_ticks += 1
            if self._idle_ticks == int(5.0 / self.dt):
                self.get_logger().warning(
                    "Robot idle for 5 s — is the teleop node active?"
                )
        else:
            self._idle_ticks = 0

        reward = compute_reward(state, action, self._step_count)
        success = check_success(state)
        failure = check_failure(state)
        terminal = success or failure

        self._obs_buf.append(self._prev_obs)
        self._act_buf.append(action)
        self._rew_buf.append(reward)
        self._next_obs_buf.append(obs)
        self._term_buf.append(terminal)
        self._vel_cmd_buf.append(
            self.last_velocity_cmd.copy()
            if self.last_velocity_cmd is not None
            else np.zeros(6, dtype=np.float32)
        )

        self._step_count += 1
        self._prev_joints = joints
        self._prev_obs = obs

        if terminal:
            reason = "success" if success else "failure"
            self._save_episode(reason)
        elif self._step_count >= self.max_steps:
            self._save_episode("max_steps")

    # ------------------------------------------------------------------ #
    #  Persistence                                                        #
    # ------------------------------------------------------------------ #

    def _save_episode(self, reason: str) -> None:
        if not self._obs_buf:
            self._reset_episode()
            return

        path = self.output_dir / f"demo_episode_{self.episode_num:03d}.npz"
        np.savez_compressed(
            path,
            observations=np.array(self._obs_buf, dtype=np.float32),
            actions=np.array(self._act_buf, dtype=np.float32),
            rewards=np.array(self._rew_buf, dtype=np.float32),
            next_observations=np.array(self._next_obs_buf, dtype=np.float32),
            terminals=np.array(self._term_buf, dtype=bool),
            velocity_commands=np.array(self._vel_cmd_buf, dtype=np.float32),
        )
        n = len(self._obs_buf)
        self._total_transitions += n
        self._episodes_saved += 1
        self.get_logger().info(
            f"Episode {self.episode_num:03d}: {n} steps, "
            f"ended={reason} → {path.name}"
        )
        self.episode_num += 1
        self._reset_episode()

    def _reset_episode(self) -> None:
        self._obs_buf.clear()
        self._act_buf.clear()
        self._rew_buf.clear()
        self._next_obs_buf.clear()
        self._term_buf.clear()
        self._vel_cmd_buf.clear()
        self._step_count = 0
        self._prev_joints = None
        self._prev_obs = None
        self._idle_ticks = 0

    def finish(self) -> None:
        if self._obs_buf:
            self._save_episode("interrupted")

        clip_pct = 100.0 * self._clipped_count / max(self._total_actions, 1)
        self.get_logger().info(
            f"Done. {self._episodes_saved} episodes, "
            f"{self._total_transitions} transitions → {self.output_dir}/\n"
            f"  Clipped: {self._clipped_count}/{self._total_actions} "
            f"({clip_pct:.1f}%)"
        )
        if clip_pct > 10.0:
            self.get_logger().warning(
                f"High clipping rate ({clip_pct:.0f}%). "
                f"Consider increasing the hz param (currently {1.0 / self.dt:.0f})."
            )


def main() -> None:
    rclpy.init()
    node = DemoRecorderNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.finish()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
