#!/usr/bin/env python3
"""
visualize_ik.py — Move the robot to IK reference poses and verify visually in RViz.

Usage (with sim already running):
  cd ur3e_rl_ws
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/visualize_ik.py                    # 6 representative positions
  python3 scripts/visualize_ik.py --cube 0.0 -0.3 1.11
  python3 scripts/visualize_ik.py --random 10

For each cube position the script:
  1. Computes an IK reference configuration using ur3e_rl_env.kinematics
     (elbow-up, gripper pointing straight down, TCP 7 cm above cube)
  2. Moves the robot to home first (clean starting point)
  3. Moves the robot to the IK configuration
  4. Publishes markers: yellow sphere at the approach target, red sphere at cube
  5. Reads the actual TCP position from /tcp_pose_broadcaster/pose
  6. Reports the error between IK-predicted and actual TCP position
  7. Waits for Enter before the next position

What to look for in RViz:
  - The gripper TCP should land ON the yellow sphere (< 3 cm is good)
  - The gripper fingers should be pointing straight DOWN
  - The arm should be in an elbow-UP configuration (the "hump" shape)
  - No self-collisions or extreme joint limits
"""

from __future__ import annotations
import argparse
import numpy as np

from ur3e_rl_env.kinematics import (
    compute_ik_reference,
    forward_kinematics,
    fk_tcp_z_axis,
    DEFAULT_APPROACH_HEIGHT,
)

HOME = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])


# ── ROS node ──────────────────────────────────────────────────────────────────

def run(cube_positions: list) -> None:
    import rclpy
    from rclpy.node import Node
    from geometry_msgs.msg import PoseStamped
    from visualization_msgs.msg import Marker
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from rclpy.duration import Duration

    rclpy.init()

    class IKVisNode(Node):
        def __init__(self):
            super().__init__("ik_visualizer")
            self.traj_pub = self.create_publisher(
                JointTrajectory,
                "/scaled_joint_trajectory_controller/joint_trajectory",
                10,
            )
            self.marker_pub = self.create_publisher(Marker, "/ik_target_marker", 10)
            self.tcp_pos = None
            self.create_subscription(
                PoseStamped, "/tcp_pose_broadcaster/pose", self._tcp_cb, 10
            )

        def _tcp_cb(self, msg: PoseStamped) -> None:
            self.tcp_pos = np.array([
                msg.pose.position.x,
                msg.pose.position.y,
                msg.pose.position.z,
            ])

        def send_joints(self, joints: np.ndarray, duration: float = 3.0) -> None:
            msg = JointTrajectory()
            msg.joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint",      "wrist_2_joint",       "wrist_3_joint",
            ]
            pt = JointTrajectoryPoint()
            pt.positions = [float(j) for j in joints]
            pt.time_from_start = Duration(seconds=float(duration)).to_msg()
            msg.points.append(pt)
            self.traj_pub.publish(msg)

        def publish_marker(
            self,
            pos: np.ndarray,
            color: tuple[float, float, float] = (1.0, 1.0, 0.0),
            marker_id: int = 0,
        ) -> None:
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.id = marker_id
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.9
            m.lifetime.sec = 60
            self.marker_pub.publish(m)

        def spin_for(self, secs: float) -> None:
            deadline = self.get_clock().now().nanoseconds + int(secs * 1e9)
            while self.get_clock().now().nanoseconds < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

    node = IKVisNode()

    print("\nIK Visualizer ready.")
    print("Open RViz with:  rviz2 -d scripts/training_monitor.rviz")
    print("Add a Marker display subscribed to /ik_target_marker for the target spheres.")
    print("Yellow sphere = approach target (7 cm above cube).")
    print("Red sphere    = cube centre.")
    print("Press Ctrl+C to quit.\n")

    for i, cube in enumerate(cube_positions):
        cube   = np.asarray(cube, dtype=float)
        target = cube + np.array([0.0, 0.0, DEFAULT_APPROACH_HEIGHT])

        print(f"─── Position {i+1}/{len(cube_positions)} ───────────────────────────")
        print(f"  Cube world:     ({cube[0]:+.3f}, {cube[1]:+.3f}, {cube[2]:.3f})")
        print(f"  Approach target:({target[0]:+.3f}, {target[1]:+.3f}, {target[2]:.3f})")

        # Solve IK
        reachable, joints, fk_err = compute_ik_reference(cube)
        ik_tcp   = forward_kinematics(joints)
        ik_zaxis = fk_tcp_z_axis(joints)
        orient_err = float(np.linalg.norm(ik_zaxis - np.array([0., 0., -1.])))

        print(f"  IK reachable:   {reachable}  (FK error {fk_err*100:.1f} cm)")
        print(f"  Joints (deg):   {np.degrees(joints).round(1).tolist()}")
        print(f"  IK FK TCP:      ({ik_tcp[0]:+.3f}, {ik_tcp[1]:+.3f}, {ik_tcp[2]:.3f})")
        print(f"  Gripper Z-axis: ({ik_zaxis[0]:+.3f}, {ik_zaxis[1]:+.3f}, {ik_zaxis[2]:+.3f})"
              f"  orient err {orient_err:.3f}  ({'pointing down ✓' if orient_err < 0.35 else 'NOT pointing down ✗'})")

        if not reachable:
            print("  WARNING: IK did not converge — skipping this position.")
            continue

        # Publish markers
        node.publish_marker(target, color=(1.0, 1.0, 0.0), marker_id=0)
        node.publish_marker(cube,   color=(1.0, 0.2, 0.2), marker_id=1)

        # Home first
        print("  Moving to home...", end="", flush=True)
        node.send_joints(HOME, duration=2.0)
        node.spin_for(2.8)
        print(" done")

        # IK configuration
        print("  Moving to IK pose...", end="", flush=True)
        node.send_joints(joints, duration=3.0)
        node.spin_for(4.0)
        print(" done")

        # Compare actual vs predicted TCP
        if node.tcp_pos is not None:
            actual_err = float(np.linalg.norm(node.tcp_pos - target))
            print(f"\n  Predicted TCP:  ({ik_tcp[0]:+.4f}, {ik_tcp[1]:+.4f}, {ik_tcp[2]:.4f})")
            print(f"  Actual TCP:     ({node.tcp_pos[0]:+.4f}, {node.tcp_pos[1]:+.4f}, {node.tcp_pos[2]:.4f})")
            print(f"  TCP error:      {actual_err*100:.1f} cm  "
                  f"{'✓ good' if actual_err < 0.05 else '✗ off — FK model may need tuning'}")
        else:
            print("  (no TCP reading — is /tcp_pose_broadcaster/pose publishing?)")

        input("\n  [Enter] for next position, Ctrl+C to quit...\n")

    print("Done.")
    node.destroy_node()
    rclpy.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize IK reference poses in RViz")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument(
        "--cube", nargs=3, type=float, metavar=("X", "Y", "Z"),
        help="Single cube position in world frame",
    )
    grp.add_argument(
        "--random", type=int, default=0, metavar="N",
        help="N random positions from the spawn zone (seed=42 for reproducibility)",
    )
    args = parser.parse_args()

    if args.cube:
        positions = [args.cube]
    elif args.random > 0:
        rng = np.random.default_rng(42)
        positions = [
            [rng.uniform(-0.20, 0.20), rng.uniform(-0.45, -0.10), 1.11]
            for _ in range(args.random)
        ]
    else:
        # Default: 6 representative positions covering the spawn zone
        positions = [
            [ 0.00, -0.30, 1.11],   # centre
            [ 0.15, -0.30, 1.11],   # right
            [-0.15, -0.30, 1.11],   # left
            [ 0.00, -0.40, 1.11],   # far
            [ 0.10, -0.15, 1.11],   # near base
            [-0.10, -0.40, 1.11],   # far-left
        ]

    run(positions)


if __name__ == "__main__":
    main()
