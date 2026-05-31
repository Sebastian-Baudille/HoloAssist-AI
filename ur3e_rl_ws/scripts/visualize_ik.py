#!/usr/bin/env python3
"""
visualize_ik.py — Move the robot to IK reference poses and verify visually in RViz.

Usage (with sim already running):
  source /opt/ros/humble/setup.bash && source install/setup.bash
  python3 scripts/visualize_ik.py
  python3 scripts/visualize_ik.py --cube 0.0 -0.3 1.11
  python3 scripts/visualize_ik.py --random 10

For each cube position the script:
  1. Computes an IK reference configuration (elbow-up, gripper pointing down)
  2. Moves the robot home first (so motion is clean)
  3. Moves the robot to the IK configuration
  4. Publishes a marker at the TARGET position (7 cm above cube) so you can
     compare it to where the gripper actually ends up in RViz
  5. Prints the TCP position from /tcp_pose_broadcaster/pose (ground truth FK)
  6. Waits for you to press Enter before the next position

Watch in RViz: the gripper TCP (green axes) should be near the yellow sphere marker.
If the robot self-collides or the TCP is far off, the IK seed needs adjustment.
"""

from __future__ import annotations
import argparse
import sys
import time
import numpy as np

# ── Inline IK (no import needed — same logic as ik_reference.py) ────────────

_DH_A     = np.array([0.0,      -0.24355, -0.21320, 0.0,      0.0,      0.0    ])
_DH_D     = np.array([0.15185,   0.0,      0.0,     0.13105,  0.08535,  0.09210])
_DH_ALPHA = np.array([np.pi/2,   0.0,      0.0,     np.pi/2, -np.pi/2,  0.0    ])
GRIPPER_LENGTH = 0.218
BASE_Z = 0.82
L1 = 0.24355
L2 = 0.21320
APPROACH_HEIGHT = 0.07
FK_Z_OFFSET = 0.27   # analytic FK underestimates Z by this much — applied to IK target


def _dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [ 0,     sa,     ca,    d],
                     [ 0,      0,      0,    1]])


def analytical_fk(joints):
    """Approximate FK (used for seeding — ground truth comes from ROS)."""
    T = np.eye(4)
    T[2, 3] = BASE_Z
    for i in range(6):
        T = T @ _dh(joints[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
    T_grip = np.eye(4); T_grip[2, 3] = GRIPPER_LENGTH
    T = T @ T_grip
    return T[:3, 3]


def ik_seed(cube_pos):
    """Elbow-up planar seed configuration."""
    tcp_target = cube_pos + np.array([0, 0, APPROACH_HEIGHT])

    # Align plane with cube: yaw=0 so robot faces -Y (where cubes are)
    pan = np.arctan2(cube_pos[1], cube_pos[0])

    wrist_drop = _DH_D[3] + _DH_D[4] + _DH_D[5] + GRIPPER_LENGTH
    r_horiz    = float(np.sqrt(cube_pos[0]**2 + cube_pos[1]**2))
    dz         = (tcp_target[2] + wrist_drop) - (BASE_Z + _DH_D[0])
    dr         = r_horiz
    reach      = float(np.clip(np.sqrt(dr**2 + dz**2),
                               abs(L1 - L2) + 1e-4, L1 + L2 - 1e-4))

    cos_elbow = float(np.clip((reach**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1))
    elbow     = -np.arccos(cos_elbow)
    alpha     = np.arctan2(dz, dr)
    beta      = np.arctan2(L2*np.sin(-elbow), L1 + L2*np.cos(elbow))
    lift      = alpha - beta
    wrist1    = -(lift + elbow) - np.pi / 2.0

    return np.array([pan, lift, elbow, wrist1, 0.0, 0.0])


def ik_solve(cube_pos):
    """Numerical IK with multiple seeds."""
    from scipy.optimize import minimize
    target = cube_pos + np.array([0, 0, APPROACH_HEIGHT])

    seeds = [
        ik_seed(cube_pos),
        np.array([0, -np.pi/3, np.pi/2, -np.pi/6, -np.pi/2, 0]),
        np.array([np.pi/4, -np.pi/4, np.pi/3, -np.pi/6, -np.pi/2, 0]),
        np.array([-np.pi/4, -np.pi/3, np.pi/2, -np.pi/6, -np.pi/2, 0]),
    ]

    bounds = [(-2*np.pi, 2*np.pi), (-np.pi, -0.2), (-2.5, np.pi),
              (-np.pi, 0.0), (-0.15, 0.15), (-np.pi, np.pi)]

    analytic_target = target - np.array([0.0, 0.0, FK_Z_OFFSET])

    def fk_z_axis(q):
        """TCP Z-axis direction (for top-down: should be [0,0,-1])."""
        T = np.eye(4)
        for i in range(6):
            T = T @ _dh(q[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
        Tg = np.eye(4); Tg[2,3] = GRIPPER_LENGTH
        return (T @ Tg)[:3, 2]

    def cost(q):
        pos_err    = np.linalg.norm(analytical_fk(q) - analytic_target)
        orient_err = np.linalg.norm(fk_z_axis(q) - np.array([0.,0.,-1.]))
        return pos_err**2 + 0.5*orient_err**2

    # Expand seeds with more variants
    pan = seeds[0][0]
    seeds += [
        np.array([pan, -np.pi/2,  np.pi/3, -np.pi/6, 0., 0.]),
        np.array([pan, -2.0,      np.pi/2, -np.pi/6, 0., 0.]),
        np.array([pan+np.pi, -np.pi/3, np.pi/2, -np.pi/6, 0., 0.]),
    ]

    best_joints, best_err = None, float("inf")
    for seed in seeds:
        r = minimize(cost, seed, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': 400, 'ftol': 1e-12})
        err = np.linalg.norm(analytical_fk(r.x) - analytic_target)
        if err < best_err:
            best_err = err
            best_joints = r.x

    return best_joints, best_err


# ── ROS node ──────────────────────────────────────────────────────────────────

def run(cube_positions):
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
                JointTrajectory, "/scaled_joint_trajectory_controller/joint_trajectory", 10)
            self.marker_pub = self.create_publisher(Marker, "/ik_target_marker", 10)
            self.tcp_pos = None
            self.create_subscription(PoseStamped, "/tcp_pose_broadcaster/pose",
                                     self._tcp_cb, 10)

        def _tcp_cb(self, msg):
            self.tcp_pos = np.array([msg.pose.position.x,
                                      msg.pose.position.y,
                                      msg.pose.position.z])

        def send_joints(self, joints, duration=3.0):
            msg = JointTrajectory()
            msg.joint_names = [
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ]
            pt = JointTrajectoryPoint()
            pt.positions = [float(j) for j in joints]
            pt.time_from_start = Duration(seconds=float(duration)).to_msg()
            msg.points.append(pt)
            self.traj_pub.publish(msg)

        def publish_marker(self, pos, color=(1.0, 1.0, 0.0)):
            m = Marker()
            m.header.frame_id = "world"
            m.header.stamp = self.get_clock().now().to_msg()
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(pos[0])
            m.pose.position.y = float(pos[1])
            m.pose.position.z = float(pos[2])
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r, m.color.g, m.color.b = color
            m.color.a = 0.9
            m.lifetime.sec = 30
            self.marker_pub.publish(m)

        def spin_briefly(self, secs=0.2):
            deadline = self.get_clock().now().nanoseconds + int(secs * 1e9)
            while self.get_clock().now().nanoseconds < deadline:
                rclpy.spin_once(self, timeout_sec=0.05)

    HOME = np.array([0.0, -np.pi/2, 0.0, -np.pi/2, 0.0, 0.0])
    node = IKVisNode()

    print("\nIK Visualizer ready.")
    print("Make sure RViz is open with the training_monitor.rviz config.")
    print("Add a 'Marker' display subscribed to /ik_target_marker to see the target sphere.")
    print("Press Ctrl+C to quit.\n")

    for i, cube in enumerate(cube_positions):
        cube = np.asarray(cube, dtype=float)
        target = cube + np.array([0, 0, APPROACH_HEIGHT])

        print(f"─── Position {i+1}/{len(cube_positions)} ──────────────────────")
        print(f"  Cube:   {cube}")
        print(f"  Target: {target}  ({APPROACH_HEIGHT*100:.0f} cm above cube)")

        joints, analytic_err = ik_solve(cube)
        print(f"  IK seed error (analytic): {analytic_err*100:.1f} cm")
        print(f"  Joints (deg): {np.degrees(joints).round(1)}")

        # Publish yellow sphere at target
        node.publish_marker(target, color=(1.0, 1.0, 0.0))
        node.publish_marker(cube,   color=(1.0, 0.2, 0.2))  # red at cube

        # Move to home first
        print("  Moving to home...", end="", flush=True)
        node.send_joints(HOME, duration=2.0)
        node.spin_briefly(2.5)
        print(" done")

        # Move to IK pose
        print("  Moving to IK configuration...", end="", flush=True)
        node.send_joints(joints, duration=3.0)
        node.spin_briefly(3.5)
        print(" done")

        # Read actual TCP
        node.spin_briefly(0.5)
        if node.tcp_pos is not None:
            err = np.linalg.norm(node.tcp_pos - target)
            print(f"  Actual TCP:    {node.tcp_pos.round(4)}")
            print(f"  Target:        {target.round(4)}")
            print(f"  TCP→target:    {err*100:.1f} cm  {'✓ good' if err < 0.05 else '✗ off'}")
        else:
            print("  (no TCP reading yet)")

        input("\n  [Enter] for next position, Ctrl+C to quit...")
        print()

    print("Done.")
    node.destroy_node()
    rclpy.shutdown()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualize IK reference poses in RViz")
    grp = parser.add_mutually_exclusive_group()
    grp.add_argument("--cube", nargs=3, type=float, metavar=("X", "Y", "Z"),
                     help="Single cube position in world frame")
    grp.add_argument("--random", type=int, default=0, metavar="N",
                     help="N random positions from the spawn zone")
    args = parser.parse_args()

    if args.cube:
        positions = [args.cube]
    elif args.random > 0:
        np.random.seed(42)
        positions = []
        for _ in range(args.random):
            x = np.random.uniform(-0.20, 0.20)
            y = np.random.uniform(-0.45, -0.10)
            positions.append([x, y, 1.11])
    else:
        # Default: a spread of representative positions
        positions = [
            [ 0.00, -0.30, 1.11],  # centre
            [ 0.15, -0.30, 1.11],  # right
            [-0.15, -0.30, 1.11],  # left
            [ 0.00, -0.40, 1.11],  # far
            [ 0.10, -0.15, 1.11],  # near base
            [-0.10, -0.40, 1.11],  # far-left
        ]

    run(positions)


if __name__ == "__main__":
    main()
