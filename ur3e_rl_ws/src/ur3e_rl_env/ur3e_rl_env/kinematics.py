"""
kinematics.py — UR3e forward kinematics and IK reference solver.

Computes joint angles that place the gripper TCP directly above a cube,
pointing straight down (top-down approach for grasping).

World-frame FK
--------------
The robot is spawned at (0, 0, ROBOT_BASE_Z) with robot_yaw=π in Gazebo,
but the /tcp_pose_broadcaster/pose topic publishes in the Gazebo world frame
where only the Z offset applies — the yaw rotation is NOT reflected in the
published TCP coordinates.  Empirically confirmed: the actual TCP and the
DH-chain FK (with Z offset only, no yaw rotation) agree to < 1 mm.

    T_world = T_Z_offset × T_DH_chain × T_gripper

At shoulder_pan=0 the arm extends toward -Y world, so pans near 0 target
cubes that are directly in front of the robot (Y < 0).

Gripper top-down approach
-------------------------
For the gripper to point straight down ([0,0,-1] in world frame) the
wrist_2 joint must be near ±π/2 — NOT near 0.  wrist_2 = 0 keeps the
gripper pointing sideways (in the arm's sagittal plane).

This module is ROS-free — safe to import from training code and reward
functions.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# ── UR3e DH parameters ────────────────────────────────────────────────────────
# Source: ur_description/config/ur3e/default_kinematics.yaml
_DH_A     = np.array([0.0,      -0.24355, -0.21320, 0.0,      0.0,      0.0    ])
_DH_D     = np.array([0.15185,   0.0,      0.0,     0.13105,  0.08535,  0.09210])
_DH_ALPHA = np.array([np.pi/2,   0.0,      0.0,     np.pi/2, -np.pi/2,  0.0    ])

GRIPPER_LENGTH = 0.218   # flange → gripper_tcp (m)

# ── Robot base Z offset ────────────────────────────────────────────────────────
# Matches launch-file default: robot_z=1.10 m.
# Only a Z translation is applied — robot_yaw=π is NOT included because the
# world-frame TCP broadcaster does not reflect that rotation.
ROBOT_BASE_Z = 1.10   # base_link Z in world frame (m)

_T_BASE = np.array(
    [[1, 0, 0, 0],
     [0, 1, 0, 0],
     [0, 0, 1, ROBOT_BASE_Z],
     [0, 0, 0, 1]],
    dtype=float,
)

L1 = abs(_DH_A[1])   # upper arm = 0.24355 m
L2 = abs(_DH_A[2])   # forearm   = 0.21320 m

# How far above the cube centre to aim the TCP (world Z direction).
DEFAULT_APPROACH_HEIGHT = 0.07   # m

# IK solver tolerances
_IK_POS_TOL  = 0.03   # 3 cm acceptable FK residual
_IK_MAX_ITER = 400


# ── Forward kinematics ────────────────────────────────────────────────────────

def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [ 0,     sa,     ca,    d],
                     [ 0,      0,      0,    1]])


def fk_full(joints: np.ndarray) -> np.ndarray:
    """Return the full 4×4 world-frame transform for the gripper_tcp frame.

    T_world = T_BASE × T_DH_chain × T_gripper
    """
    T = np.eye(4)
    for i in range(6):
        T = T @ _dh(joints[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
    T_grip = np.eye(4)
    T_grip[2, 3] = GRIPPER_LENGTH
    return _T_BASE @ T @ T_grip


def forward_kinematics(joints: np.ndarray) -> np.ndarray:
    """Return gripper_tcp world-frame XYZ for 6 joint angles (rad)."""
    return fk_full(joints)[:3, 3]


def fk_tcp_z_axis(joints: np.ndarray) -> np.ndarray:
    """Return the gripper Z-axis direction in world frame.

    For a top-down approach this should be [0, 0, -1].
    Requires wrist_2 ≈ ±π/2 to achieve this orientation.
    """
    return fk_full(joints)[:3, 2]


# ── IK seed (pan-aligned, wrist_2=π/2 for downward approach) ─────────────────

def _approach_seed(cube_pos: np.ndarray) -> np.ndarray:
    """
    Return a sensible starting joint configuration for the IK solver.

    shoulder_pan is computed analytically so the arm faces the cube.
    The remaining joints use fixed elbow-up defaults that are near the
    natural top-down approach configuration for this workspace.

    Pan derivation (world frame):
      At pan=0 the arm extends toward −Y world.  The arm direction is
      (−sin(pan), −cos(pan)).  For a cube at world (cx, cy):
        −sin(p) = cx/r,  −cos(p) = cy/r  →  p = arctan2(−cx, −cy)
      Examples: cube at (0,−0.3) → pan=0; cube at (0.1,−0.3) → pan≈−18°.
    """
    cx, cy = float(cube_pos[0]), float(cube_pos[1])
    pan = np.arctan2(-cx, -cy)

    # Fixed elbow-up defaults tuned for the typical workspace
    # (shoulder_lift≈-π/2, elbow≈-π/2, wrist_1≈-π/2, wrist_2≈π/2)
    return np.array([pan, -np.pi/2, -np.pi/2, -np.pi/2, np.pi/2, 0.0])


# ── Numerical IK ─────────────────────────────────────────────────────────────

def compute_ik_reference(
    cube_pos: np.ndarray | list,
    approach_height: float = DEFAULT_APPROACH_HEIGHT,
) -> tuple[bool, np.ndarray, float]:
    """
    Compute IK joint angles to place the gripper TCP directly above the cube,
    pointing straight down (top-down approach).

    Returns
    -------
    reachable  : True if IK converged within _IK_POS_TOL
    joints     : shape (6,), joint angles in radians
    fk_error_m : residual position error in metres
    """
    cube   = np.asarray(cube_pos, dtype=float)
    target = cube + np.array([0.0, 0.0, approach_height])

    TARGET_Z_AXIS = np.array([0.0, 0.0, -1.0])

    def cost(q: np.ndarray) -> float:
        pos_err    = np.linalg.norm(forward_kinematics(q) - target)
        orient_err = np.linalg.norm(fk_tcp_z_axis(q) - TARGET_Z_AXIS)
        config_pen = 0.05 * max(0.0, q[1] + 0.3) ** 2
        return pos_err**2 + 0.5 * orient_err**2 + config_pen

    # Joint bounds matching training limits.
    # wrist_2 must be allowed near ±π/2 for downward approach — DO NOT restrict to ≈0.
    bounds = [
        (-2 * np.pi,  2 * np.pi),   # shoulder_pan
        (-np.pi,     -0.2),          # shoulder_lift: keep arm elevated
        (-2.5,        np.pi),        # elbow
        (-np.pi,      0.0),          # wrist_1
        (-np.pi,      np.pi),        # wrist_2: full range — ≈π/2 needed for downward grip
        (-np.pi,      np.pi),        # wrist_3: gripper roll
    ]

    base_seed = _approach_seed(cube)
    pan       = base_seed[0]

    # Multiple seeds: vary lift, elbow, and wrist angles around the base seed
    seeds = [
        base_seed,
        np.array([pan, -np.pi/2,  -np.pi/2, -np.pi/2, -np.pi/2, 0.0]),
        np.array([pan, -0.5,      -1.0,     -0.5,      np.pi/2,  0.0]),
        np.array([pan, -0.7,      -1.2,     -0.3,      np.pi/2,  0.0]),
        np.array([pan, -1.0,      -1.5,     -0.2,      np.pi/2,  0.0]),
        np.array([pan, -1.3,      -1.0,     -0.5,      np.pi/2,  0.0]),
        np.array([pan, -0.4,      -0.8,     -1.0,      np.pi/2,  0.0]),
        np.array([pan, -0.6,      -1.4,      0.0,      np.pi/2,  0.0]),
        np.array([pan, -1.5,      -1.8,      0.0,      np.pi/2,  0.0]),
        np.array([pan, -np.pi/3,  -np.pi/2, -np.pi/2,  np.pi/2,  0.0]),
        np.array([pan, -np.pi/4,  -np.pi/3, -np.pi/2,  np.pi/2,  0.0]),
        np.array([pan, -2.0,      -np.pi/2, -np.pi/2,  np.pi/2,  0.0]),
    ]

    best_joints, best_cost = base_seed, float("inf")
    for s in seeds:
        r = minimize(cost, s, method="L-BFGS-B", bounds=bounds,
                     options={"maxiter": _IK_MAX_ITER, "ftol": 1e-14})
        if r.fun < best_cost:
            best_cost   = r.fun
            best_joints = r.x

    fk_error  = float(np.linalg.norm(forward_kinematics(best_joints) - target))
    reachable = fk_error < _IK_POS_TOL

    return reachable, best_joints, fk_error


# ── Reachability scan ─────────────────────────────────────────────────────────

def scan_spawn_zone(
    x_range: tuple[float, float] = (-0.20, 0.20),
    y_range: tuple[float, float] = (-0.45, -0.10),
    cube_z: float = 1.11,
    approach_height: float = DEFAULT_APPROACH_HEIGHT,
    grid_steps: int = 8,
) -> dict:
    """Check IK reachability for a grid of cube positions in the spawn zone."""
    xs = np.linspace(x_range[0], x_range[1], grid_steps)
    ys = np.linspace(y_range[0], y_range[1], grid_steps)
    reachable, unreachable = [], []

    for x in xs:
        for y in ys:
            cube = np.array([x, y, cube_z])
            ok, joints, err = compute_ik_reference(cube, approach_height)
            entry: dict = {"pos": cube.tolist(), "error_m": round(err, 4)}
            if ok:
                entry["joints_deg"] = np.degrees(joints).round(1).tolist()
                reachable.append(entry)
            else:
                unreachable.append(entry)

    total = len(reachable) + len(unreachable)
    return {
        "reachable":   reachable,
        "unreachable": unreachable,
        "stats": {
            "total":         total,
            "n_reachable":   len(reachable),
            "n_unreachable": len(unreachable),
            "pct_reachable": round(100 * len(reachable) / max(total, 1), 1),
        },
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Scanning spawn zone reachability ...")
    print(f"  X: -0.20 → 0.20 m,  Y: -0.45 → -0.10 m,  Z: 1.11 m")
    print(f"  Approach height: {DEFAULT_APPROACH_HEIGHT*100:.0f} cm above cube")
    print(f"  IK tolerance: {_IK_POS_TOL*100:.0f} cm\n")

    results = scan_spawn_zone(grid_steps=6)
    stats   = results["stats"]

    print(f"{'POSITION':30s}  {'STATUS':12s}  {'FK ERROR':10s}")
    print("-" * 56)
    for r in results["reachable"]:
        p = r["pos"]
        print(f"  ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})  {'OK':12s}  {r['error_m']*100:.1f} cm")
    for r in results["unreachable"]:
        p = r["pos"]
        print(f"  ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})  {'UNREACHABLE':12s}  {r['error_m']*100:.1f} cm")

    print(f"\n{'─'*56}")
    print(f"Reachable: {stats['n_reachable']}/{stats['total']} ({stats['pct_reachable']}%)")
    if results["unreachable"]:
        print("Unreachable positions detected — consider tightening spawn bounds.")
