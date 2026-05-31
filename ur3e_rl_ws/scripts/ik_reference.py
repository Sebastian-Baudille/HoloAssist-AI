"""
ik_reference.py — IK reference configuration for UR3e + RG2 gripper.

Computes joint angles that place the gripper TCP directly above a cube,
pointing straight down (top-down approach for grasping).

Uses scipy numerical IK seeded with the planar polygon approach:
  1. shoulder_pan aligns the arm plane with the cube direction
  2. 2-link planar IK (law of cosines) for shoulder_lift and elbow — elbow-up solution
  3. wrist_1 keeps the TCP vertical
  4. wrist_2 = wrist_3 = 0

This module is ROS-free and importable from training code.
"""

from __future__ import annotations
import numpy as np
from scipy.optimize import minimize

# ── UR3e DH parameters (from ur_description/config/ur3e/default_kinematics.yaml)
_DH_A     = np.array([0.0,      -0.24355, -0.21320, 0.0,      0.0,      0.0    ])
_DH_D     = np.array([0.15185,   0.0,      0.0,     0.13105,  0.08535,  0.09210])
_DH_ALPHA = np.array([np.pi/2,   0.0,      0.0,     np.pi/2, -np.pi/2,  0.0    ])

GRIPPER_LENGTH  = 0.218     # flange → gripper_tcp (m)
BASE_Z          = 0.82      # robot base world Z (m)
BASE_YAW        = np.pi     # robot faces -X in world frame

# The analytic DH model underestimates Z by ~0.27 m relative to the real robot.
# Measured empirically: for 6 cube positions the actual TCP Z was consistently
# 0.26–0.28 m above the analytic FK prediction (mean 0.27 m).
# To target real Z = Z_real, set analytic target Z = Z_real - FK_Z_OFFSET.
FK_Z_OFFSET = 0.27

L1 = abs(_DH_A[1])          # upper arm = 0.24355 m
L2 = abs(_DH_A[2])          # forearm   = 0.21320 m

DEFAULT_APPROACH_HEIGHT = 0.07   # m above cube centre

# IK tolerances
_IK_POS_TOL  = 0.03   # 3 cm — acceptable FK residual for a reference pose
_IK_MAX_ITER = 200


# ── Forward kinematics ────────────────────────────────────────────────────────

def _dh(theta, d, a, alpha):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([[ct, -st*ca,  st*sa, a*ct],
                     [st,  ct*ca, -ct*sa, a*st],
                     [ 0,     sa,     ca,    d],
                     [ 0,      0,      0,    1]])


def fk_full(joints: np.ndarray) -> np.ndarray:
    """Return full 4×4 world-frame transform for the gripper_tcp frame."""
    cy, sy = np.cos(BASE_YAW), np.sin(BASE_YAW)
    T = np.array([[cy, -sy, 0, 0],
                  [sy,  cy, 0, 0],
                  [ 0,   0, 1, BASE_Z],
                  [ 0,   0, 0, 1]], dtype=float)
    for i in range(6):
        T = T @ _dh(joints[i], _DH_D[i], _DH_A[i], _DH_ALPHA[i])
    T_grip = np.eye(4); T_grip[2, 3] = GRIPPER_LENGTH
    return T @ T_grip


def forward_kinematics(joints: np.ndarray) -> np.ndarray:
    """Return gripper_tcp world-frame XYZ for 6 joint angles (rad)."""
    return fk_full(joints)[:3, 3]


def fk_tcp_z_axis(joints: np.ndarray) -> np.ndarray:
    """Return the gripper's Z-axis direction in world frame (should be [0,0,-1] for top-down)."""
    return fk_full(joints)[:3, 2]


# ── Planar IK seed (elbow-up) ─────────────────────────────────────────────────

def _planar_seed(cube_pos: np.ndarray, approach_height: float) -> np.ndarray:
    """
    Compute an approximate elbow-up configuration using planar geometry.
    Used as the starting point for the numerical solver.
    """
    tcp_target = cube_pos + np.array([0.0, 0.0, approach_height])

    # Shoulder_pan: align arm plane with cube direction in world XY
    # (account for base_yaw=pi: cube direction in robot frame is rotated)
    cube_in_robot = np.array([
        np.cos(-BASE_YAW) * cube_pos[0] - np.sin(-BASE_YAW) * cube_pos[1],
        np.sin(-BASE_YAW) * cube_pos[0] + np.cos(-BASE_YAW) * cube_pos[1],
    ])
    pan = np.arctan2(cube_in_robot[1], cube_in_robot[0])

    # In the sagittal plane, target the 2-link arm to a position that puts
    # the TCP at tcp_target when wrist hangs straight down.
    # Approximate wrist-to-TCP straight-line length:
    wrist_drop = _DH_D[3] + _DH_D[4] + _DH_D[5] + GRIPPER_LENGTH  # ≈ 0.527 m

    r_horiz    = float(np.sqrt(cube_pos[0]**2 + cube_pos[1]**2))
    target_r   = r_horiz
    target_z   = tcp_target[2] + wrist_drop          # wrist_1 must be above TCP
    shoulder_z = BASE_Z + _DH_D[0]
    dr = target_r
    dz = target_z - shoulder_z

    reach = float(np.sqrt(dr**2 + dz**2))
    reach = float(np.clip(reach, abs(L1 - L2) + 1e-6, L1 + L2 - 1e-6))

    cos_elbow = float(np.clip((reach**2 - L1**2 - L2**2) / (2*L1*L2), -1, 1))
    elbow     = -np.arccos(cos_elbow)                 # negative = elbow-up
    alpha     = np.arctan2(dz, dr)
    beta      = np.arctan2(L2 * np.sin(-elbow), L1 + L2 * np.cos(elbow))
    lift      = alpha - beta
    wrist1    = -(lift + elbow) - np.pi / 2.0

    return np.array([pan, lift, elbow, wrist1, 0.0, 0.0])


# ── Numerical IK ─────────────────────────────────────────────────────────────

def compute_ik_reference(
    cube_pos: np.ndarray | list,
    approach_height: float = DEFAULT_APPROACH_HEIGHT,
) -> tuple[bool, np.ndarray, float]:
    """
    Compute IK joint angles to place gripper TCP directly above the cube,
    pointing straight down.

    Returns
    -------
    reachable   : bool — True if IK converged within tolerance
    joints      : ndarray shape (6,) — joint angles in radians
    fk_error_m  : float — residual position error in metres
    """
    cube   = np.asarray(cube_pos, dtype=float)
    target = cube + np.array([0.0, 0.0, approach_height])

    # Analytic FK underestimates Z by FK_Z_OFFSET — subtract it from the
    # analytic target so the real robot hits the correct real-world Z.
    analytic_target = target - np.array([0.0, 0.0, FK_Z_OFFSET])
    seed = _planar_seed(cube, approach_height)

    # Target gripper Z-axis in world frame for top-down approach = [0, 0, -1]
    TARGET_Z_AXIS = np.array([0.0, 0.0, -1.0])

    def cost(q):
        pos_err     = np.linalg.norm(forward_kinematics(q) - analytic_target)
        # Orientation: gripper Z-axis must point straight down
        tcp_z       = fk_tcp_z_axis(q)
        orient_err  = np.linalg.norm(tcp_z - TARGET_Z_AXIS)
        # Safety: shoulder away from horizontal
        config_pen  = 0.05 * max(0.0, q[1] + 0.3)**2
        return pos_err**2 + 0.5 * orient_err**2 + config_pen

    # Joint bounds matching training limits.
    # wrist_2 tightly constrained near 0 — non-zero wrist_2 tilts the gripper sideways.
    bounds = [
        (-2*np.pi,  2*np.pi),   # shoulder_pan
        (-np.pi,   -0.2),        # shoulder_lift: keep arm elevated
        (-2.5,      np.pi),      # elbow
        (-np.pi,    0.0),        # wrist_1
        (-0.15,     0.15),       # wrist_2: near-zero keeps gripper pointing down
        (-np.pi,    np.pi),      # wrist_3: gripper roll (doesn't affect approach direction)
    ]

    # Multiple seeds: elbow-up, elbow-down, and mirrored variants
    pan = seed[0]
    extra_seeds = [
        seed,
        np.array([pan, -np.pi/3,  np.pi/2, -np.pi/6, 0.0, 0.0]),
        np.array([pan, -np.pi/2,  np.pi/3, -np.pi/6, 0.0, 0.0]),
        np.array([pan, -np.pi/4,  np.pi/2, -np.pi/4, 0.0, 0.0]),
        np.array([pan, -2.0,      np.pi/2, -np.pi/6, 0.0, 0.0]),
        np.array([pan + np.pi, -np.pi/3, np.pi/2, -np.pi/6, 0.0, 0.0]),
    ]

    best_joints, best_cost = seed, float("inf")
    for s in extra_seeds:
        r = minimize(cost, s, method='L-BFGS-B', bounds=bounds,
                     options={'maxiter': _IK_MAX_ITER, 'ftol': 1e-12})
        if r.fun < best_cost:
            best_cost  = r.fun
            best_joints = r.x

    joints    = best_joints
    fk_error  = float(np.linalg.norm(forward_kinematics(joints) - analytic_target))
    reachable = fk_error < _IK_POS_TOL

    return reachable, joints, fk_error


# ── Reachability scan ─────────────────────────────────────────────────────────

def scan_spawn_zone(
    x_range=(-0.20, 0.20),
    y_range=(-0.45, -0.10),
    cube_z=1.11,
    approach_height=DEFAULT_APPROACH_HEIGHT,
    grid_steps=8,
) -> dict:
    """
    Check IK reachability for a grid of cube positions in the spawn zone.

    Returns a dict with 'reachable', 'unreachable', and 'stats'.
    """
    xs = np.linspace(x_range[0], x_range[1], grid_steps)
    ys = np.linspace(y_range[0], y_range[1], grid_steps)
    reachable, unreachable = [], []

    for x in xs:
        for y in ys:
            cube = np.array([x, y, cube_z])
            ok, joints, err = compute_ik_reference(cube, approach_height)
            entry = {"pos": cube.tolist(), "error_m": round(err, 4)}
            if ok:
                entry["joints_deg"] = np.degrees(joints).round(1).tolist()
                reachable.append(entry)
            else:
                entry["error_m"] = round(err, 4)
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
    import json

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
        print(f"  ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})  {'✓ reachable':12s}  {r['error_m']*100:.1f} cm")
    for r in results["unreachable"]:
        p = r["pos"]
        print(f"  ({p[0]:+.2f}, {p[1]:+.2f}, {p[2]:.2f})  {'✗ unreachable':12s}  {r['error_m']*100:.1f} cm")

    print(f"\n{'─'*56}")
    print(f"Reachable: {stats['n_reachable']}/{stats['total']} ({stats['pct_reachable']}%)")

    if results["unreachable"]:
        print("\nUnreachable positions detected — consider tightening spawn bounds.")
