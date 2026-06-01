from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ur3e_rl_env.kinematics import fk_tcp_z_axis


# ── Success / failure thresholds ──────────────────────────────────────────────

SUCCESS_DISTANCE_M   = 0.04   # TCP must be within 4 cm of cube
MIN_EE_Z_M           = 0.02   # minimum safe end-effector height

# Gripper orientation: ||tcp_z_axis - [0,0,-1]|| at 20° tilt ≈ 0.35.
# Both distance AND orientation must be satisfied for success.
ORIENT_SUCCESS_TOL   = 0.35   # ~20° from straight down

# ── Per-step reward weights ───────────────────────────────────────────────────

DISTANCE_WEIGHT           = 0.30   # pull TCP toward cube
TIME_PENALTY              = 0.001  # per step — encourages speed
ACTION_PENALTY_SCALE      = 0.01   # penalises large joint deltas
COLLISION_PENALTY         = 0.50   # one-off on collision flag

# Orientation: penalise gripper not pointing straight down.
# At 90° off, orient_err ≈ 1.41, penalty ≈ 0.21 — comparable to a 70 cm distance penalty.
ORIENT_WEIGHT             = 0.15

# IK reference tracking: soft pull toward the analytically correct configuration.
# The error is ||current_joints - ik_ref_joints|| in radians.
IK_TRACK_WEIGHT           = 0.10

# Elbow near-straight: penalise when |elbow| is small (arm approaching singularity).
# Pushes the policy toward the elbow-up "hump" shape with a healthy bend.
ELBOW_NEAR_STRAIGHT_WEIGHT    = 0.20
ELBOW_NEAR_STRAIGHT_THRESHOLD = 0.40   # rad — below this the arm is considered near-straight

# Shoulder-lift soft safety: gradient warning before the hard joint limit.
SHOULDER_LIFT_UPPER_RAD  = -0.2    # must match constants.py
CONFIG_PENALTY_SCALE      = 2.0

# Terminal bonus
REACH_BONUS               = 5.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def _gripper_orient_error(joints: np.ndarray) -> float:
    """||tcp_z_axis - [0,0,-1]|| — 0 when gripper points straight down."""
    z_axis = fk_tcp_z_axis(np.asarray(joints, dtype=float))
    return float(np.linalg.norm(z_axis - np.array([0.0, 0.0, -1.0])))


# ── Success / failure checks ──────────────────────────────────────────────────

def check_success(state: Mapping[str, object]) -> bool:
    """
    Phase A success: TCP within 4 cm of the cube AND gripper pointing down
    within ORIENT_SUCCESS_TOL (~20°).
    """
    dist_ok = (
        _distance(state["end_effector_position"], state["object_position"])
        <= SUCCESS_DISTANCE_M
    )
    if not dist_ok:
        return False

    if "joint_positions" in state:
        orient_err = _gripper_orient_error(state["joint_positions"])
        if orient_err > ORIENT_SUCCESS_TOL:
            return False

    return True


def check_failure(state: Mapping[str, object]) -> bool:
    if bool(state.get("collision_flag", False)):
        return True
    ee_pos = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    return float(ee_pos[2]) < MIN_EE_Z_M


# ── Reward computation ────────────────────────────────────────────────────────

def compute_reward(
    state: Mapping[str, object] | Sequence[float],
    action: Sequence[float] | None = None,
    step_count: int = 0,
    info: Mapping[str, object] | None = None,
) -> float:
    info_map   = dict(info or {})
    action_vec = np.asarray(
        action if action is not None else np.zeros(6), dtype=np.float32
    ).reshape(-1)

    if isinstance(state, Mapping):
        ee_pos   = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
        cube_pos = np.asarray(state["object_position"],       dtype=np.float32).reshape(3)
        timestep = float(step_count)
        info_map.setdefault("collision", bool(state.get("collision_flag", False)))
    else:
        obs      = np.asarray(state, dtype=np.float32).reshape(-1)
        ee_pos   = obs[0:3]
        cube_pos = obs[3:6]
        timestep = float(step_count)

    dist_to_cube = float(np.linalg.norm(ee_pos - cube_pos))

    # ── Core reach signal ─────────────────────────────────────────────────────
    reward  = -DISTANCE_WEIGHT    * dist_to_cube
    reward -= TIME_PENALTY        * timestep
    reward -= ACTION_PENALTY_SCALE * float(np.sum(np.square(action_vec[:6])))
    reward -= COLLISION_PENALTY   * float(bool(info_map.get("collision", False)))

    if bool(info_map.get("reached", False)):
        reward += REACH_BONUS

    # ── Geometry terms (require joint positions) ──────────────────────────────
    if isinstance(state, Mapping) and "joint_positions" in state:
        joints = np.asarray(state["joint_positions"], dtype=np.float32)

        # 1. Orientation: gripper Z-axis should point straight down
        orient_err = _gripper_orient_error(joints)
        reward -= ORIENT_WEIGHT * orient_err

        # 2. IK reference tracking: pull toward analytically correct configuration.
        #    ik_reference_joints is set by the env at each reset; None if IK failed.
        ik_ref = info_map.get("ik_reference_joints")
        if ik_ref is not None:
            ik_ref     = np.asarray(ik_ref, dtype=np.float32)
            config_err = float(np.linalg.norm(joints - ik_ref))
            reward    -= IK_TRACK_WEIGHT * config_err

        # 3. Elbow near-straight penalty: |elbow| < threshold → arm approaching
        #    singularity / near-straight.  Penalty ramps linearly from 0 at the
        #    threshold to ELBOW_NEAR_STRAIGHT_WEIGHT when elbow = 0.
        elbow = float(joints[2])
        if abs(elbow) < ELBOW_NEAR_STRAIGHT_THRESHOLD:
            nearness = (ELBOW_NEAR_STRAIGHT_THRESHOLD - abs(elbow)) / ELBOW_NEAR_STRAIGHT_THRESHOLD
            reward  -= ELBOW_NEAR_STRAIGHT_WEIGHT * nearness

        # 4. Shoulder-lift soft safety: gradient warning before the hard limit.
        shoulder_lift = float(joints[1])
        if shoulder_lift > SHOULDER_LIFT_UPPER_RAD - 0.3:
            proximity = shoulder_lift - (SHOULDER_LIFT_UPPER_RAD - 0.3)
            reward   -= CONFIG_PENALTY_SCALE * max(0.0, proximity)

    return float(reward)
