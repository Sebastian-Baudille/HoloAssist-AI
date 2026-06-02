from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SUCCESS_DISTANCE_M = 0.04
MIN_EE_Z_M = 0.02
COLLISION_PENALTY = 10.0
SUCCESS_REWARD = 10.0
TIME_PENALTY = 0.01
ACTION_PENALTY_SCALE = 0.01

# ── Success / failure thresholds ──────────────────────────────────────────────

SUCCESS_DISTANCE_M   = 0.04   # TCP must be within 4 cm of cube
MIN_EE_Z_M           = 0.02   # minimum safe end-effector height

# Gripper orientation: ||tcp_z_axis - [0,0,-1]|| at 20° tilt ≈ 0.35.
# Both distance AND orientation must be satisfied for success.
ORIENT_SUCCESS_TOL   = 0.35   # ~20° from straight down

# ── Per-step reward weights ───────────────────────────────────────────────────

DISTANCE_WEIGHT           = 1.0    # dominant signal — pull TCP toward cube
TIME_PENALTY              = 0.0    # removed: cumulative penalty buries distance signal
ACTION_PENALTY_SCALE      = 0.002  # tiny — don't punish exploration
COLLISION_PENALTY         = 0.50   # one-off on collision flag

# Orientation: small secondary signal — don't let it compete with distance.
ORIENT_WEIGHT             = 0.05

# IK reference tracking: disabled — competes with distance from home and
# is non-stationary (different per episode), confusing early-stage learning.
IK_TRACK_WEIGHT           = 0.0

# Elbow near-straight: disabled — home position has elbow=0 which triggers
# maximum penalty immediately, fighting the distance signal from step 1.
ELBOW_NEAR_STRAIGHT_WEIGHT    = 0.0
ELBOW_NEAR_STRAIGHT_THRESHOLD = 0.40

# Shoulder-lift soft safety: keep off during initial training.
SHOULDER_LIFT_UPPER_RAD  = -0.2
CONFIG_PENALTY_SCALE      = 0.0

# Terminal bonus — large enough to dominate a full episode of distance penalty.
REACH_BONUS               = 15.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def check_success(state: Mapping[str, object]) -> bool:
    """First reach-task success: the end effector is within 4 cm of the cube."""

    return (
        _distance(
            state["end_effector_position"],  # type: ignore[index]
            state["object_position"],  # type: ignore[index]
        )
        <= SUCCESS_DISTANCE_M
    )


def check_failure(state: Mapping[str, object]) -> bool:
    if bool(state.get("collision_flag", False)):
        return True
    end_effector_position = np.asarray(
        state["end_effector_position"],  # type: ignore[index]
        dtype=np.float32,
    ).reshape(3)
    return float(end_effector_position[2]) < MIN_EE_Z_M


def compute_reward(
    state: Mapping[str, object],
    action: Sequence[float] | None = None,
    step_count: int = 0,
) -> float:
    del step_count
    ee_to_cube = _distance(
        state["end_effector_position"],  # type: ignore[index]
        state["object_position"],  # type: ignore[index]
    )
    reward = -ee_to_cube - TIME_PENALTY

    if action is not None:
        reward -= ACTION_PENALTY_SCALE * float(np.linalg.norm(np.asarray(action, dtype=np.float32)))

    if check_success(state):
        reward += SUCCESS_REWARD

    if bool(state.get("collision_flag", False)):
        reward -= COLLISION_PENALTY

    return float(reward)

