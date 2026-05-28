from __future__ import annotations

import math

# Max per-step joint motion cap (radians) used as a slew limit when converting
# normalized absolute targets into commanded trajectories.
JOINT_DELTA_ACTION_SCALE_RAD = 0.24

# Observation layout settings.
OBSERVATION_SIZE_13D = 13
# Phase A layout: ee_xyz(3) + cube_xyz(3) + joint_positions(6) + ee_height(1) + timestep(1) = 14
OBSERVATION_SIZE_PHASE_A = 14

# Workspace bounds used for observation normalization and perception filtering.
WORKSPACE_X_MIN = -0.46
WORKSPACE_X_MAX = 0.40
WORKSPACE_Y_MIN = -0.56
WORKSPACE_Y_MAX = 0.14
TABLE_TOP_Z = 1.07
WORKSPACE_Z_MIN = TABLE_TOP_Z + 0.01
WORKSPACE_Z_MAX = TABLE_TOP_Z + 0.15
WORKSPACE_HEIGHT_M = WORKSPACE_Z_MAX

# Default bin position in world coordinates.
BIN_POSITION_X = 0.28
BIN_POSITION_Y = 0.0
BIN_POSITION_Z = 1.078

# Conservative UR3e software limits used by normalization and clamping.
UR3E_JOINT_LOWER_LIMITS_RAD = (
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.0 * math.pi,
    -2.0 * math.pi,
)
UR3E_JOINT_UPPER_LIMITS_RAD = (
    2.0 * math.pi,
    2.0 * math.pi,
    2.0 * math.pi,
    2.0 * math.pi,
    2.0 * math.pi,
    2.0 * math.pi,
)

JOINT_TARGET_DURATION_SEC = 0.3
