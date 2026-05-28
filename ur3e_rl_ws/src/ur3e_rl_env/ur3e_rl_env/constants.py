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

# UR3e joint limits for tabletop manipulation.
# Tighter than hardware limits to prevent self-collision and upper-arm sweeping.
#
# Joint indices:  0=shoulder_pan  1=shoulder_lift  2=elbow
#                 3=wrist_1       4=wrist_2        5=wrist_3
#
# shoulder_lift upper = -0.2 rad (~11° above horizontal):
#   keeps upper arm link elevated, preventing it from sweeping through
#   the table workspace and knocking cubes. Home pose uses -π/2.
#
# elbow lower = -2.5 rad:
#   prevents extreme reverse-elbow fold that causes the forearm to swing
#   back into the base/upper-arm region.
UR3E_JOINT_LOWER_LIMITS_RAD = (
    -2.0 * math.pi,  # shoulder_pan:   full rotation fine
    -math.pi,        # shoulder_lift:  down to pointing-back-and-up
    -2.5,            # elbow:          ~143° reverse limit
    -math.pi,        # wrist_1
    -math.pi,        # wrist_2
    -math.pi,        # wrist_3
)
UR3E_JOINT_UPPER_LIMITS_RAD = (
     2.0 * math.pi,  # shoulder_pan
    -0.2,            # shoulder_lift:  no higher than ~11° above horizontal
     math.pi,        # elbow
     0.0,            # wrist_1:        keeps wrist from flipping past neutral
     math.pi,        # wrist_2
     math.pi,        # wrist_3
)

JOINT_TARGET_DURATION_SEC = 0.3
