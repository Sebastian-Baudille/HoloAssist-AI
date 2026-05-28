from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SUCCESS_DISTANCE_M   = 0.04   # TCP must be within 4 cm of cube
MIN_EE_Z_M           = 0.02   # minimum safe end-effector height
COLLISION_PENALTY    = 0.5
TIME_PENALTY         = 0.001  # per step
ACTION_PENALTY_SCALE = 0.01   # penalises large joint deltas
REACH_BONUS          = 5.0    # one-off terminal bonus when TCP reaches cube


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def check_success(state: Mapping[str, object]) -> bool:
    """Phase A success: TCP within 4 cm of the cube."""
    return (
        _distance(
            state["end_effector_position"],
            state["object_position"],
        )
        <= SUCCESS_DISTANCE_M
    )


def check_failure(state: Mapping[str, object]) -> bool:
    if bool(state.get("collision_flag", False)):
        return True
    ee_pos = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    return float(ee_pos[2]) < MIN_EE_Z_M


def compute_reward(
    state: Mapping[str, object] | Sequence[float],
    action: Sequence[float] | None = None,
    step_count: int = 0,
    info: Mapping[str, object] | None = None,
) -> float:
    info_map   = dict(info or {})
    action_vec = np.asarray(action if action is not None else np.zeros(6), dtype=np.float32).reshape(-1)

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

    reward  = -0.3  * dist_to_cube
    reward -= TIME_PENALTY         * timestep
    reward -= ACTION_PENALTY_SCALE * float(np.sum(np.square(action_vec[:6])))
    reward -= COLLISION_PENALTY    * float(bool(info_map.get("collision", False)))

    if bool(info_map.get("reached", False)):
        reward += REACH_BONUS

    return float(reward)
