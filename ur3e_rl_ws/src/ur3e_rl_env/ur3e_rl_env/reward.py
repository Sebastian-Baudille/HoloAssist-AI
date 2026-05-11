from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SUCCESS_DISTANCE_M = 0.04
MIN_EE_Z_M = 0.02
COLLISION_PENALTY = 10.0
SUCCESS_REWARD = 10.0
TIME_PENALTY = 0.01
ACTION_PENALTY_SCALE = 0.01


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

