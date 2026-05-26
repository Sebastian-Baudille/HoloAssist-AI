from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


SUCCESS_DISTANCE_M = 0.04
MIN_EE_Z_M = 0.02
COLLISION_PENALTY = 0.5
TIME_PENALTY = 0.001
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
    state: Mapping[str, object] | Sequence[float],
    action: Sequence[float] | None = None,
    step_count: int = 0,
    info: Mapping[str, object] | None = None,
) -> float:
    info_map = dict(info or {})
    action_vec = np.asarray(action if action is not None else np.zeros(6), dtype=np.float32).reshape(-1)

    if isinstance(state, Mapping):
        ee_pos = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)  # type: ignore[index]
        cube_pos = np.asarray(state["object_position"], dtype=np.float32).reshape(3)  # type: ignore[index]
        bin_pos = np.asarray(state.get("goal_position", state["object_position"]), dtype=np.float32).reshape(3)  # type: ignore[index]
        grasped = float(state.get("grasped", 0.0))
        timestep = float(step_count)
        info_map.setdefault("collision", bool(state.get("collision_flag", False)))
    else:
        obs = np.asarray(state, dtype=np.float32).reshape(-1)
        if obs.shape[0] < 13:
            raise ValueError(f"Expected observation with at least 13 values, got shape {obs.shape}")
        ee_pos = obs[0:3]
        cube_pos = obs[3:6]
        bin_pos = obs[6:9]
        grasped = float(obs[9])
        timestep = float(obs[12])

    dist_to_cube = float(np.linalg.norm(ee_pos - cube_pos))
    dist_to_bin = float(np.linalg.norm(cube_pos - bin_pos))

    reward = 0.0
    reward -= 0.3 * dist_to_cube

    if grasped > 0.5:
        reward += 5.0
        reward -= 0.5 * dist_to_bin

    if bool(info_map.get("cube_in_bin", False)):
        reward += 10.0

    reward -= COLLISION_PENALTY * float(bool(info_map.get("collision", False)))
    reward -= TIME_PENALTY * timestep
    reward -= ACTION_PENALTY_SCALE * float(np.sum(np.square(action_vec[:6])))

    return float(reward)
