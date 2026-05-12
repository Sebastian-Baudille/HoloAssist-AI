from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


UR3E_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

# Simple, conservative software limits for the first RL pass. These are not a
# replacement for the real controller safety limits.
UR3E_JOINT_LIMITS = {
    "shoulder_pan_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "shoulder_lift_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "elbow_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "wrist_1_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "wrist_2_joint": (-2.0 * math.pi, 2.0 * math.pi),
    "wrist_3_joint": (-2.0 * math.pi, 2.0 * math.pi),
}


@dataclass(frozen=True)
class SafetyResult:
    safe: bool
    reason: str = ""


class SafetyChecker:
    """Small command safety layer used before publishing joint targets."""

    def __init__(
        self,
        max_delta_rad: float = 0.24,
        min_end_effector_z: float = 0.02,
        joint_limits: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        self.max_delta_rad = float(max_delta_rad)
        self.min_end_effector_z = float(min_end_effector_z)
        self.joint_limits = dict(joint_limits or UR3E_JOINT_LIMITS)

    def clamp_joint_deltas(self, deltas: Sequence[float]) -> np.ndarray:
        deltas_array = np.asarray(deltas, dtype=np.float32).reshape(6)
        return np.clip(deltas_array, -self.max_delta_rad, self.max_delta_rad)

    def clamp_target_joints(self, target_joints: Sequence[float]) -> np.ndarray:
        target_array = np.asarray(target_joints, dtype=np.float32).reshape(6)
        lower = np.array(
            [self.joint_limits[name][0] for name in UR3E_JOINT_NAMES],
            dtype=np.float32,
        )
        upper = np.array(
            [self.joint_limits[name][1] for name in UR3E_JOINT_NAMES],
            dtype=np.float32,
        )
        return np.clip(target_array, lower, upper)

    def make_safe_target(
        self,
        current_joints: Sequence[float],
        requested_delta: Sequence[float],
    ) -> np.ndarray:
        current_array = np.asarray(current_joints, dtype=np.float32).reshape(6)
        safe_delta = self.clamp_joint_deltas(requested_delta)
        return self.clamp_target_joints(current_array + safe_delta)

    def check_collision(self, collision_flag: bool) -> SafetyResult:
        if bool(collision_flag):
            return SafetyResult(False, "collision flag is set")
        return SafetyResult(True)

    def check_end_effector_height(self, end_effector_position: Sequence[float]) -> SafetyResult:
        ee_position = np.asarray(end_effector_position, dtype=np.float32).reshape(3)
        if float(ee_position[2]) < self.min_end_effector_z:
            return SafetyResult(False, "end effector is below minimum z height")
        return SafetyResult(True)

    def check_state(self, state: Mapping[str, object]) -> SafetyResult:
        collision_result = self.check_collision(bool(state.get("collision_flag", False)))
        if not collision_result.safe:
            return collision_result

        ee_position = state.get("end_effector_position")
        if ee_position is None:
            return SafetyResult(False, "missing end effector position")
        return self.check_end_effector_height(ee_position)

