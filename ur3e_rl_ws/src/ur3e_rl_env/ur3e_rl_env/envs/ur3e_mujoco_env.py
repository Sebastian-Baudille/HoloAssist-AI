"""
ur3e_mujoco_env.py — MuJoCo training backend for HoloAssist-AI.

Drop-in replacement for ur3e_pick_place_env.py (Gazebo).
No ROS. No middleware. Pure Python + MuJoCo.

Obs/action spaces are identical to the Gazebo env — trained weights
transfer directly to the ROS deployment node.

Key design decisions:
- RG2 gripper has all-fixed joints in the URDF; gripper is tracked
  as a virtual bool (_gripper_closed) with no physics actuation.
- Obs normalization replicates build_observation() from ros_interface.py
  exactly, including the binary gripper_state at index 10.
- Cube target = nearest cube to EE each step (matches Gazebo behaviour).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ur3e_rl_env.constants import (
    JOINT_DELTA_ACTION_SCALE_RAD,
    OBSERVATION_SIZE_13D,
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    WORKSPACE_HEIGHT_M,
    TABLE_TOP_Z,
    BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
    JOINT_TARGET_DURATION_SEC,
)
from ur3e_rl_env.reward import compute_reward, check_failure

# ── Paths ──────────────────────────────────────────────────────────────────────
# __file__ = ur3e_rl_env/envs/ur3e_mujoco_env.py
# Go up: envs/ → ur3e_rl_env/ (package) → src/ur3e_rl_env/ (repo root)
_PKG_DIR    = Path(__file__).parent.parent   # ur3e_rl_env/
_SRC_DIR    = _PKG_DIR.parent                # src/ur3e_rl_env/
ASSETS_PATH = _SRC_DIR / "assets" / "mujoco"
SCENE_XML   = str(ASSETS_PATH / "scene.xml")

# ── Episode / physics constants ────────────────────────────────────────────────
MAX_EPISODE_STEPS = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))

# 0.002 s × 50 = 0.1 s per RL step (Gazebo uses 0.2-0.3 s; shorter is fine
# for training speed — obs/action spaces are what matter for weight transfer)
PHYSICS_STEPS_PER_RL_STEP = 50

# Home config matching Gazebo env HOME_JOINTS and the Gazebo launch send_home cmd
HOME_JOINTS = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0], dtype=np.float64)

# Cube spawn ranges — must match ur3e_pick_place_env.py env-var defaults
CUBE_X_RANGE = (
    float(os.getenv("UR3E_RL_CUBE_X_MIN", "-0.20")),
    float(os.getenv("UR3E_RL_CUBE_X_MAX",  "0.20")),
)
CUBE_Y_RANGE = (
    float(os.getenv("UR3E_RL_CUBE_Y_MIN", "-0.45")),
    float(os.getenv("UR3E_RL_CUBE_Y_MAX", "-0.10")),
)
CUBE_Z = float(os.getenv("UR3E_RL_CUBE_Z", "1.11"))

# Grasping thresholds — from ros_interface.py
GRASP_PROXIMITY_THRESHOLD_M = 0.03
CUBE_IN_BIN_DIST_M = 0.08

# Standard UR3e arm joint names (no prefix — confirmed from test_urdf_load.py)
_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

_NUM_CUBES = 4
_TCP_BODY_NAME = "gripper_tcp"  # link from rg2_fixed.xacro: tool0 + 0.218 m along Z


def _normalize_axis(value: float, min_value: float, max_value: float) -> float:
    """Exact copy of ros_interface._normalize_axis for weight-transfer compatibility."""
    span = max(max_value - min_value, 1e-6)
    return float(np.clip(2.0 * ((value - min_value) / span) - 1.0, -1.0, 1.0))


def _normalize_xyz(xyz: np.ndarray) -> np.ndarray:
    """Exact copy of ros_interface._normalize_xyz."""
    return np.array([
        _normalize_axis(float(xyz[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
        _normalize_axis(float(xyz[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
        _normalize_axis(float(xyz[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
    ], dtype=np.float32)


class UR3eMuJoCoEnv(gym.Env):
    """
    MuJoCo-based UR3e pick-and-place environment.
    Identical obs/action spaces to UR3ePickPlaceEnv (Gazebo version).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()

        if not os.path.exists(SCENE_XML):
            raise FileNotFoundError(
                f"Scene XML not found: {SCENE_XML}\n"
                "Run the URDF conversion tasks (Tasks 1-2) first."
            )

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        # Spaces — must match Gazebo env exactly
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(OBSERVATION_SIZE_13D,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        # Joint limits for action denormalisation
        self._joint_lower = np.array(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float64)
        self._joint_upper = np.array(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float64)
        self._joint_range = (self._joint_upper - self._joint_lower) / 2.0
        self._joint_mid   = (self._joint_upper + self._joint_lower) / 2.0

        # Fixed bin position
        self._bin_pos = np.array(
            [BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z], dtype=np.float32
        )

        self._cache_ids()

        # Virtual gripper state — RG2 has no movable joint in URDF
        self._gripper_closed = False

        self.step_count  = 0
        self.render_mode = render_mode
        self._viewer     = None

        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ── ID caching ─────────────────────────────────────────────────────────────

    def _cache_ids(self) -> None:
        """Cache MuJoCo IDs to avoid per-step string lookups."""
        # TCP body
        try:
            self._tcp_body_id = self.model.body(_TCP_BODY_NAME).id
        except Exception:
            self._tcp_body_id = None
            for i in range(self.model.nbody):
                if "tcp" in self.model.body(i).name.lower():
                    self._tcp_body_id = i
                    break
            if self._tcp_body_id is None:
                raise RuntimeError(
                    f"Cannot find TCP body '{_TCP_BODY_NAME}' in model. "
                    f"Bodies: {[self.model.body(i).name for i in range(self.model.nbody)]}"
                )

        # Cube body IDs
        self._cube_body_ids: list[int] = []
        for i in range(_NUM_CUBES):
            self._cube_body_ids.append(self.model.body(f"cube_{i}").id)

        # Arm joint qpos addresses
        self._arm_qpos_addrs: list[int] = []
        for name in _ARM_JOINT_NAMES:
            try:
                jnt = self.model.joint(name)
                self._arm_qpos_addrs.append(int(jnt.qposadr[0]))
            except Exception as exc:
                raise RuntimeError(
                    f"Arm joint '{name}' not found in MuJoCo model. Error: {exc}"
                ) from exc

        # Cube freejoint qpos addresses (7 values each: xyz + quat)
        self._cube_qpos_addrs: list[int] = []
        for i in range(_NUM_CUBES):
            jnt_id = self.model.body(f"cube_{i}").jntadr[0]
            self._cube_qpos_addrs.append(int(self.model.jnt_qposadr[jnt_id]))

        print(
            f"[UR3eMuJoCoEnv] tcp_body={self._tcp_body_id}, "
            f"arm_joints={len(self._arm_qpos_addrs)}/6, "
            f"actuators={self.model.nu}"
        )

    # ── Observation ────────────────────────────────────────────────────────────

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._tcp_body_id].astype(np.float32)

    def _get_cube_pos(self, cube_idx: int) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids[cube_idx]].astype(np.float32)

    def _nearest_cube_idx(self, ee_pos: np.ndarray) -> int:
        """Return index of cube nearest to EE — matches Gazebo _current_object_position."""
        dists = [
            float(np.linalg.norm(self._get_cube_pos(i) - ee_pos))
            for i in range(_NUM_CUBES)
        ]
        return int(np.argmin(dists))

    def _get_obs(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray:
        """Build 13D observation. Byte-for-byte identical to build_observation()."""
        grasped       = self._check_grasped(ee_pos, cube_pos)
        gripper_state = 1.0 if self._gripper_closed else 0.0
        ee_height_norm = float(np.clip(float(ee_pos[2]) / WORKSPACE_HEIGHT_M, -1.0, 1.0))
        timestep_norm  = float(
            np.clip(float(self.step_count) / max(float(MAX_EPISODE_STEPS), 1.0), 0.0, 1.0)
        )

        obs = np.array([
            *_normalize_xyz(ee_pos),        # [0:3]
            *_normalize_xyz(cube_pos),       # [3:6]
            *_normalize_xyz(self._bin_pos),  # [6:9]
            grasped,                         # [9]  binary
            gripper_state,                   # [10] binary
            ee_height_norm,                  # [11] ee_z / WORKSPACE_HEIGHT_M
            timestep_norm,                   # [12] step / max_steps
        ], dtype=np.float32)

        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _check_grasped(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> float:
        """Grasped = EE within 3cm of cube AND gripper closed. Matches ros_interface.py."""
        dist = float(np.linalg.norm(ee_pos - cube_pos))
        return 1.0 if (dist < GRASP_PROXIMITY_THRESHOLD_M and self._gripper_closed) else 0.0

    # ── Action application ─────────────────────────────────────────────────────

    def _apply_action(self, action: np.ndarray) -> None:
        # Denormalise arm: [-1,1] → joint angles (matches pick_place_env.py step())
        arm_targets = action[:6].astype(np.float64) * self._joint_range + self._joint_mid

        # Slew limit — matches JOINT_DELTA_ACTION_SCALE_RAD in safety_checker
        current = np.array(
            [self.data.qpos[addr] for addr in self._arm_qpos_addrs], dtype=np.float64
        )
        arm_targets = np.clip(
            arm_targets,
            current - JOINT_DELTA_ACTION_SCALE_RAD,
            current + JOINT_DELTA_ACTION_SCALE_RAD,
        )
        arm_targets = np.clip(arm_targets, self._joint_lower, self._joint_upper)
        self.data.ctrl[:6] = arm_targets

        # Gripper: virtual state — action[6] > 0.5 close, < -0.5 open
        gripper_cmd = float(action[6])
        if gripper_cmd > 0.5:
            self._gripper_closed = True
        elif gripper_cmd < -0.5:
            self._gripper_closed = False

    # ── step() ─────────────────────────────────────────────────────────────────

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(7), -1.0, 1.0)

        self._apply_action(action)
        for _ in range(PHYSICS_STEPS_PER_RL_STEP):
            mujoco.mj_step(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_idx = self._nearest_cube_idx(ee_pos)
        cube_pos = self._get_cube_pos(cube_idx)
        grasped  = self._check_grasped(ee_pos, cube_pos)

        cube_in_bin = bool(
            np.linalg.norm(cube_pos - self._bin_pos) < CUBE_IN_BIN_DIST_M
            and grasped < 0.5
        )

        info = {
            "collision":        False,
            "cube_in_bin":      cube_in_bin,
            "is_success":       cube_in_bin,
            "distance_to_cube": float(np.linalg.norm(ee_pos - cube_pos)),
        }

        obs    = self._get_obs(ee_pos, cube_pos)
        reward = compute_reward(obs, action, self.step_count, info)

        # check_failure: EE z < 0.02 or collision_flag (matches Gazebo)
        failure = check_failure({
            "end_effector_position": ee_pos,
            "collision_flag": False,
        })
        terminated = bool(cube_in_bin) or failure
        truncated  = self.step_count >= MAX_EPISODE_STEPS

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        return obs, reward, terminated, truncated, info

    # ── reset() ────────────────────────────────────────────────────────────────

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self._gripper_closed = False

        # Set arm to home position
        for i, addr in enumerate(self._arm_qpos_addrs):
            self.data.qpos[addr] = HOME_JOINTS[i]

        # Randomise cube positions — match Gazebo env spawn ranges
        for i in range(_NUM_CUBES):
            addr = self._cube_qpos_addrs[i]
            x = float(self.np_random.uniform(CUBE_X_RANGE[0], CUBE_X_RANGE[1]))
            y = float(self.np_random.uniform(CUBE_Y_RANGE[0], CUBE_Y_RANGE[1]))
            self.data.qpos[addr    ] = x
            self.data.qpos[addr + 1] = y
            self.data.qpos[addr + 2] = CUBE_Z
            self.data.qpos[addr + 3] = 1.0  # quaternion w
            self.data.qpos[addr + 4] = 0.0
            self.data.qpos[addr + 5] = 0.0
            self.data.qpos[addr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

        # Settle physics — cubes fall and land on table
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        # step_count reset AFTER settle so obs[12]=0 at reset
        self.step_count = 0

        ee_pos   = self._get_ee_pos()
        cube_idx = self._nearest_cube_idx(ee_pos)
        cube_pos = self._get_cube_pos(cube_idx)
        obs = self._get_obs(ee_pos, cube_pos)

        return obs, {"target_cube": f"cube_{cube_idx}"}

    # ── render() / close() ─────────────────────────────────────────────────────

    def render(self):
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            return renderer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
