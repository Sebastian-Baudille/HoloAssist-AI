"""
transport_env.py — Stage 3: move the grasped cube to the bin.

The cube is kinematically attached to the TCP at reset (simulating a grasp).
Each physics step the cube is teleported to follow the TCP exactly.
The model only needs to learn: move TCP from wherever it is to the bin.

Obs (7D): [EE_x, EE_y, EE_z, cube_x, cube_y, cube_z, dist_to_bin_norm]
Action (3D): Cartesian delta (dx, dy, dz) in [-1, 1]; ±2 cm/step.
Reward: -dist(cube, bin)  (dense, always ≤ 0)
Done:   dist(cube, bin) < RELEASE_DIST_M (success) OR step >= MAX_STEPS
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ur3e_rl_env.constants import (
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
)
from ur3e_rl_env.ik import build_ik_cache, cartesian_to_joint_targets

_SRC_DIR     = Path(__file__).parent.parent.parent
SCENE_XML    = str(_SRC_DIR / "assets" / "mujoco" / "scene.xml")

MAX_STEPS      = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))
PHYSICS_STEPS  = 50
RELEASE_DIST_M = 0.08   # success threshold: 8 cm
HOME_JOINTS    = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

CUBE_X_RANGE = (float(os.getenv("UR3E_RL_CUBE_X_MIN", "-0.20")),
                float(os.getenv("UR3E_RL_CUBE_X_MAX",  "0.20")))
CUBE_Y_RANGE = (float(os.getenv("UR3E_RL_CUBE_Y_MIN", "-0.45")),
                float(os.getenv("UR3E_RL_CUBE_Y_MAX", "-0.10")))
CUBE_Z_START = float(os.getenv("UR3E_RL_CUBE_Z", "1.11"))
_NUM_CUBES   = 4

# 3 cm below TCP — approximately half-cube-height (cube is 4 cm tall)
_HOLD_OFFSET = np.array([0.0, 0.0, -0.03])

_BIN_POS = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z],
                    dtype=np.float32)
# Maximum distance cube can be from bin (for normalisation)
_MAX_BIN_DIST = 1.0


def _norm(v: float, lo: float, hi: float) -> float:
    # Intentional copy of ros_interface._normalize_xyz — keeps this file ROS-free.
    span = max(hi - lo, 1e-6)
    return float(np.clip(2.0 * (v - lo) / span - 1.0, -1.0, 1.0))


def _norm_xyz(pos: np.ndarray) -> np.ndarray:
    return np.array([
        _norm(float(pos[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
        _norm(float(pos[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
        _norm(float(pos[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
    ], dtype=np.float32)


class UR3eTransportEnv(gym.Env):
    """Transport: move arm+cube from current position to within 8 cm of bin."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (7,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self._joint_lower = np.array(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float64)
        self._joint_upper = np.array(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float64)
        self._ik_cache    = build_ik_cache(self.model)

        self._cube_body_ids   = [self.model.body(f"cube_{i}").id for i in range(_NUM_CUBES)]
        self._cube_qpos_addrs = []
        self._cube_dof_addrs  = []
        for i in range(_NUM_CUBES):
            jnt_id = self.model.body(f"cube_{i}").jntadr[0]
            self._cube_qpos_addrs.append(int(self.model.jnt_qposadr[jnt_id]))
            self._cube_dof_addrs.append(int(self.model.jnt_dofadr[jnt_id]))

        self._target_cube_idx = 0
        self.step_count       = 0
        self.render_mode      = render_mode
        self._viewer          = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self._renderer: mujoco.Renderer | None = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)

    def _pin_cube_to_tcp(self) -> None:
        """Teleport the target cube to follow the TCP each physics step."""
        tcp_pos  = self.data.xpos[self._ik_cache["tcp_body_id"]].copy()
        tcp_quat = self.data.xquat[self._ik_cache["tcp_body_id"]].copy()
        cube_world_pos = tcp_pos + _HOLD_OFFSET
        addr = self._cube_qpos_addrs[self._target_cube_idx]
        dof  = self._cube_dof_addrs[self._target_cube_idx]
        self.data.qpos[addr:addr+3]   = cube_world_pos
        self.data.qpos[addr+3:addr+7] = tcp_quat
        self.data.qvel[dof:dof+6]     = 0.0

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids[self._target_cube_idx]].astype(np.float32)

    def _get_obs(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray:
        dist_to_bin = float(np.linalg.norm(cube_pos - _BIN_POS))
        dist_norm   = float(np.clip(2.0 * dist_to_bin / _MAX_BIN_DIST - 1.0, -1.0, 1.0))
        return np.concatenate([
            _norm_xyz(ee_pos),
            _norm_xyz(cube_pos),
            [dist_norm],
        ]).astype(np.float32)

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)

        # Check pre-step cube position — handles externally teleported cube (e.g. tests)
        # and the edge case where the cube is already at the bin before physics runs.
        pre_cube_pos = self._get_cube_pos()
        pre_dist_bin = float(np.linalg.norm(pre_cube_pos - _BIN_POS))
        if pre_dist_bin < RELEASE_DIST_M:
            ee_pos   = self._get_ee_pos()
            obs      = self._get_obs(ee_pos, pre_cube_pos)
            info     = {"is_success": True, "dist_to_bin": pre_dist_bin}
            return obs, -pre_dist_bin, True, self.step_count >= MAX_STEPS, info

        q_target = cartesian_to_joint_targets(
            self.model, self.data, self._ik_cache,
            delta_xyz=action.astype(np.float64),
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
        )
        self.data.ctrl[:6] = q_target

        for _ in range(PHYSICS_STEPS):
            self._pin_cube_to_tcp()
            mujoco.mj_step(self.model, self.data)
        self._pin_cube_to_tcp()  # ensure final position is consistent

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        dist_bin = float(np.linalg.norm(cube_pos - _BIN_POS))

        reward     = -dist_bin
        success    = dist_bin < RELEASE_DIST_M
        terminated = success
        truncated  = self.step_count >= MAX_STEPS
        info       = {"is_success": success, "dist_to_bin": dist_bin}

        obs = self._get_obs(ee_pos, cube_pos)

        if self.render_mode == "human" and self._viewer is not None:
            self._viewer.sync()

        return obs, reward, terminated, truncated, info

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        for i, addr in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr] = HOME_JOINTS[i]

        self._target_cube_idx = int(self.np_random.integers(0, _NUM_CUBES))
        for i in range(_NUM_CUBES):
            addr = self._cube_qpos_addrs[i]
            self.data.qpos[addr]   = float(self.np_random.uniform(*CUBE_X_RANGE))
            self.data.qpos[addr+1] = float(self.np_random.uniform(*CUBE_Y_RANGE))
            self.data.qpos[addr+2] = CUBE_Z_START
            self.data.qpos[addr+3] = 1.0
            self.data.qpos[addr+4:addr+7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        # Pin the target cube to TCP immediately (it's "already grasped")
        self._pin_cube_to_tcp()
        mujoco.mj_forward(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        return self._get_obs(ee_pos, cube_pos), {"target_cube": f"cube_{self._target_cube_idx}"}

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
