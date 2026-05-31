"""
reach_env.py — Stage 1: move TCP to the nearest cube.

Obs (6D): normalised [EE_x, EE_y, EE_z, cube_x, cube_y, cube_z]
Action (3D): Cartesian delta (dx, dy, dz) in [-1, 1]; scaled to ±2 cm/step.
Reward: -dist(EE, cube)  (dense, always ≤ 0; maximised at 0 = touching)
Done:   dist < GRASP_DIST_M (success) OR step >= MAX_STEPS (timeout)

The IK layer (ik.py) converts the 3D action to joint targets while keeping
the gripper pointing down toward the table.
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
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
)
from ur3e_rl_env.ik import build_ik_cache, cartesian_to_joint_targets

_SRC_DIR    = Path(__file__).parent.parent.parent
SCENE_XML   = str(_SRC_DIR / "assets" / "mujoco" / "scene.xml")

MAX_STEPS       = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "200"))
PHYSICS_STEPS   = 50          # physics steps per RL step (0.1 s sim time)
GRASP_DIST_M    = 0.05        # success threshold: 5 cm
HOME_JOINTS     = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

CUBE_X_RANGE = (float(os.getenv("UR3E_RL_CUBE_X_MIN", "-0.20")),
                float(os.getenv("UR3E_RL_CUBE_X_MAX",  "0.20")))
CUBE_Y_RANGE = (float(os.getenv("UR3E_RL_CUBE_Y_MIN", "-0.45")),
                float(os.getenv("UR3E_RL_CUBE_Y_MAX", "-0.10")))
CUBE_Z       = float(os.getenv("UR3E_RL_CUBE_Z", "1.11"))
_NUM_CUBES   = 4


def _norm(v: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-6)
    return float(np.clip(2.0 * (v - lo) / span - 1.0, -1.0, 1.0))


# Intentional copy of ros_interface._normalize_xyz — keeps this file ROS-free.
def _norm_xyz(pos: np.ndarray) -> np.ndarray:
    return np.array([
        _norm(float(pos[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
        _norm(float(pos[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
        _norm(float(pos[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
    ], dtype=np.float32)


class UR3eReachEnv(gym.Env):
    """Reach: move TCP to within GRASP_DIST_M of the nearest cube."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode: str | None = None) -> None:
        super().__init__()
        self.model  = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data   = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (6,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self._joint_lower = np.array(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float64)
        self._joint_upper = np.array(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float64)
        self._ik_cache    = build_ik_cache(self.model)

        # Cube body IDs and qpos/dof addresses
        self._cube_body_ids  = [self.model.body(f"cube_{i}").id for i in range(_NUM_CUBES)]
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
        self._renderer: mujoco.Renderer | None = None
        if render_mode == "rgb_array":
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self, idx: int) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids[idx]].astype(np.float32)

    def _nearest_cube(self, ee_pos: np.ndarray) -> int:
        dists = [np.linalg.norm(self._get_cube_pos(i) - ee_pos) for i in range(_NUM_CUBES)]
        return int(np.argmin(dists))

    def _get_obs(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray:
        # Note: WORKSPACE_Z_MAX=1.22 is below HOME TCP Z≈1.79, so obs[2] saturates
        # at 1.0 during initial arm descent (~57 cm). Policy still learns via X/Y
        # gradient and the overall distance reward.
        return np.concatenate([_norm_xyz(ee_pos), _norm_xyz(cube_pos)]).astype(np.float32)

    def step(self, action: np.ndarray):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float32).reshape(3), -1.0, 1.0)

        q_target = cartesian_to_joint_targets(
            self.model, self.data, self._ik_cache,
            delta_xyz=action.astype(np.float64),
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
        )
        self.data.ctrl[:6] = q_target
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos(self._target_cube_idx)
        dist     = float(np.linalg.norm(ee_pos - cube_pos))

        reward     = -dist
        success    = dist < GRASP_DIST_M
        terminated = success
        truncated  = self.step_count >= MAX_STEPS
        info       = {"is_success": success, "dist_to_cube": dist}

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

        for i in range(_NUM_CUBES):
            addr = self._cube_qpos_addrs[i]
            self.data.qpos[addr]   = float(self.np_random.uniform(*CUBE_X_RANGE))
            self.data.qpos[addr+1] = float(self.np_random.uniform(*CUBE_Y_RANGE))
            self.data.qpos[addr+2] = CUBE_Z
            self.data.qpos[addr+3] = 1.0
            self.data.qpos[addr+4:addr+7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee_pos()
        self._target_cube_idx = self._nearest_cube(ee_pos)
        cube_pos = self._get_cube_pos(self._target_cube_idx)

        return self._get_obs(ee_pos, cube_pos), {"target_cube": f"cube_{self._target_cube_idx}"}

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def render(self):
        if self.render_mode == "rgb_array" and self._renderer is not None:
            self._renderer.update_scene(self.data)
            return self._renderer.render()
