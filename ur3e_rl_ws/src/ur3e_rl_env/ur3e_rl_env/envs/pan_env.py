"""
pan_env.py — Sub-policy 1: rotate shoulder_pan to face the cube.

Obs (2D):    [cos(pan_err), sin(pan_err)]
Action (1D): [Δpan] × 0.08 rad/step
Fixed:       lift, elbow, wrists all at HOME

Reward: cos(pan_err)   — +1 facing, -1 opposite
Success: |pan_err| < 15°
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

from ur3e_rl_env.ik import build_ik_cache

_SRC_DIR  = Path(__file__).parent.parent.parent
SCENE_XML = str(_SRC_DIR / "assets" / "mujoco" / "scene_reach.xml")

MAX_STEPS     = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "300"))
PHYSICS_STEPS = 20
PAN_SCALE     = 0.04   # finer control — 2.3°/step so arm can settle within 5°
SUCCESS_RAD   = np.radians(5)

# All joints fixed except pan
_FIXED = dict(lift=-np.pi/2, elbow=1.7, w1=-np.pi/2, w2=-np.pi/2, w3=np.pi/2)

CUBE_Z      = 0.02
CUBE_R_MIN  = 0.40
CUBE_R_MAX  = 0.48
_TOTAL_CUBES = 1


def _arm_phi0(model: mujoco.MjModel, data: mujoco.MjData, tcp_body_id: int,
              arm_qpos_addrs: list) -> float:
    """XY angle of the arm's reach direction when shoulder_pan=0.

    Empirically calibrated: at pan=0 the arm naturally reaches a cube placed
    at spawn angle -73° (cx=-0.421, cy=-0.129). The FK-derived EE angle
    (-152.9°) under-estimates by ~10° due to arm offset geometry, so we use
    the reference-cube direction instead.
    """
    ref_cx = 0.44 * np.sin(np.radians(-73))
    ref_cy = -0.44 * np.cos(np.radians(-73))
    return float(np.arctan2(ref_cy, ref_cx))   # -163.0 deg = -2.845 rad


def _optimal_pan(cube_pos: np.ndarray, phi0: float) -> float:
    """Pan angle (rad) that aligns the arm's sagittal plane with the cube XY."""
    return float(np.arctan2(cube_pos[1], cube_pos[0])) - phi0


def _pan_err(pan: float, cube_pos: np.ndarray, phi0: float) -> float:
    optimal = _optimal_pan(cube_pos, phi0)
    return (pan - optimal + np.pi) % (2 * np.pi) - np.pi


class UR3ePanEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        self._ik_cache = build_ik_cache(self.model)
        self._cube_body_ids   = [self.model.body("cube_0").id]
        self._cube_qpos_addrs = [int(self.model.jnt_qposadr[self.model.body("cube_0").jntadr[0]])]

        self._phi0 = _arm_phi0(self.model, self.data,
                               self._ik_cache["tcp_body_id"],
                               self._ik_cache["arm_qpos_addrs"])

        self._current_pan = 0.0
        self.step_count   = 0
        self.render_mode  = render_mode
        self._viewer      = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_cube_pos(self):
        return self.data.xpos[self._cube_body_ids[0]].astype(np.float32)

    def _get_pan(self):
        return float(self.data.qpos[self._ik_cache["arm_qpos_addrs"][0]])

    def _obs(self, pan, cube_pos):
        err = _pan_err(pan, cube_pos, self._phi0)
        return np.array([np.cos(err), np.sin(err)], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        dpan = float(np.clip(action.reshape(1)[0], -1, 1)) * PAN_SCALE
        self._current_pan = float(np.clip(self._current_pan + dpan, -2*np.pi, 2*np.pi))

        q = np.array([self._current_pan, _FIXED["lift"], _FIXED["elbow"],
                      _FIXED["w1"], _FIXED["w2"], _FIXED["w3"]])
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        cube_pos = self._get_cube_pos()
        pan      = self._get_pan()
        err      = _pan_err(pan, cube_pos, self._phi0)

        success = abs(err) < SUCCESS_RAD
        # cos(err) - 1.0 is always ≤ 0 (0 only when perfect alignment).
        # Hovering near threshold is now costly; early success is always better.
        reward  = float(np.cos(err)) - 1.0 + (20.0 if success else 0.0)

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return (self._obs(pan, cube_pos), reward,
                success, self.step_count >= MAX_STEPS,
                {"is_success": success, "pan_err_deg": abs(err)*180/np.pi,
                 "optimal_pan": _optimal_pan(cube_pos, self._phi0)})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # Random starting pan — full range so policy learns all directions
        self._current_pan = float(self.np_random.uniform(-np.pi, np.pi))
        q = np.array([self._current_pan, _FIXED["lift"], _FIXED["elbow"],
                      _FIXED["w1"], _FIXED["w2"], _FIXED["w3"]])
        for i, addr in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr] = q[i]
        self.data.ctrl[:6] = q

        addr   = self._cube_qpos_addrs[0]
        radius = float(self.np_random.uniform(CUBE_R_MIN, CUBE_R_MAX))
        angle  = float(self.np_random.uniform(-np.pi/3, np.pi/3))
        cx, cy = radius * np.sin(angle), -radius * np.cos(angle)
        self.data.qpos[addr:addr+3] = [cx, cy, CUBE_Z]
        self.data.qpos[addr+3] = 1.0
        self.data.qpos[addr+4:addr+7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        cube_pos = self._get_cube_pos()
        pan      = self._get_pan()
        return self._obs(pan, cube_pos), {}

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
