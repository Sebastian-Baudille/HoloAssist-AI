"""
height_env.py — Sub-policy 3: adjust wrist_1 to fine-tune EE height.

Assumes pan is correct and arm is roughly extended to the cube.
Wrist_1 tips the EE slightly up/down for final Z alignment.

Obs (3D):    [z_err_norm, wrist1_norm, dist_norm]
Action (1D): [Δwrist_1] × 0.04 rad/step
Fixed:       pan=optimal, lift+elbow near solution, wrist_2=-π/2

Reward: -dist + progress*3   (only escape is getting to cube)
Success: 3D dist < 5 cm
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
from ur3e_rl_env.kinematics import compute_ik_reference
from ur3e_rl_env.envs.pan_env import _arm_phi0, _optimal_pan

_SRC_DIR  = Path(__file__).parent.parent.parent
SCENE_XML = str(_SRC_DIR / "assets" / "mujoco" / "scene_reach.xml")

MAX_STEPS     = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "100"))
PHYSICS_STEPS = 20
W1_SCALE      = 0.04
GRASP_DIST_M  = 0.05

W1_LIM    = (-np.pi, 0.0)
_FIXED_W2 = -np.pi / 2
_FIXED_W3 = 0.0

CUBE_Z     = 0.02
CUBE_R_MIN = 0.40
CUBE_R_MAX = 0.48
_TOTAL_CUBES = 1


class UR3eHeightEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (1,), dtype=np.float32)

        self._ik_cache = build_ik_cache(self.model)
        self._phi0 = _arm_phi0(self.model, self.data,
                               self._ik_cache["tcp_body_id"],
                               self._ik_cache["arm_qpos_addrs"])
        self._cube_body_ids   = [self.model.body("cube_0").id]
        self._cube_qpos_addrs = [int(self.model.jnt_qposadr[self.model.body("cube_0").jntadr[0]])]

        self._locked = np.zeros(6)   # full locked joint config (all except wrist_1)
        self._prev_dist = 0.0
        self.step_count = 0
        self.render_mode = render_mode
        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_ee_pos(self):
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self):
        return self.data.xpos[self._cube_body_ids[0]].astype(np.float32)

    def _get_joints(self):
        return np.array([self.data.qpos[a] for a in self._ik_cache["arm_qpos_addrs"]],
                        dtype=np.float32)

    def _obs(self, joints, ee_pos, cube_pos):
        z_err = float(ee_pos[2] - cube_pos[2])
        dist  = float(np.linalg.norm(ee_pos - cube_pos))
        return np.array([
            float(np.clip(z_err  / 0.3, -1, 1)),
            float(np.clip(joints[3] / np.pi, -1, 1)),   # wrist_1
            float(np.clip(dist   / 0.4, 0,  1)),
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        dw1   = float(np.clip(action.reshape(1)[0], -1, 1)) * W1_SCALE
        joints = self._get_joints().astype(np.float64)
        new_w1 = float(np.clip(joints[3] + dw1, *W1_LIM))

        q = self._locked.copy()
        q[3] = new_w1
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        joints   = self._get_joints()
        dist     = float(np.linalg.norm(ee_pos - cube_pos))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        success = dist < GRASP_DIST_M
        reward  = -dist + progress * 3.0 + (5.0 if success else 0.0)

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return (self._obs(joints, ee_pos, cube_pos), reward,
                success, self.step_count >= MAX_STEPS,
                {"is_success": success, "dist_to_cube": dist})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # Spawn cube
        addr   = self._cube_qpos_addrs[0]
        radius = float(self.np_random.uniform(CUBE_R_MIN, CUBE_R_MAX))
        angle  = float(self.np_random.uniform(-np.pi/3, np.pi/3))
        cx, cy = radius * np.sin(angle), -radius * np.cos(angle)
        self.data.qpos[addr:addr+3] = [cx, cy, CUBE_Z]
        self.data.qpos[addr+3] = 1.0
        self.data.qpos[addr+4:addr+7] = 0.0

        cube_pos = np.array([cx, cy, CUBE_Z])

        # Use IK to find a good pan+lift+elbow, then randomise wrist_1
        pan = _optimal_pan(np.array([cx, cy, CUBE_Z]), self._phi0)
        ok, ik_joints, _ = compute_ik_reference(cube_pos.astype(float), approach_height=0.0)
        if ok:
            lift, elbow = float(ik_joints[1]), float(ik_joints[2])
        else:
            lift, elbow = -np.pi/2, 1.7

        # Randomise wrist_1 so policy learns to correct it
        w1 = float(self.np_random.uniform(-np.pi, 0.0))

        q = np.array([pan, lift, elbow, w1, _FIXED_W2, _FIXED_W3])
        self._locked = np.array([pan, lift, elbow, w1, _FIXED_W2, _FIXED_W3])

        for i, addr2 in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr2] = q[i]
        self.data.ctrl[:6] = q

        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        joints   = self._get_joints()
        self._locked = joints.astype(np.float64).copy()
        self._prev_dist = float(np.linalg.norm(ee_pos - cube_pos))

        return self._obs(joints, ee_pos, cube_pos), {}

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
