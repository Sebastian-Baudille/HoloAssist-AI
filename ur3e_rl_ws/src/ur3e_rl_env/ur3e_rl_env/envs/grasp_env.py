"""
grasp_env.py — Sub-policy 3: position EE on top of cube.

Starts from extend's handoff (EE above cube). Task: descend to TARGET_Z_ABOVE
directly above the cube. Wrist_3 is fixed at π/2 (pre-rotated for grasp).

Obs (5D):    [xy_dist_norm, z_err_norm, lift_norm, elbow_norm, wrist1_norm]
Action (3D): [Δlift, Δelbow, Δwrist_1] × scale
Fixed:       pan=locked, wrist_2=-π/2, wrist_3=π/2

Reward: -dist_to_target/MAX_DIST + progress*3 + success bonus
Success: 3D dist to target < 2 cm
"""
from __future__ import annotations
import os
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ur3e_rl_env.ik import build_ik_cache
from ur3e_rl_env.kinematics import forward_kinematics
from ur3e_rl_env.envs.pan_env import _arm_phi0, _optimal_pan

_SRC_DIR  = Path(__file__).parent.parent.parent
SCENE_XML = str(_SRC_DIR / "assets" / "mujoco" / "scene_reach.xml")

MAX_STEPS     = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "300"))
PHYSICS_STEPS = 10
LIFT_SCALE    = 0.05
ELBOW_SCALE   = 0.05
W1_SCALE      = 0.05
SUCCESS_DIST  = 0.02

LIFT_LIM  = (-2.5, -0.3)
ELBOW_LIM = (0.2,   2.5)
W1_LIM    = (-np.pi, np.pi / 2)
_FIXED_W2 = -np.pi / 2
_FIXED_W3 = np.pi / 2

# Target: this far above cube centre
TARGET_Z_ABOVE = 0.015

# Cube spawning — same as extend
CUBE_Z      = 0.02
CUBE_CY     = -0.13
CUBE_CX_MIN = -0.50
CUBE_CX_MAX = -0.22

# Extend handoff zone
HANDOFF_Z_MIN      = 0.05
HANDOFF_Z_MAX      = 0.10
HANDOFF_XY_RADIUS  = 0.06

_HOME_LIFT  = -np.pi / 2
_HOME_ELBOW = 1.7

MAX_DIST = 0.70


class UR3eGraspEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self._ik_cache        = build_ik_cache(self.model)
        self._cube_body_ids   = [self.model.body("cube_0").id]
        self._cube_qpos_addrs = [int(self.model.jnt_qposadr[self.model.body("cube_0").jntadr[0]])]

        self._phi0       = _arm_phi0(self.model, self.data,
                                     self._ik_cache["tcp_body_id"],
                                     self._ik_cache["arm_qpos_addrs"])
        self._locked_pan = 0.0
        self._target     = np.zeros(3, dtype=np.float32)
        self._prev_dist  = 0.0
        self.step_count  = 0
        self.render_mode = render_mode
        self._viewer     = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_ee_pos(self):
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self):
        return self.data.xpos[self._cube_body_ids[0]].astype(np.float32)

    def _get_joints(self):
        return np.array([self.data.qpos[a] for a in self._ik_cache["arm_qpos_addrs"]],
                        dtype=np.float32)

    def _obs(self, joints, ee_pos):
        xy_dist = float(np.linalg.norm(ee_pos[:2] - self._target[:2]))
        z_err   = float(ee_pos[2] - self._target[2])
        return np.array([
            float(np.clip(xy_dist        / 0.6,  0,  1)),
            float(np.clip(z_err          / 0.5, -1,  1)),
            float(np.clip(joints[1] / np.pi,    -1,  1)),   # lift
            float(np.clip(joints[2] / np.pi,    -1,  1)),   # elbow
            float(np.clip(joints[3] / (np.pi / 2), -1, 1)),   # wrist_1 (range -π to π/2)
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1, 1)

        joints    = self._get_joints().astype(np.float64)
        new_lift  = float(np.clip(joints[1] + action[0] * LIFT_SCALE,  *LIFT_LIM))
        new_elbow = float(np.clip(joints[2] + action[1] * ELBOW_SCALE, *ELBOW_LIM))
        new_w1    = float(np.clip(joints[3] + action[2] * W1_SCALE,    *W1_LIM))

        q = joints.copy()
        q[0] = self._locked_pan
        q[1] = new_lift
        q[2] = new_elbow
        q[3] = new_w1
        q[4] = _FIXED_W2
        q[5] = _FIXED_W3
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        ee_pos  = self._get_ee_pos()
        joints  = self._get_joints()
        dist    = float(np.linalg.norm(ee_pos - self._target))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        success = dist < SUCCESS_DIST
        reward  = -2.0 * (dist / MAX_DIST) + progress * 3.0 + (50.0 if success else 0.0)

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return (self._obs(joints, ee_pos), reward,
                success, self.step_count >= MAX_STEPS,
                {"is_success": success, "dist_to_target": dist})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        addr = self._cube_qpos_addrs[0]
        cx   = float(self.np_random.uniform(CUBE_CX_MIN, CUBE_CX_MAX))
        cy   = CUBE_CY
        self.data.qpos[addr:addr+3] = [cx, cy, CUBE_Z]
        self.data.qpos[addr+3] = 1.0
        self.data.qpos[addr+4:addr+7] = 0.0

        self._locked_pan = 0.0
        self._target     = np.array([cx, cy, CUBE_Z + TARGET_Z_ABOVE], dtype=np.float32)

        # Rejection-sample until EE is above the cube (simulating extend handoff)
        lift, elbow = _HOME_LIFT, _HOME_ELBOW  # fallback
        for _ in range(50):
            l = float(self.np_random.uniform(-2.0, -0.5))
            e = float(self.np_random.uniform(0.5, 2.5))
            q_test  = np.array([self._locked_pan, l, e, -np.pi/2, _FIXED_W2, _FIXED_W3])
            ee_test = forward_kinematics(q_test)
            xy_err  = float(np.linalg.norm(ee_test[:2] - np.array([cx, cy])))
            z_above = float(ee_test[2]) - CUBE_Z
            if xy_err < HANDOFF_XY_RADIUS and HANDOFF_Z_MIN < z_above < HANDOFF_Z_MAX:
                lift, elbow = l, e
                break

        q = np.array([self._locked_pan, lift, elbow, -np.pi/2, _FIXED_W2, _FIXED_W3])
        for i, addr2 in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr2] = q[i]
        self.data.ctrl[:6] = q

        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        ee_pos  = self._get_ee_pos()
        joints  = self._get_joints()
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target))

        return self._obs(joints, ee_pos), {}

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
