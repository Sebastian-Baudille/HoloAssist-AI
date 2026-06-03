"""
grasp_env_chained.py — Grasp env that resets from real pan+extend handoffs.

Instead of rejection-sampling a plausible handoff pose, each episode runs the
trained pan and extend policies to completion, then hands the resulting joint
state to grasp. This means grasp trains on exactly the distribution it will see
at inference time.

Obs / action / rewards: identical to grasp_env.py.
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
from ur3e_rl_env.envs.pan_env    import (_arm_phi0, _pan_err,
                                          PAN_SCALE, PHYSICS_STEPS as PAN_PS)
from ur3e_rl_env.envs.extend_env import (LIFT_SCALE as EL, ELBOW_SCALE as EE,
                                          W1_SCALE as EW1, LIFT_LIM, ELBOW_LIM,
                                          W1_LIM, _FIXED_W2 as E_W2,
                                          _FIXED_W3 as E_W3,
                                          PHYSICS_STEPS as EXT_PS,
                                          CUBE_CX_MIN, CUBE_CX_MAX,
                                          CUBE_CY, CUBE_Z,
                                          _HOME_LIFT, _HOME_ELBOW)
from ur3e_rl_env.envs.grasp_env  import TARGET_Z_ABOVE, MAX_DIST, SUCCESS_DIST

_SRC_DIR  = Path(__file__).parent.parent.parent
SCENE_XML = str(_SRC_DIR / "assets" / "mujoco" / "scene_reach.xml")

MAX_STEPS     = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "300"))
PHYSICS_STEPS = 10
LIFT_SCALE    = 0.05
ELBOW_SCALE   = 0.05
W1_SCALE      = 0.05

LIFT_LIM_G  = (-2.5, -0.3)
ELBOW_LIM_G = (0.2,   2.5)
W1_LIM_G    = (-np.pi, np.pi / 2)
_FIXED_W2   = -np.pi / 2
_FIXED_W3   =  np.pi / 2

PAN_DONE_RAD  = np.radians(10)
EXTEND_DONE_M = 0.08
MAX_PAN_STEPS    = 200
MAX_EXTEND_STEPS = 200


class UR3eGraspChainedEnv(gym.Env):
    """Grasp env with chained reset: runs pan+extend policies each episode."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, pan_model_path: str, extend_model_path: str,
                 render_mode=None):
        super().__init__()
        # Import here to avoid circular / heavy import at module level
        from stable_baselines3 import PPO
        self._pan_model    = PPO.load(pan_model_path,    device="cpu")
        self._extend_model = PPO.load(extend_model_path, device="cpu")

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)

        self.observation_space = spaces.Box(-1.0, 1.0, (5,), dtype=np.float32)
        self.action_space      = spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)

        self._ik_cache        = build_ik_cache(self.model)
        self._cube_body_ids   = [self.model.body("cube_0").id]
        self._cube_qpos_addrs = [int(self.model.jnt_qposadr[
                                     self.model.body("cube_0").jntadr[0]])]

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

    # ── helpers ──────────────────────────────────────────────────────────────

    def _get_ee(self):
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube(self):
        return self.data.xpos[self._cube_body_ids[0]].astype(np.float32)

    def _get_joints(self):
        return np.array([self.data.qpos[a]
                         for a in self._ik_cache["arm_qpos_addrs"]], dtype=np.float32)

    def _obs(self, joints, ee_pos):
        xy_dist = float(np.linalg.norm(ee_pos[:2] - self._target[:2]))
        z_err   = float(ee_pos[2] - self._target[2])
        return np.array([
            float(np.clip(xy_dist           / 0.6,        0,  1)),
            float(np.clip(z_err             / 0.5,       -1,  1)),
            float(np.clip(joints[1] / np.pi,             -1,  1)),
            float(np.clip(joints[2] / np.pi,             -1,  1)),
            float(np.clip(joints[3] / (np.pi / 2),       -1,  1)),
        ], dtype=np.float32)

    # ── chained reset ─────────────────────────────────────────────────────────

    def _place_cube(self, cx, cy):
        addr = self._cube_qpos_addrs[0]
        self.data.qpos[addr:addr+3] = [cx, cy, CUBE_Z]
        self.data.qpos[addr+3] = 1.0
        self.data.qpos[addr+4:addr+7] = 0.0

    def _home_pose(self, pan=0.0):
        q = np.array([pan, _HOME_LIFT, _HOME_ELBOW, -np.pi/2, E_W2, E_W3])
        for i, a in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[a] = q[i]
        self.data.ctrl[:6] = q
        mujoco.mj_forward(self.model, self.data)

    def _run_pan(self, cube_pos):
        """Run pan policy until within PAN_DONE_RAD or step limit."""
        for _ in range(MAX_PAN_STEPS):
            joints = self._get_joints()
            pan    = float(joints[0])
            err    = _pan_err(pan, cube_pos, self._phi0)
            if abs(err) < PAN_DONE_RAD:
                break
            pan_obs = np.array([np.cos(err), np.sin(err)], dtype=np.float32)
            action, _ = self._pan_model.predict(pan_obs, deterministic=True)
            dpan   = float(np.clip(action.reshape(1)[0], -1, 1)) * PAN_SCALE
            q      = joints.astype(np.float64).copy()
            q[0]   = float(joints[0]) + dpan
            self.data.ctrl[:6] = q
            for _ in range(PAN_PS):
                mujoco.mj_step(self.model, self.data)

    def _run_extend(self, cube_pos):
        """Run extend policy until EE XY-dist ≤ EXTEND_DONE_M or step limit."""
        for _ in range(MAX_EXTEND_STEPS):
            joints  = self._get_joints()
            ee_pos  = self._get_ee()
            xy_dist = float(np.linalg.norm(ee_pos[:2] - cube_pos[:2]))
            if xy_dist <= EXTEND_DONE_M:
                break
            z_err = float(ee_pos[2] - cube_pos[2])
            ext_obs = np.array([
                np.clip(xy_dist          / 0.6,       0,  1),
                np.clip(z_err            / 0.5,      -1,  1),
                np.clip(joints[1] / np.pi,            -1,  1),
                np.clip(joints[2] / np.pi,            -1,  1),
                np.clip(joints[3] / (np.pi / 2),      -1,  1),
            ], dtype=np.float32)
            action, _ = self._extend_model.predict(ext_obs, deterministic=True)
            delta     = np.clip(action, -1, 1)
            q         = joints.astype(np.float64).copy()
            q[1] = float(np.clip(joints[1] + delta[0]*EL,  *LIFT_LIM))
            q[2] = float(np.clip(joints[2] + delta[1]*EE,  *ELBOW_LIM))
            q[3] = float(np.clip(joints[3] + delta[2]*EW1, *W1_LIM))
            q[4] = E_W2
            q[5] = E_W3
            self.data.ctrl[:6] = q
            for _ in range(EXT_PS):
                mujoco.mj_step(self.model, self.data)

    # ── gym interface ─────────────────────────────────────────────────────────

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1, 1)

        joints    = self._get_joints().astype(np.float64)
        new_lift  = float(np.clip(joints[1] + action[0]*LIFT_SCALE,  *LIFT_LIM_G))
        new_elbow = float(np.clip(joints[2] + action[1]*ELBOW_SCALE, *ELBOW_LIM_G))
        new_w1    = float(np.clip(joints[3] + action[2]*W1_SCALE,    *W1_LIM_G))

        q      = joints.copy()
        q[0]   = self._locked_pan
        q[1]   = new_lift
        q[2]   = new_elbow
        q[3]   = new_w1
        q[4]   = _FIXED_W2
        q[5]   = _FIXED_W3
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        ee_pos   = self._get_ee()
        joints   = self._get_joints()
        dist     = float(np.linalg.norm(ee_pos - self._target))
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

        cx = float(self.np_random.uniform(CUBE_CX_MIN, CUBE_CX_MAX))
        self._place_cube(cx, CUBE_CY)
        self._home_pose(pan=0.0)

        cube_pos = self._get_cube()
        self._run_pan(cube_pos)
        self._run_extend(cube_pos)

        # Lock pan at wherever pan policy left it
        self._locked_pan = float(self._get_joints()[0])
        cube_pos         = self._get_cube()   # re-fetch (cube may have shifted slightly)
        self._target     = cube_pos + np.array([0.0, 0.0, TARGET_Z_ABOVE], dtype=np.float32)

        ee_pos  = self._get_ee()
        joints  = self._get_joints()
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target))

        return self._obs(joints, ee_pos), {}

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
