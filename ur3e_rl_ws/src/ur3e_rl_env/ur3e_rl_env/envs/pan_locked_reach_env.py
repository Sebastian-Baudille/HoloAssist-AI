"""
pan_locked_reach_env.py — Sub-policy 2: extend + descend to above cube.

Replaces the separate extend and grasp sub-policies. Pan is locked at the
optimal angle; this policy controls lift/elbow/wrist_1 to move the EE from
pan's home pose all the way to TARGET_Z_ABOVE above the cube.

Obs (5D):    [xy_dist_norm, z_err_norm, lift_norm, elbow_norm, wrist1_norm]
Action (3D): [Δlift, Δelbow, Δwrist_1] × 0.05 rad/step
Fixed:       pan=optimal, wrist_2=-π/2, wrist_3=computed from cube yaw
Success:     EE within jaw XY margin and Z tolerance
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

MAX_STEPS     = int(os.getenv("UR3E_RL_MAX_EPISODE_STEPS", "400"))
PHYSICS_STEPS = 10
LIFT_SCALE    = 0.05
ELBOW_SCALE   = 0.05
W1_SCALE      = 0.05

LIFT_LIM  = (-2.5, -0.3)
ELBOW_LIM = (0.2,   2.5)
W1_LIM    = (-np.pi, np.pi / 2)
_FIXED_W2 = -np.pi / 2
# wrist_3 is computed per-episode from cube yaw — see _w3_from_cube_yaw()

TARGET_Z_ABOVE  = 0.015   # target this far above cube centre
JAW_XY_MARGIN = 0.009   # tight centre requirement — cube must be well inside jaws
JAW_Z_TOL     = 0.012   # ±12mm Z tolerance
FLOOR_Z       = 0.005   # kill episode if EE literally hits the ground

CUBE_Z     = 0.02
CUBE_R_MIN = 0.40   # matches pan_env — polar spawn
CUBE_R_MAX = 0.48

_HOME_LIFT   = -np.pi / 2
_HOME_ELBOW  = 1.7
_RESET_NOISE = 0.15

MAX_DIST = 0.70


def _w3_from_cube_yaw(cube_quat: np.ndarray) -> float:
    """Compute wrist_3 angle to align gripper jaws with cube's horizontal axis.

    cube_quat: (w, x, y, z) from data.xquat.
    The cube has 4-fold symmetry so we snap to the nearest π/2 multiple.
    """
    w, x, y, z = cube_quat
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    yaw_snapped = round(yaw / (np.pi / 2)) * (np.pi / 2)
    return float(np.clip(np.pi / 2 + yaw_snapped, -np.pi, np.pi))


class UR3ePanLockedReachEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        super().__init__()
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
        self._locked_w3  = np.pi / 2
        self._target     = np.zeros(3, dtype=np.float32)
        self._prev_dist  = 0.0
        self.step_count  = 0
        self.render_mode = render_mode
        self._viewer     = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

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
            float(np.clip(xy_dist          / 0.6,       0,  1)),
            float(np.clip(z_err            / 0.5,      -1,  1)),
            float(np.clip(joints[1] / np.pi,            -1,  1)),
            float(np.clip(joints[2] / np.pi,            -1,  1)),
            float(np.clip(joints[3] / (np.pi / 2),      -1,  1)),
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(3), -1, 1)

        joints    = self._get_joints().astype(np.float64)
        new_lift  = float(np.clip(joints[1] + action[0]*LIFT_SCALE,  *LIFT_LIM))
        new_elbow = float(np.clip(joints[2] + action[1]*ELBOW_SCALE, *ELBOW_LIM))
        new_w1    = float(np.clip(joints[3] + action[2]*W1_SCALE,    *W1_LIM))

        q = joints.copy()
        q[0] = self._locked_pan
        q[1] = new_lift
        q[2] = new_elbow
        q[3] = new_w1
        q[4] = _FIXED_W2
        q[5] = self._locked_w3
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            mujoco.mj_step(self.model, self.data)

        ee_pos  = self._get_ee()
        joints  = self._get_joints()
        xy_dist  = float(np.linalg.norm(ee_pos[:2] - self._target[:2]))
        z_err    = float(ee_pos[2] - self._target[2])
        dist     = float(np.linalg.norm(ee_pos - self._target))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        floor_hit = float(ee_pos[2]) < FLOOR_Z
        success   = xy_dist < JAW_XY_MARGIN and abs(z_err) < JAW_Z_TOL
        reward    = (-2.0 * (dist / MAX_DIST) + progress * 3.0
                     + (50.0 if success else 0.0)
                     + (-5.0 if floor_hit else 0.0))

        if self.render_mode == "human" and self._viewer:
            self._viewer.sync()

        return (self._obs(joints, ee_pos), reward,
                success or floor_hit, self.step_count >= MAX_STEPS,
                {"is_success": success, "dist_to_target": dist,
                 "xy_dist": xy_dist, "z_err": z_err, "floor_hit": floor_hit})

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        # Place cube with polar spawn matching pan_env, plus random yaw
        addr     = self._cube_qpos_addrs[0]
        radius   = float(self.np_random.uniform(CUBE_R_MIN, CUBE_R_MAX))
        angle    = float(self.np_random.uniform(-np.pi / 3, np.pi / 3))
        cx       = radius * np.sin(angle)
        cy       = -radius * np.cos(angle)
        yaw      = float(self.np_random.uniform(0, np.pi / 2))
        half_yaw = yaw / 2.0
        self.data.qpos[addr:addr+3] = [cx, cy, CUBE_Z]
        self.data.qpos[addr+3]      = np.cos(half_yaw)
        self.data.qpos[addr+4]      = 0.0
        self.data.qpos[addr+5]      = 0.0
        self.data.qpos[addr+6]      = np.sin(half_yaw)

        cube_pos         = np.array([cx, cy, CUBE_Z], dtype=np.float32)
        self._locked_pan = float(_optimal_pan(cube_pos, self._phi0))
        self._locked_w3  = _w3_from_cube_yaw(
            np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)]))
        self._target     = cube_pos + np.array([0.0, 0.0, TARGET_Z_ABOVE], dtype=np.float32)

        lift  = float(np.clip(self.np_random.normal(_HOME_LIFT,  _RESET_NOISE), *LIFT_LIM))
        elbow = float(np.clip(self.np_random.normal(_HOME_ELBOW, _RESET_NOISE), *ELBOW_LIM))

        q_test = np.array([self._locked_pan, lift, elbow, -np.pi/2, _FIXED_W2, self._locked_w3])
        if forward_kinematics(q_test)[2] < 0.02:
            lift, elbow = _HOME_LIFT, _HOME_ELBOW

        q = np.array([self._locked_pan, lift, elbow, -np.pi/2, _FIXED_W2, self._locked_w3])
        for i, addr2 in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr2] = q[i]
        self.data.ctrl[:6] = q

        mujoco.mj_forward(self.model, self.data)
        for _ in range(30):
            mujoco.mj_step(self.model, self.data)

        ee_pos = self._get_ee()
        joints = self._get_joints()
        self._prev_dist = float(np.linalg.norm(ee_pos - self._target))

        return self._obs(joints, ee_pos), {}

    def close(self):
        if self._viewer:
            self._viewer.close()
            self._viewer = None
