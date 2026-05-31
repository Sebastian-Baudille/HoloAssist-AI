"""
coordinator.py — State machine chaining Reach → Grasp → Transport → Release.

MuJoCoCoordinator runs entirely in Python/MuJoCo with no ROS.
Loads two trained PPO models and drives the full pick-and-place sequence.

Stage transitions:
  REACH      — Model 1 runs until dist(TCP, cube) < GRASP_DIST_M
  GRASP      — pin cube to TCP kinematically (one step, no model)
  TRANSPORT  — Model 2 runs until dist(cube, bin) < RELEASE_DIST_M
  RELEASE    — unpin cube, episode done
"""
from __future__ import annotations
from enum import Enum
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from ur3e_rl_env.constants import (
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
)
from ur3e_rl_env.ik import build_ik_cache, cartesian_to_joint_targets

_SRC_DIR  = Path(__file__).parent.parent
SCENE_XML = str(_SRC_DIR / "assets" / "mujoco" / "scene.xml")

GRASP_DIST_M   = 0.05
RELEASE_DIST_M = 0.08
PHYSICS_STEPS  = 50
HOME_JOINTS    = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
# 3 cm below TCP — approximately half-cube-height (cube is 4 cm tall)
_HOLD_OFFSET   = np.array([0.0, 0.0, -0.03])
_NUM_CUBES     = 4

CUBE_X_RANGE = (-0.20, 0.20)
CUBE_Y_RANGE = (-0.45, -0.10)
CUBE_Z       = 1.11
_BIN_POS     = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z], dtype=np.float32)


class Stage(Enum):
    REACH     = "reach"
    GRASP     = "grasp"
    TRANSPORT = "transport"
    RELEASE   = "release"
    DONE      = "done"


def _norm(v, lo, hi):
    # Intentional copy of ros_interface._normalize_xyz — keeps this file ROS-free.
    span = max(hi - lo, 1e-6)
    return float(np.clip(2.0 * (v - lo) / span - 1.0, -1.0, 1.0))


def _norm_xyz(pos):
    return np.array([
        _norm(float(pos[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
        _norm(float(pos[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
        _norm(float(pos[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
    ], dtype=np.float32)


class MuJoCoCoordinator:
    """
    Full pick-and-place coordinator using two trained PPO models.
    Call reset() to start a new episode, then step() in a loop.
    """

    def __init__(
        self,
        reach_model_path:     str,
        transport_model_path: str,
        render_mode:          str | None = None,
        rng_seed:             int = 0,
    ) -> None:
        from stable_baselines3 import PPO

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML)
        self.data  = mujoco.MjData(self.model)
        self._ik_cache     = build_ik_cache(self.model)
        self._joint_lower  = np.array(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float64)
        self._joint_upper  = np.array(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float64)
        self._rng          = np.random.default_rng(rng_seed)
        self._target_cube  = 0
        self._grasped      = False
        self.stage         = Stage.DONE

        self._cube_body_ids   = [self.model.body(f"cube_{i}").id for i in range(_NUM_CUBES)]
        self._cube_qpos_addrs = []
        self._cube_dof_addrs  = []
        for i in range(_NUM_CUBES):
            jnt_id = self.model.body(f"cube_{i}").jntadr[0]
            self._cube_qpos_addrs.append(int(self.model.jnt_qposadr[jnt_id]))
            self._cube_dof_addrs.append(int(self.model.jnt_dofadr[jnt_id]))

        print(f"Loading reach model:     {reach_model_path}")
        self._reach_model     = PPO.load(reach_model_path, device="cpu")
        print(f"Loading transport model: {transport_model_path}")
        self._transport_model = PPO.load(transport_model_path, device="cpu")

        self._viewer = None
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _get_tcp_pos(self) -> np.ndarray:
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids[self._target_cube]].astype(np.float32)

    def _apply_cartesian(self, action: np.ndarray) -> None:
        q = cartesian_to_joint_targets(
            self.model, self.data, self._ik_cache,
            delta_xyz=action.astype(np.float64),
            joint_lower=self._joint_lower,
            joint_upper=self._joint_upper,
        )
        self.data.ctrl[:6] = q
        for _ in range(PHYSICS_STEPS):
            if self._grasped:
                self._pin_cube()
            mujoco.mj_step(self.model, self.data)
        if self._grasped:
            self._pin_cube()

    def _pin_cube(self) -> None:
        tcp_pos  = self.data.xpos[self._ik_cache["tcp_body_id"]].copy()
        tcp_quat = self.data.xquat[self._ik_cache["tcp_body_id"]].copy()
        addr     = self._cube_qpos_addrs[self._target_cube]
        dof      = self._cube_dof_addrs[self._target_cube]
        self.data.qpos[addr:addr+3]   = tcp_pos + _HOLD_OFFSET
        self.data.qpos[addr+3:addr+7] = tcp_quat
        self.data.qvel[dof:dof+6]     = 0.0

    def _nearest_cube(self) -> int:
        tcp = self._get_tcp_pos()
        dists = [np.linalg.norm(self.data.xpos[self._cube_body_ids[i]] - tcp)
                 for i in range(_NUM_CUBES)]
        return int(np.argmin(dists))

    def _reach_obs(self) -> np.ndarray:
        ee  = self._get_tcp_pos()
        cub = self._get_cube_pos()
        return np.concatenate([_norm_xyz(ee), _norm_xyz(cub)]).astype(np.float32)

    def _transport_obs(self) -> np.ndarray:
        ee       = self._get_tcp_pos()
        cub      = self._get_cube_pos()
        dist_bin = float(np.linalg.norm(cub - _BIN_POS))
        dist_norm = float(np.clip(2.0 * dist_bin / 1.0 - 1.0, -1.0, 1.0))
        return np.concatenate([_norm_xyz(ee), _norm_xyz(cub), [dist_norm]]).astype(np.float32)

    def is_running(self) -> bool:
        if self._viewer is not None:
            return self._viewer.is_running()
        return True

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self._grasped = False

        for i, addr in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr] = HOME_JOINTS[i]

        for i in range(_NUM_CUBES):
            addr = self._cube_qpos_addrs[i]
            self.data.qpos[addr]   = float(self._rng.uniform(*CUBE_X_RANGE))
            self.data.qpos[addr+1] = float(self._rng.uniform(*CUBE_Y_RANGE))
            self.data.qpos[addr+2] = CUBE_Z
            self.data.qpos[addr+3] = 1.0
            self.data.qpos[addr+4:addr+7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        for _ in range(200):
            mujoco.mj_step(self.model, self.data)

        self._target_cube = self._nearest_cube()
        self.stage        = Stage.REACH
        print(f"  reset → target cube_{self._target_cube}, stage=REACH")

    def step(self) -> Stage:
        """Run one RL step. Returns the stage AFTER the step."""

        if self.stage == Stage.REACH:
            obs    = self._reach_obs()
            action, _ = self._reach_model.predict(obs, deterministic=True)
            self._apply_cartesian(action)
            dist = float(np.linalg.norm(self._get_tcp_pos() - self._get_cube_pos()))
            if dist < GRASP_DIST_M:
                self.stage = Stage.GRASP

        elif self.stage == Stage.GRASP:
            self._grasped = True
            self._pin_cube()
            mujoco.mj_forward(self.model, self.data)
            self.stage = Stage.TRANSPORT
            print("  grasped → stage=TRANSPORT")

        elif self.stage == Stage.TRANSPORT:
            obs    = self._transport_obs()
            action, _ = self._transport_model.predict(obs, deterministic=True)
            self._apply_cartesian(action)
            dist_bin = float(np.linalg.norm(self._get_cube_pos() - _BIN_POS))
            if dist_bin < RELEASE_DIST_M:
                self.stage = Stage.RELEASE

        elif self.stage == Stage.RELEASE:
            self._grasped = False
            mujoco.mj_forward(self.model, self.data)
            self.stage = Stage.DONE
            print(f"  released → dist_to_bin={np.linalg.norm(self._get_cube_pos() - _BIN_POS):.3f} m")

        if self._viewer is not None:
            self._viewer.sync()

        return self.stage

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
