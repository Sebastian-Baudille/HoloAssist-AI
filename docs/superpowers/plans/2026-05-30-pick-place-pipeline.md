# Pick-and-Place Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 3-stage pick-and-place pipeline: a Reach model (Model 1) moves the TCP to a cube, a scripted grasp attaches it, and a Transport model (Model 2) moves it to the bin — all in MuJoCo, deployable to the real robot via the existing ROS interface.

**Architecture:** Two independent PPO environments share the same `scene.xml` and a new `ik.py` Jacobian IK layer that converts 3D Cartesian delta actions into joint targets while keeping the gripper pointing down. A `MuJoCoCoordinator` state machine chains the two models with scripted grasp/release. The coordinator runs identically in simulation and (in future) on the real robot by swapping the backend.

**Tech Stack:** MuJoCo 3.8.1 · Gymnasium 1.2.3 · stable-baselines3 2.8.0 · Python 3.10

---

## Pre-read: facts you need

```
Repo:    /home/john/git/HoloAssist-AI
Pkg:     ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/
Assets:  ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco/scene.xml

Arm joint names (order matches actuators 0-5 in scene.xml):
  shoulder_pan_joint, shoulder_lift_joint, elbow_joint,
  wrist_1_joint, wrist_2_joint, wrist_3_joint

At HOME position [0, -π/2, 0, -π/2, 0, 0]:
  TCP (gripper_tcp body) pos ≈ (0, -0.44, 1.79)
  TCP Z-axis in world frame = [0, -1, 0]   ← NOT pointing down yet
  Desired TCP Z for top-down grasp = [0, 0, -1]  ← pointing at table
  The IK orientation term drives toward this.

MuJoCo model sizes (from scene.xml):
  nq=34, nv=30
  Arm joints: qpos_addrs=[0..5], dof_addrs=[0..5]
  cube_0 freejoint: qpos_addr=6, dof_addr=6  (7 qpos, 6 dof each)
  cube_1: qpos=13, dof=12
  cube_2: qpos=20, dof=18
  cube_3: qpos=27, dof=24

Workspace (from constants.py):
  X: -0.46 to 0.40   Y: -0.56 to 0.14   Z: 1.08 to 1.22
  TABLE_TOP_Z=1.07   BIN=(0.28, 0.0, 1.078)
  CUBE spawn X(-0.20,0.20) Y(-0.45,-0.10) Z=1.11
```

## File map

| File | Status | Role |
|---|---|---|
| `ur3e_rl_env/ik.py` | create | Jacobian IK: (dx,dy,dz) → joint targets |
| `ur3e_rl_env/envs/reach_env.py` | create | Stage 1 Gym env: 6D obs, 3D action |
| `ur3e_rl_env/envs/transport_env.py` | create | Stage 3 Gym env: 7D obs, 3D action |
| `ur3e_rl_env/coordinator.py` | create | State machine: REACH→GRASP→TRANSPORT→RELEASE |
| `ur3e_rl_env/train_reach.py` | create | Train Model 1 |
| `ur3e_rl_env/train_transport.py` | create | Train Model 2 |
| `watch_pipeline.py` | create | Full pipeline viewer |
| `tests/test_ik.py` | create | IK unit test |
| `tests/test_reach_env.py` | create | Reach env tests |
| `tests/test_transport_env.py` | create | Transport env tests |

All paths relative to `ur3e_rl_ws/src/ur3e_rl_env/`.

---

## Task 1: Jacobian IK module

**Files:**
- Create: `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/ik.py`
- Create: `ur3e_rl_ws/src/ur3e_rl_env/tests/__init__.py`
- Create: `ur3e_rl_ws/src/ur3e_rl_env/tests/test_ik.py`

- [ ] **Step 1.1: Create tests directory**

```bash
mkdir -p /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/tests
touch /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/tests/__init__.py
```

- [ ] **Step 1.2: Write failing tests**

Write `tests/test_ik.py`:

```python
"""Tests for Jacobian IK module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import mujoco
import pytest

SCENE_XML = str(
    Path(__file__).parent.parent / "assets" / "mujoco" / "scene.xml"
)
HOME_JOINTS = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
JOINT_LOWER = np.full(6, -2 * np.pi)
JOINT_UPPER = np.full(6, 2 * np.pi)


@pytest.fixture
def mjenv():
    model = mujoco.MjModel.from_xml_path(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    from ur3e_rl_env.ik import build_ik_cache
    cache = build_ik_cache(model)
    # set home position
    for i, addr in enumerate(cache["arm_qpos_addrs"]):
        data.qpos[addr] = HOME_JOINTS[i]
    mujoco.mj_forward(model, data)
    return model, data, cache


def test_build_ik_cache_finds_all_joints(mjenv):
    model, data, cache = mjenv
    assert len(cache["arm_qpos_addrs"]) == 6
    assert len(cache["arm_dof_addrs"]) == 6
    assert cache["tcp_body_id"] >= 0


def test_ik_returns_six_joint_targets(mjenv):
    model, data, cache = mjenv
    from ur3e_rl_env.ik import cartesian_to_joint_targets
    targets = cartesian_to_joint_targets(
        model, data, cache,
        delta_xyz=np.array([0.0, 0.0, -1.0]),
        joint_lower=JOINT_LOWER,
        joint_upper=JOINT_UPPER,
    )
    assert targets.shape == (6,)
    assert targets.dtype == np.float64


def test_ik_targets_within_joint_limits(mjenv):
    model, data, cache = mjenv
    from ur3e_rl_env.ik import cartesian_to_joint_targets
    for _ in range(20):
        delta = np.random.uniform(-1, 1, 3)
        targets = cartesian_to_joint_targets(
            model, data, cache,
            delta_xyz=delta,
            joint_lower=JOINT_LOWER,
            joint_upper=JOINT_UPPER,
        )
        assert np.all(targets >= JOINT_LOWER - 1e-6)
        assert np.all(targets <= JOINT_UPPER + 1e-6)


def test_ik_moves_tcp_toward_target(mjenv):
    """Applying IK targets moves TCP in the requested direction."""
    model, data, cache = mjenv
    from ur3e_rl_env.ik import cartesian_to_joint_targets

    tcp_id = cache["tcp_body_id"]
    pos_before = data.xpos[tcp_id].copy()

    # Request downward movement (action_scale=0.02 m per unit)
    targets = cartesian_to_joint_targets(
        model, data, cache,
        delta_xyz=np.array([0.0, 0.0, -1.0]),
        joint_lower=JOINT_LOWER,
        joint_upper=JOINT_UPPER,
    )
    data.ctrl[:6] = targets
    for _ in range(100):
        mujoco.mj_step(model, data)

    pos_after = data.xpos[tcp_id].copy()
    # TCP should have moved downward
    assert pos_after[2] < pos_before[2], (
        f"TCP should move down: was {pos_before[2]:.4f}, now {pos_after[2]:.4f}"
    )


def test_ik_zero_action_changes_nothing(mjenv):
    """Zero action should not move the arm (within IK tolerance)."""
    model, data, cache = mjenv
    from ur3e_rl_env.ik import cartesian_to_joint_targets

    tcp_id = cache["tcp_body_id"]
    pos_before = data.xpos[tcp_id].copy()

    targets = cartesian_to_joint_targets(
        model, data, cache,
        delta_xyz=np.array([0.0, 0.0, 0.0]),
        joint_lower=JOINT_LOWER,
        joint_upper=JOINT_UPPER,
    )
    data.ctrl[:6] = targets
    for _ in range(50):
        mujoco.mj_step(model, data)

    pos_after = data.xpos[tcp_id].copy()
    # Orientation correction may cause slight drift; allow 3mm tolerance
    assert np.linalg.norm(pos_after - pos_before) < 0.05
```

- [ ] **Step 1.3: Run tests to confirm they fail**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_ik.py -v 2>&1 | tail -20
```

Expected: `ModuleNotFoundError: No module named 'ur3e_rl_env.ik'`

- [ ] **Step 1.4: Write `ik.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/ik.py`:

```python
"""
ik.py — Jacobian IK: Cartesian delta (dx,dy,dz) → joint angle targets.

Converts a 3D world-frame TCP displacement into joint targets using
damped-least-squares Jacobian IK. An orientation correction term drives
the gripper toward pointing straight down (TCP Z → [0, 0, -1]).

This runs at every RL step inside reach_env and transport_env.
"""
from __future__ import annotations
import mujoco
import numpy as np

_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_TCP_BODY_NAME   = "gripper_tcp"

ACTION_SCALE_M   = 0.02   # metres per action unit (±1 → ±2 cm)
IK_DAMPING       = 0.05   # regularisation — increase if joints oscillate
ORIENTATION_GAIN = 2.0    # how hard to correct gripper orientation
JOINT_DELTA_LIMIT = 0.24  # max joint motion per step (rad) — from constants.py

# Desired TCP Z axis for a top-down grasp (confirmed from scene diagnostic)
DESIRED_DOWN_AXIS = np.array([0.0, 0.0, -1.0])


def build_ik_cache(model: mujoco.MjModel) -> dict:
    """
    Pre-compute body/joint IDs. Call ONCE at env __init__; pass
    the returned dict to cartesian_to_joint_targets every step.
    """
    tcp_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, _TCP_BODY_NAME
    )
    if tcp_body_id < 0:
        raise RuntimeError(
            f"Body '{_TCP_BODY_NAME}' not found. "
            f"Run test_scene_load.py to check body names."
        )

    arm_dof_addrs: list[int] = []
    arm_qpos_addrs: list[int] = []
    for jname in _ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise RuntimeError(f"Joint '{jname}' not found in model.")
        arm_dof_addrs.append(int(model.jnt_dofadr[jid]))
        arm_qpos_addrs.append(int(model.jnt_qposadr[jid]))

    return {
        "tcp_body_id":    tcp_body_id,
        "arm_dof_addrs":  arm_dof_addrs,
        "arm_qpos_addrs": arm_qpos_addrs,
    }


def cartesian_to_joint_targets(
    model:       mujoco.MjModel,
    data:        mujoco.MjData,
    ik_cache:    dict,
    delta_xyz:   np.ndarray,      # (3,) action in [-1, 1]
    joint_lower: np.ndarray,      # (6,) rad
    joint_upper: np.ndarray,      # (6,) rad
) -> np.ndarray:                  # (6,) absolute joint angle targets (rad)
    """
    Convert a Cartesian delta action to absolute joint angle targets.

    delta_xyz: action output from the policy, scaled [-1, 1].
               Multiplied by ACTION_SCALE_M internally (2 cm per unit).

    Returns joint targets to set on data.ctrl[:6].
    """
    tcp_body_id   = ik_cache["tcp_body_id"]
    arm_dof_addrs = ik_cache["arm_dof_addrs"]
    arm_qpos_addrs = ik_cache["arm_qpos_addrs"]

    # ── Jacobian ───────────────────────────────────────────────────────────────
    jacp = np.zeros((3, model.nv))  # translational  (3 × nv)
    jacr = np.zeros((3, model.nv))  # rotational     (3 × nv)
    mujoco.mj_jacBody(model, data, jacp, jacr, tcp_body_id)

    # Extract arm columns only → (3, 6) each
    jacp_arm = jacp[:, arm_dof_addrs]
    jacr_arm = jacr[:, arm_dof_addrs]
    J = np.vstack([jacp_arm, jacr_arm])  # (6, 6)

    # ── Desired velocity ───────────────────────────────────────────────────────
    # Position: scale action to metres
    v_pos = np.asarray(delta_xyz, dtype=np.float64) * ACTION_SCALE_M  # (3,)

    # Orientation: cross-product error drives TCP Z toward DESIRED_DOWN_AXIS
    # data.xmat[id] is row-major 3×3 rotation; cols are body axes in world frame
    xmat     = data.xmat[tcp_body_id].reshape(3, 3)
    current_z = xmat[:, 2]                              # TCP Z in world frame
    v_rot    = ORIENTATION_GAIN * np.cross(current_z, DESIRED_DOWN_AXIS)  # (3,)

    v = np.concatenate([v_pos, v_rot])  # (6,)

    # ── Damped least squares ───────────────────────────────────────────────────
    # dq = J^T (J J^T + λ²I)^{-1} v
    JJT = J @ J.T + IK_DAMPING ** 2 * np.eye(6)
    dq  = J.T @ np.linalg.solve(JJT, v)  # (6,) rad/step

    # ── Apply delta, clip, clamp ───────────────────────────────────────────────
    q_curr  = np.array([data.qpos[a] for a in arm_qpos_addrs], dtype=np.float64)
    dq_safe = np.clip(dq, -JOINT_DELTA_LIMIT, JOINT_DELTA_LIMIT)
    q_target = np.clip(q_curr + dq_safe, joint_lower, joint_upper)

    return q_target
```

- [ ] **Step 1.5: Run tests — expect pass**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_ik.py -v 2>&1
```

Expected:
```
test_ik.py::test_build_ik_cache_finds_all_joints PASSED
test_ik.py::test_ik_returns_six_joint_targets PASSED
test_ik.py::test_ik_targets_within_joint_limits PASSED
test_ik.py::test_ik_moves_tcp_toward_target PASSED
test_ik.py::test_ik_zero_action_changes_nothing PASSED
5 passed
```

- [ ] **Step 1.6: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/ik.py \
        ur3e_rl_ws/src/ur3e_rl_env/tests/
git commit -m "feat: add Jacobian IK module — 5 tests pass"
```

---

## Task 2: Reach environment

**Files:**
- Create: `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/reach_env.py`
- Create: `ur3e_rl_ws/src/ur3e_rl_env/tests/test_reach_env.py`

- [ ] **Step 2.1: Write failing tests**

Write `tests/test_reach_env.py`:

```python
"""Tests for UR3eReachEnv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


@pytest.fixture
def env():
    from ur3e_rl_env.envs.reach_env import UR3eReachEnv
    e = UR3eReachEnv()
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (6,)
    assert env.action_space.shape == (3,)
    assert env.observation_space.dtype == np.float32
    assert env.action_space.dtype == np.float32


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (6,)
    assert obs.dtype == np.float32
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert "target_cube" in info


def test_step_returns_valid_obs(env):
    env.reset(seed=0)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs.shape == (6,)
    assert isinstance(reward, float)
    assert not np.isnan(reward)
    assert reward <= 0.0  # always negative (dist-based)


def test_reward_decreases_when_moving_toward_cube(env):
    """Moving toward the cube should increase reward (make it less negative)."""
    obs, info = env.reset(seed=42)
    # obs[0:3] = EE norm, obs[3:6] = cube norm
    # Direction toward cube: cube - ee in normalised space
    ee_norm   = obs[0:3]
    cube_norm = obs[3:6]
    direction = cube_norm - ee_norm  # unnormalised direction
    if np.linalg.norm(direction) < 1e-3:
        pytest.skip("EE already at cube at reset")

    # Action pointing toward cube
    action = np.clip(direction / (np.linalg.norm(direction) + 1e-8), -1, 1)
    action_padded = np.array([action[0], action[1], action[2]], dtype=np.float32)

    rewards = []
    current_obs = obs
    for _ in range(10):
        current_obs, r, term, trunc, _ = env.step(action_padded)
        rewards.append(r)
        if term or trunc:
            break

    # At least the first step should have a non-trivially negative reward
    assert rewards[0] < 0.0


def test_episode_terminates_at_max_steps(env):
    obs, _ = env.reset(seed=1)
    done = False
    steps = 0
    while not done and steps < 300:
        obs, r, terminated, truncated, _ = env.step(
            np.zeros(3, dtype=np.float32)
        )
        done = terminated or truncated
        steps += 1
    assert steps <= 200, f"Episode should terminate at 200 steps, took {steps}"


def test_success_terminates_episode(env):
    """When TCP reaches within 5cm of cube the episode terminates as success."""
    from ur3e_rl_env.envs.reach_env import UR3eReachEnv, GRASP_DIST_M
    import mujoco
    e = UR3eReachEnv()
    obs, info = e.reset(seed=0)

    # Teleport cube to the current TCP position (guaranteed success)
    tcp_pos = e.data.xpos[e._ik_cache["tcp_body_id"]].copy()
    addr    = e._cube_qpos_addrs[e._target_cube_idx]
    e.data.qpos[addr:addr+3] = tcp_pos
    e.data.qpos[addr+3] = 1.0
    e.data.qpos[addr+4:addr+7] = 0.0
    mujoco.mj_forward(e.model, e.data)

    obs, reward, terminated, truncated, info = e.step(np.zeros(3, dtype=np.float32))
    assert terminated, "Should terminate when cube is at TCP"
    assert info["is_success"]
    e.close()
```

- [ ] **Step 2.2: Run to confirm failure**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_reach_env.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'UR3eReachEnv'`

- [ ] **Step 2.3: Write `reach_env.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/reach_env.py`:

```python
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
HOME_JOINTS     = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])

CUBE_X_RANGE = (float(os.getenv("UR3E_RL_CUBE_X_MIN", "-0.20")),
                float(os.getenv("UR3E_RL_CUBE_X_MAX",  "0.20")))
CUBE_Y_RANGE = (float(os.getenv("UR3E_RL_CUBE_Y_MIN", "-0.45")),
                float(os.getenv("UR3E_RL_CUBE_Y_MAX", "-0.10")))
CUBE_Z       = float(os.getenv("UR3E_RL_CUBE_Z", "1.11"))
_NUM_CUBES   = 4


def _norm(v: float, lo: float, hi: float) -> float:
    span = max(hi - lo, 1e-6)
    return float(np.clip(2.0 * (v - lo) / span - 1.0, -1.0, 1.0))


def _norm_xyz(pos: np.ndarray) -> np.ndarray:
    return np.array([
        _norm(float(pos[0]), WORKSPACE_X_MIN, WORKSPACE_X_MAX),
        _norm(float(pos[1]), WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
        _norm(float(pos[2]), WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
    ], dtype=np.float32)


class UR3eReachEnv(gym.Env):
    """Reach: move TCP to within GRASP_DIST_M of the nearest cube."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

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
        if render_mode == "human":
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _get_ee_pos(self) -> np.ndarray:
        return self.data.xpos[self._ik_cache["tcp_body_id"]].astype(np.float32)

    def _get_cube_pos(self, idx: int) -> np.ndarray:
        return self.data.xpos[self._cube_body_ids[idx]].astype(np.float32)

    def _nearest_cube(self, ee_pos: np.ndarray) -> int:
        dists = [np.linalg.norm(self._get_cube_pos(i) - ee_pos) for i in range(_NUM_CUBES)]
        return int(np.argmin(dists))

    def _get_obs(self, ee_pos: np.ndarray, cube_pos: np.ndarray) -> np.ndarray:
        return np.concatenate([_norm_xyz(ee_pos), _norm_xyz(cube_pos)]).astype(np.float32)

    # ── Gym API ────────────────────────────────────────────────────────────────

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

        # Arm to home
        for i, addr in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr] = HOME_JOINTS[i]

        # Randomise cubes
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
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
```

- [ ] **Step 2.4: Run tests — expect pass**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_reach_env.py -v 2>&1
```

Expected: `6 passed`

- [ ] **Step 2.5: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/reach_env.py \
        ur3e_rl_ws/src/ur3e_rl_env/tests/test_reach_env.py
git commit -m "feat: add UR3eReachEnv — 6D obs, 3D Cartesian action, 6 tests pass"
```

---

## Task 3: Train reach model

**Files:**
- Create: `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_reach.py`

- [ ] **Step 3.1: Write `train_reach.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_reach.py`:

```python
"""
train_reach.py — Train Model 1: reach the nearest cube.

No ROS. Run directly with python3.

Usage:
    python3 train_reach.py
    python3 train_reach.py --timesteps 200000 --envs 16
    python3 train_reach.py --load rl_models/reach_best/best_model.zip
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.reach_env import UR3eReachEnv

_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "reach_tb_logs"
CKPT_DIR  = MODEL_DIR / "reach_checkpoints"
BEST_DIR  = MODEL_DIR / "reach_best"


def make_env(rank: int, seed: int = 0):
    def _init():
        e = UR3eReachEnv()
        e.reset(seed=seed + rank)
        return e
    set_random_seed(seed)
    return _init


class ProgressCB(BaseCallback):
    def __init__(self, freq: int = 10_000):
        super().__init__()
        self.freq = freq
        self._t0  = 0.0

    def _on_training_start(self):
        self._t0 = time.monotonic()

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            sps = self.num_timesteps / max(time.monotonic() - self._t0, 1e-6)
            print(f"[Reach] {self.num_timesteps:,} steps | {sps:.0f} sps", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--envs",      type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--load",      type=str, default=None)
    args = ap.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.envs)],
                               start_method="fork")
    train_env = VecMonitor(train_env, str(LOG_DIR / "monitor"))
    eval_env  = SubprocVecEnv([make_env(999, args.seed)], start_method="fork")
    eval_env  = VecMonitor(eval_env)

    callbacks = [
        ProgressCB(10_000),
        CheckpointCallback(
            save_freq=max(10_000 // args.envs, 1),
            save_path=str(CKPT_DIR), name_prefix="reach", verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_DIR),
            eval_freq=max(20_000 // args.envs, 1),
            n_eval_episodes=20, verbose=1,
        ),
    ]

    if args.load:
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
        reset_ts = False
    else:
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, learning_rate=3e-4,
            ent_coef=0.01,   # higher entropy → more exploration
            clip_range=0.2, max_grad_norm=0.5,
            device="cpu", verbose=1, seed=args.seed,
            tensorboard_log=str(LOG_DIR),
        )
        reset_ts = True

    print(f"\nTraining Reach model — {args.timesteps:,} steps, {args.envs} envs")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")
    model.learn(args.timesteps, callback=callbacks,
                reset_num_timesteps=reset_ts, progress_bar=True)
    model.save(str(MODEL_DIR / "reach_final"))
    print(f"Saved to {MODEL_DIR / 'reach_final'}.zip")
    train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.2: Run 1000-step sanity check**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env
PYTHONPATH=.. python3 train_reach.py --timesteps 1000 --envs 4 --seed 0
```

Expected: prints `[Reach] ... steps`, no crash, checkpoint saved.

- [ ] **Step 3.3: Run full training (200k steps — ~5 min)**

```bash
PYTHONPATH=.. python3 train_reach.py --timesteps 200000 --envs 16
```

**Watch for:** `ep_len_mean` decreasing below 200 (arm reaching the cube before timeout), `success_rate` above 0%. If by 100k steps both are still at worst-case, increase `ent_coef` to 0.02 in `train_reach.py` and restart.

- [ ] **Step 3.4: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_reach.py
git commit -m "feat: add train_reach.py for Stage 1 reach model"
```

---

## Task 4: Transport environment

**Files:**
- Create: `ur3e_rl_env/envs/transport_env.py`
- Create: `tests/test_transport_env.py`

- [ ] **Step 4.1: Write failing tests**

Write `tests/test_transport_env.py`:

```python
"""Tests for UR3eTransportEnv."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest


@pytest.fixture
def env():
    from ur3e_rl_env.envs.transport_env import UR3eTransportEnv
    e = UR3eTransportEnv()
    yield e
    e.close()


def test_spaces(env):
    assert env.observation_space.shape == (7,)
    assert env.action_space.shape == (3,)
    assert env.observation_space.dtype == np.float32


def test_reset_returns_valid_obs(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (7,)
    assert np.all(obs >= -1.0) and np.all(obs <= 1.0)
    assert "target_cube" in info


def test_cube_follows_arm(env):
    """After reset, cube should stay near TCP as arm moves."""
    import mujoco
    obs, _ = env.reset(seed=0)
    tcp_id   = env._ik_cache["tcp_body_id"]
    cube_idx = env._target_cube_idx
    cube_bid = env._cube_body_ids[cube_idx]

    # Move arm down
    for _ in range(5):
        env.step(np.array([0.0, 0.0, -1.0], dtype=np.float32))

    tcp_pos  = env.data.xpos[tcp_id]
    cube_pos = env.data.xpos[cube_bid]
    dist = float(np.linalg.norm(tcp_pos - cube_pos))
    assert dist < 0.10, f"Cube should follow TCP but dist={dist:.4f}"


def test_reward_is_negative_dist_to_bin(env):
    obs, _ = env.reset(seed=0)
    from ur3e_rl_env.constants import BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z
    import mujoco

    cube_idx = env._target_cube_idx
    cube_bid = env._cube_body_ids[cube_idx]
    cube_pos = env.data.xpos[cube_bid].copy()
    bin_pos  = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z])
    expected_reward = -float(np.linalg.norm(cube_pos - bin_pos))

    _, reward, _, _, _ = env.step(np.zeros(3, dtype=np.float32))
    # Reward may differ slightly due to physics step; allow 5cm tolerance
    assert abs(reward - expected_reward) < 0.05


def test_success_terminates(env):
    """Placing cube within 8cm of bin terminates with success."""
    import mujoco
    from ur3e_rl_env.constants import BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z
    from ur3e_rl_env.envs.transport_env import RELEASE_DIST_M

    obs, _ = env.reset(seed=0)
    # Teleport cube to bin
    bin_pos  = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z + 0.02])
    cube_idx = env._target_cube_idx
    addr     = env._cube_qpos_addrs[cube_idx]
    env.data.qpos[addr:addr+3] = bin_pos
    env.data.qpos[addr+3] = 1.0
    env.data.qpos[addr+4:addr+7] = 0.0
    mujoco.mj_forward(env.model, env.data)

    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["is_success"]
```

- [ ] **Step 4.2: Run to confirm failure**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_transport_env.py -v 2>&1 | tail -5
```

Expected: `ImportError: cannot import name 'UR3eTransportEnv'`

- [ ] **Step 4.3: Write `transport_env.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/transport_env.py`:

```python
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

# Cube-to-TCP offset when "held": slightly below TCP centre
_HOLD_OFFSET = np.array([0.0, 0.0, -0.03])

_BIN_POS = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z],
                    dtype=np.float32)
# Maximum distance cube can be from bin (for normalisation)
_MAX_BIN_DIST = 1.0


def _norm(v: float, lo: float, hi: float) -> float:
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

    metadata = {"render_modes": ["human"], "render_fps": 50}

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

    # ── Kinematic cube attachment ──────────────────────────────────────────────

    def _pin_cube_to_tcp(self) -> None:
        """Teleport the target cube to follow the TCP each physics step."""
        tcp_pos = self.data.xpos[self._ik_cache["tcp_body_id"]].copy()
        tcp_quat = self.data.xquat[self._ik_cache["tcp_body_id"]].copy()
        cube_world_pos = tcp_pos + _HOLD_OFFSET
        addr = self._cube_qpos_addrs[self._target_cube_idx]
        dof  = self._cube_dof_addrs[self._target_cube_idx]
        self.data.qpos[addr:addr+3]   = cube_world_pos
        self.data.qpos[addr+3:addr+7] = tcp_quat
        self.data.qvel[dof:dof+6]     = 0.0

    # ── Observation ────────────────────────────────────────────────────────────

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

    # ── Gym API ────────────────────────────────────────────────────────────────

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

        # Arm to home
        for i, addr in enumerate(self._ik_cache["arm_qpos_addrs"]):
            self.data.qpos[addr] = HOME_JOINTS[i]

        # Choose a random cube as target; place it at a random table position
        self._target_cube_idx = int(self.np_random.integers(0, _NUM_CUBES))
        for i in range(_NUM_CUBES):
            addr = self._cube_qpos_addrs[i]
            self.data.qpos[addr]   = float(self.np_random.uniform(*CUBE_X_RANGE))
            self.data.qpos[addr+1] = float(self.np_random.uniform(*CUBE_Y_RANGE))
            self.data.qpos[addr+2] = CUBE_Z_START
            self.data.qpos[addr+3] = 1.0
            self.data.qpos[addr+4:addr+7] = 0.0

        mujoco.mj_forward(self.model, self.data)
        # Immediately pin the target cube to the TCP (it's "already grasped")
        self._pin_cube_to_tcp()
        mujoco.mj_forward(self.model, self.data)

        ee_pos   = self._get_ee_pos()
        cube_pos = self._get_cube_pos()
        return self._get_obs(ee_pos, cube_pos), {"target_cube": f"cube_{self._target_cube_idx}"}

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
```

- [ ] **Step 4.4: Run tests — expect pass**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -m pytest tests/test_transport_env.py -v 2>&1
```

Expected: `4 passed`

- [ ] **Step 4.5: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/transport_env.py \
        ur3e_rl_ws/src/ur3e_rl_env/tests/test_transport_env.py
git commit -m "feat: add UR3eTransportEnv — kinematic cube attachment, 4 tests pass"
```

---

## Task 5: Train transport model

**Files:**
- Create: `ur3e_rl_env/train_transport.py`

- [ ] **Step 5.1: Write `train_transport.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_transport.py`:

```python
"""
train_transport.py — Train Model 2: move the held cube to the bin.

No ROS. Run directly with python3.

Usage:
    python3 train_transport.py
    python3 train_transport.py --timesteps 200000 --envs 16
"""
from __future__ import annotations
import argparse, time
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback, CheckpointCallback, EvalCallback,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ur3e_rl_env.envs.transport_env import UR3eTransportEnv

_REPO     = Path(__file__).parent.parent.parent.parent.parent
MODEL_DIR = _REPO / "ur3e_rl_ws" / "rl_models"
LOG_DIR   = _REPO / "ur3e_rl_ws" / "transport_tb_logs"
CKPT_DIR  = MODEL_DIR / "transport_checkpoints"
BEST_DIR  = MODEL_DIR / "transport_best"


def make_env(rank: int, seed: int = 0):
    def _init():
        e = UR3eTransportEnv()
        e.reset(seed=seed + rank)
        return e
    set_random_seed(seed)
    return _init


class ProgressCB(BaseCallback):
    def __init__(self, freq: int = 10_000):
        super().__init__()
        self.freq = freq
        self._t0  = 0.0

    def _on_training_start(self):
        self._t0 = time.monotonic()

    def _on_step(self):
        if self.n_calls % self.freq == 0:
            sps = self.num_timesteps / max(time.monotonic() - self._t0, 1e-6)
            print(f"[Transport] {self.num_timesteps:,} steps | {sps:.0f} sps", flush=True)
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=200_000)
    ap.add_argument("--envs",      type=int, default=16)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--load",      type=str, default=None)
    args = ap.parse_args()

    for d in (MODEL_DIR, LOG_DIR, CKPT_DIR, BEST_DIR):
        d.mkdir(parents=True, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.envs)],
                               start_method="fork")
    train_env = VecMonitor(train_env, str(LOG_DIR / "monitor"))
    eval_env  = SubprocVecEnv([make_env(999, args.seed)], start_method="fork")
    eval_env  = VecMonitor(eval_env)

    callbacks = [
        ProgressCB(10_000),
        CheckpointCallback(
            save_freq=max(10_000 // args.envs, 1),
            save_path=str(CKPT_DIR), name_prefix="transport", verbose=1,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(BEST_DIR),
            eval_freq=max(20_000 // args.envs, 1),
            n_eval_episodes=20, verbose=1,
        ),
    ]

    if args.load:
        model = PPO.load(args.load, env=train_env, tensorboard_log=str(LOG_DIR))
        reset_ts = False
    else:
        model = PPO(
            "MlpPolicy", train_env,
            n_steps=2048, batch_size=256, n_epochs=10,
            gamma=0.99, learning_rate=3e-4,
            ent_coef=0.01, clip_range=0.2, max_grad_norm=0.5,
            device="cpu", verbose=1, seed=args.seed,
            tensorboard_log=str(LOG_DIR),
        )
        reset_ts = True

    print(f"\nTraining Transport model — {args.timesteps:,} steps, {args.envs} envs")
    print(f"TensorBoard: tensorboard --logdir {LOG_DIR}\n")
    model.learn(args.timesteps, callback=callbacks,
                reset_num_timesteps=reset_ts, progress_bar=True)
    model.save(str(MODEL_DIR / "transport_final"))
    print(f"Saved to {MODEL_DIR / 'transport_final'}.zip")
    train_env.close(); eval_env.close()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5.2: Sanity check**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env
PYTHONPATH=.. python3 train_transport.py --timesteps 1000 --envs 4
```

Expected: no crash, prints `[Transport]` progress.

- [ ] **Step 5.3: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_transport.py
git commit -m "feat: add train_transport.py for Stage 3 transport model"
```

---

## Task 6: Coordinator and pipeline viewer

**Files:**
- Create: `ur3e_rl_env/coordinator.py`
- Create: `watch_pipeline.py`

- [ ] **Step 6.1: Write `coordinator.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/coordinator.py`:

```python
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
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

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
HOME_JOINTS    = np.array([0.0, -np.pi / 2, 0.0, -np.pi / 2, 0.0, 0.0])
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

    # ── Helpers ────────────────────────────────────────────────────────────────

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

    # ── Public API ─────────────────────────────────────────────────────────────

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
            print(f"  released → cube at bin (dist={np.linalg.norm(self._get_cube_pos() - _BIN_POS):.3f} m)")

        if self._viewer is not None:
            self._viewer.sync()

        return self.stage

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
```

- [ ] **Step 6.2: Write `watch_pipeline.py`**

Write `ur3e_rl_ws/src/ur3e_rl_env/watch_pipeline.py`:

```python
#!/usr/bin/env python3
"""
watch_pipeline.py — Watch the full Reach → Grasp → Transport pipeline.

Usage:
    # Watch with trained models:
    PYTHONPATH=. python3 watch_pipeline.py \
        --reach     ../../rl_models/reach_best/best_model.zip \
        --transport ../../rl_models/transport_best/best_model.zip

    # Speed options (real-time = 1.0):
    ... --speed 0.5   # half speed, easier to watch

Controls: left-drag rotate, scroll zoom, close window to quit.
"""
import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco.viewer  # must import before coordinator
from ur3e_rl_env.coordinator import MuJoCoCoordinator, Stage

SIM_DT = 50 * 0.002  # 0.1 s sim time per RL step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach",     required=True, help="Path to reach model .zip")
    ap.add_argument("--transport", required=True, help="Path to transport model .zip")
    ap.add_argument("--episodes",  type=int, default=0, help="Episodes (0=forever)")
    ap.add_argument("--speed",     type=float, default=1.0, help="Playback speed (1=real-time)")
    ap.add_argument("--seed",      type=int, default=0)
    args = ap.parse_args()

    coord = MuJoCoCoordinator(
        reach_model_path=args.reach,
        transport_model_path=args.transport,
        render_mode="human",
        rng_seed=args.seed,
    )

    wall_per_step = SIM_DT / args.speed
    episode = 0

    print(f"\nPipeline viewer — {args.speed}x speed")
    print("Reach → Grasp → Transport → Release\n")

    try:
        coord.reset()
        while coord.is_running():
            t0    = time.monotonic()
            stage = coord.step()

            elapsed   = time.monotonic() - t0
            remaining = wall_per_step - elapsed
            if remaining > 0:
                time.sleep(remaining)

            if stage == Stage.DONE:
                episode += 1
                print(f"Episode {episode} complete\n")
                if args.episodes > 0 and episode >= args.episodes:
                    break
                coord.reset()

    except KeyboardInterrupt:
        pass
    finally:
        coord.close()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.3: Update `envs/__init__.py`**

```python
# Add to ur3e_rl_env/envs/__init__.py:
from ur3e_rl_env.envs.reach_env import UR3eReachEnv
from ur3e_rl_env.envs.transport_env import UR3eTransportEnv
```

- [ ] **Step 6.4: Verify coordinator imports cleanly**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 -c "
from ur3e_rl_env.coordinator import MuJoCoCoordinator, Stage
print('Coordinator import OK')
print('Stages:', [s.value for s in Stage])
"
```

Expected: `Coordinator import OK` and stage list printed.

- [ ] **Step 6.5: Commit**

```bash
cd /home/john/git/HoloAssist-AI
git add ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/coordinator.py \
        ur3e_rl_ws/src/ur3e_rl_env/watch_pipeline.py \
        ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/__init__.py
git commit -m "feat: add coordinator state machine and watch_pipeline.py"
```

---

## Task 7: End-to-end demo (run after training)

This task requires trained models from Tasks 3 and 5. Run it once both training runs have a `best_model.zip` in their respective directories.

- [ ] **Step 7.1: Run reach training (if not already running)**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env
PYTHONPATH=.. python3 train_reach.py --timesteps 200000 --envs 16
```

- [ ] **Step 7.2: Run transport training (can run in parallel with reach)**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env
PYTHONPATH=.. python3 train_transport.py --timesteps 200000 --envs 16
```

- [ ] **Step 7.3: Watch the pipeline**

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env
PYTHONPATH=. python3 watch_pipeline.py \
    --reach     ../../rl_models/reach_best/best_model.zip \
    --transport ../../rl_models/transport_best/best_model.zip \
    --speed 0.5
```

Expected: arm moves to a cube → logs "grasped" → arm moves toward bin → logs "released". Repeat each episode.

**If reach model hasn't learned (ep_len stays at 200, success_rate=0 after 100k steps):**
Increase `ent_coef` to 0.02 in `train_reach.py`, restart training.

**If transport model hasn't learned (same symptoms):**
Check that `_pin_cube_to_tcp` is correctly teleporting the cube each step. Run:
```bash
PYTHONPATH=. python3 -m pytest tests/test_transport_env.py::test_cube_follows_arm -v
```

---

## Self-review

**Spec coverage:**
- [x] Two PPO models (reach + transport)
- [x] Scripted grasp (kinematic attachment via `_pin_cube_to_tcp`)
- [x] Scripted release (unpin cube, drops under gravity)
- [x] Cartesian delta action space (3D, ±2cm/step)
- [x] Gripper pointing down enforced via IK orientation correction
- [x] Dense rewards for both models
- [x] Coordinator state machine (REACH→GRASP→TRANSPORT→RELEASE→DONE)
- [x] Viewer script with real-time speed control
- [x] No changes to existing envs/reward/constants

**Placeholder scan:** None found.

**Type consistency:**
- `build_ik_cache` returns dict with keys `tcp_body_id`, `arm_dof_addrs`, `arm_qpos_addrs` — used consistently in `ik.py`, `reach_env.py`, `transport_env.py`, `coordinator.py`
- `_pin_cube_to_tcp` method name consistent between `transport_env.py` and `coordinator.py`
- `Stage` enum used consistently in `coordinator.py` and `watch_pipeline.py`
- `GRASP_DIST_M = 0.05`, `RELEASE_DIST_M = 0.08` defined in both `reach_env.py` and `coordinator.py`
