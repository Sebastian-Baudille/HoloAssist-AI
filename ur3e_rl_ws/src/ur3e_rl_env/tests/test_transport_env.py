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
    # Teleport cube to bin (with zero velocity to prevent drift)
    bin_pos  = np.array([BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z + 0.02])
    cube_idx = env._target_cube_idx
    addr     = env._cube_qpos_addrs[cube_idx]
    dof      = env._cube_dof_addrs[cube_idx]
    env.data.qpos[addr:addr+3] = bin_pos
    env.data.qpos[addr+3] = 1.0
    env.data.qpos[addr+4:addr+7] = 0.0
    env.data.qvel[dof:dof+6] = 0.0
    mujoco.mj_forward(env.model, env.data)

    _, _, terminated, _, info = env.step(np.zeros(3, dtype=np.float32))
    assert terminated
    assert info["is_success"]
