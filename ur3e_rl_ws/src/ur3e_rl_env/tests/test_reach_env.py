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
    ee_norm   = obs[0:3]
    cube_norm = obs[3:6]
    direction = cube_norm - ee_norm
    if np.linalg.norm(direction) < 1e-3:
        pytest.skip("EE already at cube at reset")

    action = np.clip(direction / (np.linalg.norm(direction) + 1e-8), -1, 1)
    action_padded = np.array([action[0], action[1], action[2]], dtype=np.float32)

    rewards = []
    for _ in range(10):
        _, r, term, trunc, _ = env.step(action_padded)
        rewards.append(r)
        if term or trunc:
            break

    # Reward = -dist, so improvement means reward increases (less negative)
    # After 10 steps toward cube, final reward should be better than initial
    if len(rewards) >= 2:
        assert rewards[-1] > rewards[0], (
            f"Moving toward cube should improve reward: "
            f"start={rewards[0]:.4f} end={rewards[-1]:.4f}"
        )
    else:
        assert rewards[0] < 0.0  # episode ended early (success)


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

    # Zero cube velocity so it doesn't fall during the physics steps
    cube_dof = e._cube_dof_addrs[e._target_cube_idx]
    e.data.qvel[cube_dof:cube_dof+6] = 0.0

    obs, reward, terminated, truncated, info = e.step(np.zeros(3, dtype=np.float32))
    assert terminated, "Should terminate when cube is at TCP"
    assert info["is_success"]
    e.close()
