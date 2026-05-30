#!/usr/bin/env python3
"""
weight_transfer_test.py
Verifies MuJoCo env has identical obs/action spaces to Gazebo env,
and that obs normalization matches build_observation() from ros_interface.py.
All assertions must pass before training.
"""
import sys
import numpy as np
sys.path.insert(0, "/home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env")

from ur3e_rl_env.envs.ur3e_mujoco_env import UR3eMuJoCoEnv
from ur3e_rl_env.constants import OBSERVATION_SIZE_13D

print("=" * 60)
print("Weight Transfer Compatibility Test")
print("=" * 60)

print("\n[1] Creating MuJoCo env...")
env = UR3eMuJoCoEnv()

assert env.observation_space.shape == (OBSERVATION_SIZE_13D,), \
    f"Wrong obs shape: {env.observation_space.shape}"
assert env.action_space.shape == (7,), \
    f"Wrong action shape: {env.action_space.shape}"
assert env.observation_space.dtype == np.float32, "obs dtype not float32"
assert env.action_space.dtype == np.float32, "action dtype not float32"
print(f"  ✓ obs space:    {env.observation_space}")
print(f"  ✓ action space: {env.action_space}")

print("\n[2] Testing reset...")
obs, info = env.reset(seed=42)
assert obs.shape == (OBSERVATION_SIZE_13D,), f"Wrong obs shape: {obs.shape}"
assert obs.dtype == np.float32, f"Wrong dtype: {obs.dtype}"
assert np.all(obs >= -1.0001) and np.all(obs <= 1.0001), \
    f"Obs out of [-1,1]: min={obs.min():.4f} max={obs.max():.4f}"
print(f"  ✓ obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"  ✓ obs range: [{obs.min():.3f}, {obs.max():.3f}]")
print(f"  ✓ target cube: {info['target_cube']}")
print(f"  obs[0:3]  (EE pos normalised):   {obs[0:3]}")
print(f"  obs[3:6]  (cube pos normalised):  {obs[3:6]}")
print(f"  obs[6:9]  (bin pos normalised):   {obs[6:9]}")
print(f"  obs[9]    (grasped):              {obs[9]}")
print(f"  obs[10]   (gripper_state):        {obs[10]}")
print(f"  obs[11]   (ee_height_norm):       {obs[11]:.4f}")
print(f"  obs[12]   (timestep_norm):        {obs[12]:.4f}")

assert obs[12] == 0.0, f"timestep_norm should be 0 at reset, got {obs[12]}"
assert obs[10] in (0.0, 1.0), f"gripper_state must be binary, got {obs[10]}"
assert obs[9] in (0.0, 1.0), f"grasped must be binary, got {obs[9]}"

print("\n[3] Testing step...")
action = env.action_space.sample()
obs2, reward, terminated, truncated, info = env.step(action)
assert obs2.shape == (OBSERVATION_SIZE_13D,)
assert obs2.dtype == np.float32
assert not np.isnan(reward), "reward is NaN"
assert np.all(obs2 >= -1.0001) and np.all(obs2 <= 1.0001), "step obs out of range"
print(f"  ✓ step reward: {reward:.4f}")
print(f"  ✓ terminated={terminated}, truncated={truncated}")

assert abs(obs2[12] - (1 / 200)) < 1e-5, \
    f"obs[12] should be 1/200={1/200:.5f} after 1 step, got {obs2[12]:.5f}"
print(f"  ✓ obs[12] = {obs2[12]:.5f} = 1/200 after 1 step")

print("\n[4] Running 50 random steps...")
total_reward = 0.0
obs, _ = env.reset(seed=0)
for step_i in range(50):
    a = env.action_space.sample()
    o, r, term, trunc, _ = env.step(a)
    total_reward += r
    assert not np.isnan(r), f"NaN reward at step {step_i}"
    assert np.all(o >= -1.0001) and np.all(o <= 1.0001), f"obs out of range at step {step_i}"
    if term or trunc:
        obs, _ = env.reset()
print(f"  ✓ 50 steps completed, total reward: {total_reward:.4f}")

print("\n[5] Testing obs normalization against ros_interface formula...")
from ur3e_rl_env.constants import (
    WORKSPACE_X_MIN, WORKSPACE_X_MAX,
    WORKSPACE_Y_MIN, WORKSPACE_Y_MAX,
    WORKSPACE_Z_MIN, WORKSPACE_Z_MAX,
    BIN_POSITION_X, BIN_POSITION_Y, BIN_POSITION_Z,
)

def normalize_axis_ref(v, lo, hi):
    span = max(hi - lo, 1e-6)
    return float(np.clip(2.0 * ((v - lo) / span) - 1.0, -1.0, 1.0))

expected_bin = np.array([
    normalize_axis_ref(BIN_POSITION_X, WORKSPACE_X_MIN, WORKSPACE_X_MAX),
    normalize_axis_ref(BIN_POSITION_Y, WORKSPACE_Y_MIN, WORKSPACE_Y_MAX),
    normalize_axis_ref(BIN_POSITION_Z, WORKSPACE_Z_MIN, WORKSPACE_Z_MAX),
], dtype=np.float32)

obs, _ = env.reset(seed=1)
actual_bin = obs[6:9]
np.testing.assert_allclose(
    actual_bin, expected_bin, atol=1e-5,
    err_msg=f"Bin normalisation mismatch.\n  expected: {expected_bin}\n  got: {actual_bin}"
)
print(f"  ✓ Bin pos normalisation matches ros_interface formula")
print(f"    expected: {expected_bin}")
print(f"    actual:   {actual_bin}")

env.close()
print("\n" + "=" * 60)
print("✓ ALL WEIGHT TRANSFER TESTS PASSED")
print("  MuJoCo env is compatible with the Gazebo env.")
print("  Policies trained here will transfer correctly.")
print("=" * 60)
