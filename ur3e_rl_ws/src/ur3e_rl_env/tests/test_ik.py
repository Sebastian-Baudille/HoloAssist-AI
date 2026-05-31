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
    # Orientation correction may cause slight drift; allow 5cm tolerance
    assert np.linalg.norm(pos_after - pos_before) < 0.05
