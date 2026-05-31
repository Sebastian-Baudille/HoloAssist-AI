"""
ik.py — Jacobian IK: Cartesian delta (dx,dy,dz) → joint angle targets.

Converts a 3D world-frame TCP displacement into joint targets using
damped-least-squares Jacobian IK (translational only — no orientation
constraint, so the arm uses its full joint range to reach any pose).

This runs at every RL step inside reach_env and transport_env.
"""
from __future__ import annotations
import mujoco
import numpy as np
from ur3e_rl_env.constants import JOINT_DELTA_ACTION_SCALE_RAD

_ARM_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
_TCP_BODY_NAME = "gripper_tcp"

ACTION_SCALE_M = 0.02  # metres per action unit (±1 → ±2 cm)
IK_DAMPING     = 0.05  # regularisation — increase if joints oscillate


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

    Requires `mj_forward(model, data)` to have been called since the last
    state change (satisfied automatically by `mj_step`).

    Returns joint targets to set on data.ctrl[:6].
    """
    tcp_body_id   = ik_cache["tcp_body_id"]
    arm_dof_addrs = ik_cache["arm_dof_addrs"]
    arm_qpos_addrs = ik_cache["arm_qpos_addrs"]

    # ── Jacobian ───────────────────────────────────────────────────────────────
    jacp = np.zeros((3, model.nv))  # translational  (3 × nv)
    jacr = np.zeros((3, model.nv))  # rotational     (3 × nv)
    mujoco.mj_jacBody(model, data, jacp, jacr, tcp_body_id)

    # Extract arm columns only → (3, 6) translational Jacobian
    J = jacp[:, arm_dof_addrs]  # (3, 6)

    # ── Desired velocity ───────────────────────────────────────────────────────
    v = np.asarray(delta_xyz, dtype=np.float64) * ACTION_SCALE_M  # (3,)

    # ── Damped least squares ───────────────────────────────────────────────────
    # dq = J^T (J J^T + λ²I)^{-1} v
    JJT = J @ J.T + IK_DAMPING ** 2 * np.eye(3)
    dq  = J.T @ np.linalg.solve(JJT, v)  # (6,) rad/step

    # ── Apply delta, clip, clamp ───────────────────────────────────────────────
    q_curr  = np.array([data.qpos[a] for a in arm_qpos_addrs], dtype=np.float64)
    dq_safe = np.clip(dq, -JOINT_DELTA_ACTION_SCALE_RAD, JOINT_DELTA_ACTION_SCALE_RAD)
    q_target = np.clip(q_curr + dq_safe, joint_lower, joint_upper)

    return q_target
