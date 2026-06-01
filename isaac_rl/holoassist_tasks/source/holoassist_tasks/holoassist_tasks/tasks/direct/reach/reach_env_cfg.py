# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the HoloAssist UR3e reach task.

Ports the constants from the legacy ROS stack into Isaac Lab `@configclass`
fields. See:
  - Legacy source: ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/constants.py
  - Port plan:     isaac_rl/ISAAC_SIM_PLAN.md § Phase 4b (Mapping #1)
  - Design decisions: project-phase4-design-decisions memory entry

Sections:
  1. Env timing + spaces
  2. Simulation + scene
  3. Robot mounting (the URDF puts the base at world z=1.0)
  4. Action scaling (true joint-delta, per Q3)
  5. Home pose (UR3e parked)
  6. Target randomisation (lifted from legacy world-frame ranges)
  7. Success / failure thresholds
  8. Reward scales (kept at legacy values; tune later)
  9. Scene element toggles (table / ground / target marker)

Joint position limits are NOT in this cfg — they're read from the imported USD
at runtime via `robot.data.joint_pos_limits`. Single source of truth (USD -
PhysX - Isaac Lab API), no risk of cfg drifting from the URDF.

The robot's `ArticulationCfg` (Mapping #5) is defined here too as the
module-level constant `UR_ONROBOT_CFG`, then referenced from the env cfg via
`robot_cfg`. Pattern mirrors Isaac Lab's `isaaclab_assets.robots.universal_robots`
module — the cfg holds the USD path + actuator groups + default joint state,
the env cfg points at it.
"""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# ----------------------------------------------------------------------------
# Path to the USD asset produced by Phase 3. Resolved relative to this file so
# the cfg works regardless of cwd or who clones the repo.
#
# parents[7] ascends:
#   reach/ - direct/ - tasks/ - holoassist_tasks/ (inner module)
#   - holoassist_tasks/ (package root) - source/ - holoassist_tasks/ (project)
#   - isaac_rl/
# ----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_ISAAC_RL = _THIS_FILE.parents[7]
_USD_PATH = _ISAAC_RL / "assets" / "usd" / "ur_onrobot_prepared" / "ur_onrobot_prepared.usd"

# ----------------------------------------------------------------------------
# Robot ArticulationCfg — UR3e + RG2 (articulated, mimic-stripped). Mirrors the
# validated configuration from isaac_rl/scripts/robot_test_v0.py: 3 actuator
# groups (arm + gripper_driver + gripper_linkage), 13 joints, parked init pose.
#
# Per-joint init values for the 6 gripper-linkage joints are at +/- 0.78 rad
# (slightly inside the URDF's +/-0.7854 limit; Isaac Lab does strict
# containment so being exactly at the limit fails validation).
# ----------------------------------------------------------------------------
UR_ONROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=str(_USD_PATH).replace("\\", "/")),
    init_state=ArticulationCfg.InitialStateCfg(
        # Mount the robot on top of the table — base flange at world z=1.0.
        # The URDF's `<origin xyz="0 0 1"/>` baked into ur_onrobot_macro.xacro
        # got merged out during Isaac's "Merge Fixed Joints" import pass, so
        # the base now sits at whatever the InitialStateCfg.pos says (vs.
        # at the URDF-encoded z=1.0).
        pos=(0.0, 0.0, 1.0),
        joint_pos={
            # Arm — UR3e ZERO pose: all joints at 0 rad. The arm extends
            # straight up vertically from the base ("candle" pose). EE starts
            # at approximately z = robot_base_height + 0.55 m ≈ 1.55 m, well
            # above the target spawn height (z ≈ 1.11), giving the policy
            # plenty of vertical room to descend to targets.
            # Matches UR3e factory zero / "home" configuration as used by
            # ROS-Industrial UR drivers.
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": 0.0,
            "elbow_joint": 0.0,
            "wrist_1_joint": 0.0,
            "wrist_2_joint": 0.0,
            "wrist_3_joint": 0.0,
            # Gripper driver — closed (0 m on the prismatic)
            "finger_width": 0.0,
            # Gripper linkage — values for the closed configuration, computed
            # from the URDF mimic multipliers and padded inward from the limit.
            "finger_joint": 0.78,
            "left_inner_knuckle_joint": -0.78,
            "left_inner_finger_joint": 0.78,
            "right_outer_knuckle_joint": -0.78,
            "right_inner_knuckle_joint": -0.78,
            "right_inner_finger_joint": 0.78,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            stiffness=800.0,
            damping=40.0,
        ),
        "gripper_driver": ImplicitActuatorCfg(
            joint_names_expr=["finger_width"],
            stiffness=200.0,
            damping=10.0,
        ),
        "gripper_linkage": ImplicitActuatorCfg(
            joint_names_expr=[
                "finger_joint",
                "left_inner_knuckle_joint",
                "left_inner_finger_joint",
                "right_outer_knuckle_joint",
                "right_inner_knuckle_joint",
                "right_inner_finger_joint",
            ],
            stiffness=500.0,
            damping=25.0,
        ),
    },
)


@configclass
class HoloassistReachEnvCfg(DirectRLEnvCfg):
    """Configuration for the UR3e reach task.

    All values that were per-experiment env-vars in the legacy stack become
    typed class attributes here. Override via Hydra at the command line
    (e.g. ``--cfg-overrides episode_length_s=10.0``).
    """

    # ---------------------------------------------------------------- 1. env timing + spaces
    decimation = 4                              # 30 Hz policy at sim dt 1/120
    episode_length_s = 6.7                      # ~200 control steps = 6.7 s wall-clock
    action_space = 6                            # 6 arm joints; true delta in rad/step (gripper held closed)
    observation_space = 12                      # 6 joint_pos + 3 EE_pos + 3 target_pos (Q2 decision)
    state_space = 0                             # not using asymmetric actor-critic

    # ---------------------------------------------------------------- 2. simulation + scene
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)

    # Default num_envs intentionally low for smoke-testing. Override on the CLI
    # for full training: `--num_envs 4096`.
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ---------------------------------------------------------------- 3. robot mounting
    # Matches the URDF base offset: `<xacro:ur_onrobot ... ><origin xyz="0 0 1" .../>`
    # If the robot is ever re-mounted, update this once and the min-EE-height
    # termination auto-adjusts.
    robot_base_height_m: float = 1.0

    # The ArticulationCfg (defined at module level above as UR_ONROBOT_CFG).
    # `.replace(prim_path=...)` clones it with the env-namespaced spawn path
    # so each cloned env gets its own robot prim.
    #
    # Pattern `/World/envs/env_.*/Robot` (NOT `{ENV_REGEX_NS}/Robot`) — the
    # placeholder syntax only expands when the cfg is auto-discovered by
    # InteractiveSceneCfg; we instantiate Articulation() directly in
    # _setup_scene, so the explicit regex pattern is required.
    robot_cfg: ArticulationCfg = UR_ONROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # ---------------------------------------------------------------- 4. action scaling
    # True joint-delta semantics (Q3 decision):
    #   target = current + clip(action * action_scale_rad, +/-action_scale_rad)
    # Adjusted from legacy's 0.24 rad/step to 0.08 rad/step because Isaac runs
    # control at 30 Hz (decimation=4 x dt=1/120) vs legacy's 10 Hz. Keeping the
    # same per-step delta at 3x higher rate would yield 7.2 rad/s — well above
    # UR3e's ~3 rad/s hardware ceiling. 0.08 rad/step at 30 Hz = 2.4 rad/s,
    # matching legacy max speed (75% of hardware limit). Safer for sim-to-real.
    # See ur3e_rl_ws/.../constants.py JOINT_DELTA_ACTION_SCALE_RAD for the
    # legacy value.
    action_scale_rad: float = 0.08

    # ---------------------------------------------------------------- 5. home pose
    # UR3e zero pose (vertical "candle" — all joints at 0 rad, arm straight up).
    # EE starts at z ≈ 1.55 m (≈ 0.55 m above the robot base / table top),
    # well above the target spawn height (z ≈ 1.11). Policy descends to reach.
    # Documented here for reference; the actual init is set by UR_ONROBOT_CFG
    # above (which Articulation reads at reset time as default_joint_pos).
    # Order: shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
    home_joint_pos: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---------------------------------------------------------------- 6. target randomisation
    # Lifted from legacy CUBE_X/Y_RANGE (env vars UR3E_RL_CUBE_X_MIN/MAX, etc.).
    # World-frame; z fixed at 1.11 m (~11 cm above robot base, reachable when
    # the arm unfolds from the parked pose).
    target_pos_range_x: tuple[float, float] = (-0.20, 0.20)
    target_pos_range_y: tuple[float, float] = (-0.45, -0.10)
    target_pos_z: float = 1.11

    # ---------------------------------------------------------------- 7. success / failure
    # Success: EE within tolerance of target (matches legacy SUCCESS_DISTANCE_M).
    success_tolerance_m: float = 0.04

    # Failure: EE drops too far below the robot base (crash guard).
    # Effective min_ee_z = robot_base_height_m - min_ee_clearance_below_base_m
    # Default: 1.0 - 0.9 = 0.1 m  -  termination only fires if the EE drops
    # almost to the ground, which only happens in pathological "arm collapsed"
    # cases. Generous because:
    #   - PhysX already prevents EE from penetrating the table (rigid body collision)
    #   - Tight thresholds kill episodes before the policy can explore (Phase 4i
    #     smoke test had mean episode length ~10 with min_ee_z=0.5)
    #   - For sim-to-real, the real-robot SafetyChecker enforces its own height
    #     guard at deployment time (out of scope for the Isaac env)
    min_ee_clearance_below_base_m: float = 0.9

    # ---------------------------------------------------------------- 8. reward scales
    # Kept at legacy values where applicable (see ur3e_rl_ws/.../reward.py).
    # Tune later if needed.
    rew_scale_distance: float = -0.3            # -0.3 * dist(EE, target), always-on
    rew_scale_success: float = 10.0             # +10.0 on EE within tolerance (impulse)
    rew_scale_action: float = -0.01             # -0.01 * sum(action^2)
    rew_scale_time: float = -0.001              # -0.001 * step_count

    # ---- below-base-plane penalty ----
    # Penalises EE dropping below the "base plane" — the horizontal plane at
    # the robot's mount height. With the robot mounted on the pedestal at
    # z = robot_base_height_m = 1.0, the base plane is z = 1.0. The penalty
    # is linear in how far below the plane the EE drops, beyond a small
    # tolerance that allows for legitimate surface contact (Phase 4b grasping).
    #
    # Concept generalises: whether the robot is on a table, a pedestal, or
    # directly on the ground, "below the base plane" = below where the robot
    # is mounted = somewhere the EE shouldn't normally be in a workspace task.
    base_plane_tolerance_m: float = 0.02        # EE can dip 2cm below mount without penalty
    rew_scale_below_base_plane: float = -10.0   # per metre of depth below threshold; e.g. 5cm below -> -0.5

    # ---- v1-only smoothness extras (used by rewards/dense_reach_v1.py only) ----
    # All three terms below are ignored by the default `dense_reach.py`; only
    # `dense_reach_v1.py` reads them. Set scales to 0.0 to disable individually.
    #
    # Downward-reach incentive: penalty for EE being above the target's z.
    # Linear in the gap — nudges the policy to descend early rather than
    # swinging laterally first. Also read by v2.
    rew_scale_down_incentive: float = -0.3      # per metre of EE z above target z
    #
    # Action rate (smoothness between consecutive policy decisions): penalty
    # on the squared change between this step's action and the previous step's.
    # Standard low-pass-on-the-policy term — keeps motion from being twitchy.
    # `env.prev_actions` is cached in `_pre_physics_step` before the new action
    # is written, so the reward can compute `(actions - prev_actions)^2`.
    rew_scale_action_rate: float = -0.01        # per sum(action_delta^2) where action_delta = aₜ - aₜ₋₁
    #
    # Joint velocity (limit raw joint-space speeds): penalty on the squared
    # joint velocities reported by the articulation. Discourages fast joint
    # motion in the sim (which translates to natural-looking, calmer arm
    # behaviour and reduces the chance of overshoots).
    rew_scale_joint_vel: float = -0.005         # per sum(joint_vel^2) over the 6 arm joints

    # ---- v2-only smoothness extras (used by rewards/dense_reach_v2.py only) ----
    # V2 cranks v1's smoothness scales 4-5x and adds a jerk (second-difference)
    # term. Reads new *_v2 cfg fields so v1 and v2 coexist with different weights.
    # All three are ignored by dense_reach.py and dense_reach_v1.py.
    #
    # Action rate v2: 5x stronger than v1's -0.01. Bigger weight so the policy
    # actually pays attention to first-difference smoothness instead of letting
    # the success bonus dominate.
    rew_scale_action_rate_v2: float = -0.05     # per sum(action_delta^2); 5x v1
    #
    # Joint velocity v2: 4x stronger than v1's -0.005.
    rew_scale_joint_vel_v2: float = -0.02       # per sum(joint_vel^2); 4x v1
    #
    # Jerk: second action difference, sum((a_t - 2*a_(t-1) + a_(t-2))^2).
    # Targets the back-and-forth oscillation pattern that the first-difference
    # action_rate term misses. Requires env.prev_prev_actions cached in
    # _pre_physics_step (rolled one step before prev_actions). Zero on steps
    # 0 and 1 of each episode (reset clears it), so jerk is garbage for those
    # two steps but the spurious contribution is bounded.
    rew_scale_jerk_v2: float = -0.02            # per sum(jerk^2) over the 6 arm-joint action dims

    # ---------------------------------------------------------------- 9. scene element toggles
    # All visual / non-physics scene additions can be disabled to speed up
    # training or to inspect the env in isolation.
    add_ground_plane: bool = True               # flat infinite ground at z=0
    add_table: bool = True                      # static cuboid pedestal; top surface at robot_base_height_m
    table_size_xy_m: tuple[float, float] = (0.7, 0.7)   # 0.7 m x 0.7 m footprint
    # Pedestal-style table: full height from ground (z=0) up to the robot
    # mount (z=robot_base_height_m). With robot_base_height_m=1.0, the table
    # is a 0.7 x 0.7 x 1.0 m solid block that the robot stands on.
    table_thickness_m: float = 1.0              # full pedestal height; top at robot_base_height_m, bottom at z=0
    # Table extends slightly forward of the robot (-Y direction, where targets
    # spawn) so the robot sits on the BACK portion of the table and the
    # workspace is in front. With size 0.7x0.7 and offset -0.2, the table
    # spans y ∈ [-0.55, 0.15], comfortably containing the target y range
    # [-0.45, -0.1] AND leaving room for the robot base footprint at y=0.
    table_offset_y_m: float = -0.2              # table center pushed forward by 0.2m
    add_target_marker: bool = True              # small visual cube at the current target position
    target_marker_size_m: float = 0.04          # matches legacy cube edge length
