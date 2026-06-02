# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Config for the HoloAssist UR3e grab task.

Phase 4b task: reach to a randomly placed cube, close the gripper around it,
lift it >5 cm above the table. Successor to the reach task; reuses the same
robot articulation cfg (UR_ONROBOT_CFG) but extends:
  - Action space 6 -> 7 (adds gripper open/close signal)
  - Observation space 12 -> 16 (adds cube state + relative position)
  - Reward 5 terms -> 6 terms (adds xy_align + orient_align + grasp + lift + success)
  - Scene: rigid-body cube on the table (vs reach's visualization marker)

Locked decisions (per project-phase4b-grab-decisions memory):
  - Learned 7D gripper (not scripted)
  - Random cube spawn in reach's spawn zone
  - Fixed ready home pose (joint_noise_rad CLI hook for later use)
  - Combined single-model end-to-end (not staged)
  - No transport (lift = success)
  - Train from scratch initially (v1 init has obs/action dim mismatch — Phase 4b-tune)
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

# Reuse the same robot articulation cfg from the reach task — same USD, same
# actuator setup. We only override per-task settings (action/obs spaces, etc.)
from holoassist_tasks.tasks.direct.reach.reach_env_cfg import UR_ONROBOT_CFG


@configclass
class HoloassistGrabEnvCfg(DirectRLEnvCfg):
    """Configuration for the UR3e grab task."""

    # ---------------------------------------------------------------- 1. env timing + spaces
    decimation = 4                              # 30 Hz policy at sim dt 1/120
    episode_length_s = 6.7                      # ~200 control steps
    action_space = 7                            # 6 arm deltas + 1 gripper signal
    observation_space = 16                      # 6 joint + 3 EE + 3 cube + 3 delta + 1 width
    state_space = 0

    # ---------------------------------------------------------------- 2. simulation + scene
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 120.0, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=64,
        env_spacing=2.5,
        replicate_physics=True,
    )

    # ---------------------------------------------------------------- 3. robot
    robot_base_height_m: float = 1.0
    robot_cfg: ArticulationCfg = UR_ONROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
    )

    # Linkage drive stiffness applied via write_joint_stiffness_to_sim in env init.
    # Found empirically in grasp_test_v0: this is the sweet spot between
    # "grasp works but linkage deforms under contact" (low stiffness) and
    # "linkage stays rigid but fingers slap past cube" (very high stiffness).
    linkage_drive_stiffness: float = 2500.0
    linkage_drive_damping: float = 100.0

    # ---------------------------------------------------------------- 4. action scaling
    # Joint-delta semantics (same as reach task). 0.08 rad/step at 30 Hz = 2.4 rad/s.
    action_scale_rad: float = 0.08

    # Gripper signal mapping. action[6] in [-1, 1] -> linkage_magnitude in [0, closed_angle].
    # closed_angle 0.50 (not 0.78 = full closure): leaves room around cube so drives
    # don't try to push past it. Found empirically in grasp_test_v0.
    gripper_closed_angle: float = 0.50
    gripper_max_width: float = 0.085           # m, finger_width prismatic at fully open

    # ---------------------------------------------------------------- 5. home pose + noise
    # Fixed ready pose (matches Guy's ROS envs + our reach task). Joint noise
    # at reset is OFF by default; set joint_noise_rad > 0 to add uniform
    # ±noise to each arm joint at reset for robustness training (Phase 6+).
    home_joint_pos: list[float] = [0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0]
    joint_noise_rad: float = 0.0               # 0 = fixed; e.g. 0.05 = ±~3° per joint

    # ---------------------------------------------------------------- 6. cube spawn
    # Cube position randomized at every reset within these bounds. Matches the
    # reach task's target spawn zone (so the policy has consistent expectations
    # for "where the object is" across tasks).
    cube_size_m: float = 0.04
    cube_mass_kg: float = 0.05
    cube_friction: float = 1.0                 # static = dynamic; --friction in grasp test
    cube_pos_range_x: tuple[float, float] = (-0.20, 0.20)
    cube_pos_range_y: tuple[float, float] = (-0.45, -0.10)
    cube_pos_z: float = 1.02                   # = robot_base_height_m + cube_size_m/2

    # ---------------------------------------------------------------- 7. termination thresholds
    # Failure: EE drops below the robot's mount height minus this clearance
    # (table-crash guard). Matches reach task pattern but tightened — grab
    # starts with EE closer to the table so the safety margin can be smaller.
    min_ee_clearance_below_base_m: float = 0.5

    # Success: cube lifted at least this high above table top.
    success_lift_height: float = 0.05

    # ---------------------------------------------------------------- 8. reward scales
    # All scales validated as roughly balanced in the design table. Tune later
    # if training metrics show one term dominating.

    # Reward-module selector. Default = v0 (dense_grab.py). Subclasses
    # override this to swap in a different reward module without touching
    # the env class. Resolved by grab_env._REWARD_MODULES dict.
    reward_module: str = "dense_grab"

    # Term 1: dense reach pull (always-on, negative gradient toward cube)
    rew_scale_reach: float = -0.3

    # Term 2: XY alignment (gripper centered over cube)
    # Toned down from 3.0 → 0.5 to prevent the "hover and collect alignment
    # bonus" attractor. Over a 200-step episode, perfect hover accumulates
    # 200 × 0.5 = +100 — less than the success bonus, so success dominates.
    rew_scale_xy_align: float = 0.5
    xy_alignment_threshold: float = 0.05       # m, bonus zero at >=5 cm XY offset

    # Term 3: orientation alignment (gripper pointing down)
    # Toned down from 2.0 → 0.3 for the same reason as xy_align.
    rew_scale_orient_align: float = 0.3
    orient_alignment_threshold: float = 0.5    # ||z_axis - (0,0,-1)|| = sqrt(2) at 90°
                                                # bonus zero at err >= 0.5 (~28° off-down)

    # Gate (terms 2 + 3 only active when EE is vertically close to the cube)
    alignment_z_gate: float = 0.10             # m, gate when |EE_z - cube_z| < 10 cm

    # Term 4: grasp activation (gripper closing signal when near cube)
    # Toned down from 5.0 → 1.0. Also tightened: in dense_grab.py the
    # closing_mask now requires the gripper to ACTUALLY be closing (width < 7cm),
    # not just the action signal — prevents the "random gripper signal earns
    # bonus" reward hack.
    rew_scale_grasp_activation: float = 1.0
    grasp_distance: float = 0.04               # m, distance threshold for "near"

    # Term 5: lift bonus (proportional to height, gated on grasped state)
    # Boosted from 1.0 → 50.0 per metre. Previous value made lift_bonus
    # negligible (1.0 × 0.05m = +0.05/step at successful lift). New value
    # gives +2.5/step at peak lift — the dominant guidance signal for "lift
    # the cube" beyond the terminal success bonus.
    rew_scale_lift: float = 50.0               # per metre lifted
    grasped_gripper_width: float = 0.05        # gripper considered closed when width < 5 cm
    grasped_distance: float = 0.08             # AND gripper center within 8 cm of cube

    # Term 6: success bonus (terminal)
    # Boosted from 50.0 → 200.0. Makes the terminal goal clearly the most
    # rewarding outcome — outweighs the cumulative reward from any
    # "hover-and-collect" alternative.
    rew_scale_success: float = 200.0

    # ---------------------------------------------------------------- 9. scene element toggles
    add_ground_plane: bool = True
    add_table: bool = True
    table_size_xy_m: tuple[float, float] = (0.7, 0.7)
    table_thickness_m: float = 1.0             # full pedestal from z=0 to z=robot_base_height
    table_offset_y_m: float = -0.2             # table center pushed forward


# ============================================================================
# V1 cfg — overhead-approach reward + PhysX self-collision + higher lift goal
# ============================================================================

@configclass
class HoloassistGrabEnvCfgV1(HoloassistGrabEnvCfg):
    """V1 of the grab task. Three changes vs v0:

    1. Self-collision ENABLED on the articulation (PhysX). v0's policy
       exploited arm-through-arm fold poses that wouldn't survive on real
       hardware; v1 makes them physically impossible at the sim level.

    2. Reward swapped to dense_grab_v1 (ungated orient + approach_height
       bonus). Together these shape the trajectory into the industrial
       "fly-over → orient-down → descend" pattern rather than the
       side-sprawl v0 learned.

    3. Success lift threshold raised from 5 cm -> 10 cm (and the lift
       reward scale bumped to maintain gradient strength past the higher
       threshold).

    Everything else (action space, obs space, cube spawn zone, home pose,
    drive stiffness) inherits from HoloassistGrabEnvCfg unchanged.
    Train via Template-Holoassist-Grab-Direct-v1; v0 task name and
    behaviour are completely unchanged.
    """

    # ---- 1. self-collision via PhysX (replaces v0's no-self-collision spawn) ----
    robot_cfg: ArticulationCfg = UR_ONROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UR_ONROBOT_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
    )

    # ---- 2. reward module swap + shaping overrides ----
    reward_module: str = "dense_grab_v1"

    # Orient pull is now applied always (not only near cube). Boosted so
    # it remains a meaningful gradient through the entire trajectory.
    rew_scale_orient_align: float = 1.5

    # Lift scale boosted from 50 -> 80/m so gradient strength stays similar
    # despite the higher success_lift_height ceiling.
    rew_scale_lift: float = 80.0

    # NEW v1-only field: rewards EE being above the cube during far approach.
    # Combined with the always-on orient_align, this is what produces the
    # "fly-over-then-descend" approach trajectory.
    rew_scale_approach_height: float = 2.0
    approach_far_threshold: float = 0.08       # m, "far" = dist > this

    # ---- 3. success threshold raised ----
    success_lift_height: float = 0.10          # m, lift > 10 cm (was 5 cm)


# ============================================================================
# V2 cfg — rebalanced scales + time penalty (breaks v1's hover trap)
# ============================================================================

@configclass
class HoloassistGrabEnvCfgV2(HoloassistGrabEnvCfgV1):
    """V2 of the grab task. Inherits v1's architectural choices (self-collision
    ON, 10 cm lift target, ungated orient direction, approach_height term).
    Only the reward scales change, plus a new time_penalty term.

    Background: v1's policy learned a beautiful overhead approach but never
    descended to grasp — it found that hovering perfectly oriented above the
    cube earned ~+316 per episode (mostly from the ungated orient_align at
    scale 1.5 and approach_height at scale 2.0). v2 fixes this by:

      1. Slashing standing-reward magnitudes (orient 1.5->0.3, approach 2.0->0.5)
         so cumulative hover reward drops to ~+75/episode.
      2. Adding a per-step time_penalty of -0.5, making hover net-negative
         (~-100/episode for 200 steps of doing nothing).
      3. Boosting grasp_activation (1.0->5.0), lift (80->100), and success
         (200->300) so the descent-grasp-lift trajectory becomes dramatically
         more rewarding than any standing-still policy.

    Reward-balance sanity check:
      - Hover for 200 steps: -9 + 60 + 15 - 100 = -34 (NET NEGATIVE)
      - Successful grasp + lift: ~-50 + 250 + 250 + 300 = ~+750
      - Contrast: ~+800. Strong gradient toward success.

    Same env class as v0/v1 (HoloassistGrabEnv); switched via cfg fields
    (reward_module="dense_grab_v2"). Logs land in grab-r2-run1.
    """

    # ---- Swap reward module ----
    reward_module: str = "dense_grab_v2"

    # ---- Standing-reward magnitudes reduced ----
    rew_scale_orient_align: float = 0.3        # was 1.5 in v1
    rew_scale_approach_height: float = 0.5     # was 2.0 in v1

    # ---- Grasp/lift/success boosted ----
    rew_scale_grasp_activation: float = 5.0    # was 1.0 in v0/v1
    rew_scale_lift: float = 100.0              # was 80 in v1
    rew_scale_success: float = 300.0           # was 200 in v0/v1

    # ---- NEW v2-only field: time penalty ----
    # Constant negative reward per step. Forces forward progress; makes
    # any "do nothing" policy provably worse than a completing one.
    # -0.5/step * 200 steps = -100 cost for timing out.
    rew_scale_time_penalty: float = -0.5


# ============================================================================
# V3 cfg — back to v0 reward shape + self-collision + posture nudge
# ============================================================================

@configclass
class HoloassistGrabEnvCfgV3(HoloassistGrabEnvCfgV1):
    """V3 of the grab task — strategic retreat to v0's proven design.

    Inherits self-collision from V1's robot_cfg override (only thing kept from
    v1's design — the hard physics constraint). Reward MODULE switches to
    dense_grab_v3, which is structurally identical to v0's dense_grab.py plus
    one small posture term.

    Why this design (after v1 + v2 failures):
        v0 trained successfully (97% grasp + lift, mean_reward 195/200).
        Its only flaws were COSMETIC (side-sprawl posture, arm-through-arm
        self-collision) — both physics issues, not reward issues. v1 and v2
        each tried to fix posture by enriching rewards, and each created
        new exploits that prevented completion. V3 fixes posture via:
          1. Self-collision (hard constraint, inherited from V1)
          2. Tiny elbow_up reward (gentle nudge, max 30/episode)
        ...without touching v0's reward balance.

    DESIGN INVARIANT (rule that v1/v2 violated):
        Maximum cumulative per-step reward across an episode MUST be less
        than rew_scale_success. Otherwise the policy finds an exploit that
        avoids termination to accumulate per-step reward.

        V3 numbers:
            max per-step ~= grasp_act(1) + lift(10*0.09) + xy(0.5) + orient(0.3)
                           + elbow_up(0.15) = ~2.85/step
            max episode  ~= 570
            success bonus  = 800
            margin         = +230 (success dominates)

    Same env class (HoloassistGrabEnv); switched via cfg fields.
    Logs land in grab-r3-run1.
    """

    # ---- Reward module: v3 (v0's shape + elbow_up) ----
    reward_module: str = "dense_grab_v3"

    # ---- Reward scales: BACK TO v0 baseline (overriding v1's boosts) ----
    # v1 set orient to 1.5 ungated; v3 reverts to v0's 0.3 (and v3 reward
    # module re-applies proximity gating). v0's grasp_activation (1.0) and
    # other scales are already correct in the base cfg.
    rew_scale_orient_align: float = 0.3

    # Lift scale REDUCED from v0's 50 because success_lift_height went up
    # from 5cm -> 10cm. At 10*0.09*200 = 180 cumulative max, leaves clear
    # margin to the 800 success bonus.
    rew_scale_lift: float = 10.0

    # Success bonus BOOSTED from v0's 200 -> 800 to maintain absolute
    # dominance over any per-step accumulation pathway.
    rew_scale_success: float = 800.0

    # success_lift_height stays at 0.10 (inherited from V1 — that's the
    # whole point of doing this iteration).

    # ---- NEW v3-only field: elbow-up posture nudge ----
    # Rewards forearm_link's world-frame Z position when above threshold.
    # Linearly ramps from threshold to threshold+clamp_max.
    # Designed-small: max 0.5 * 0.3 = 0.15/step -> 30/episode max.
    rew_scale_elbow_up: float = 0.5
    elbow_up_threshold_z: float = 1.1      # m, table top is at 1.0
    elbow_up_clamp_max: float = 0.3        # m, cap reward at forearm_z = 1.4


# ============================================================================
# V4 cfg — v3 + anti-drag penalty + grasp_act boost (breaks finger-drag trap)
# ============================================================================

@configclass
class HoloassistGrabEnvCfgV4(HoloassistGrabEnvCfgV3):
    """V4 of the grab task. Inherits everything from V3 (which itself inherits
    v0's reward shape + V1's self-collision + 10 cm lift target + elbow_up
    posture nudge). Two coupled changes target the v3 "finger drag" trap:

      1. New anti-drag penalty term: -1.0/step when either finger tip Z is at
         or below the table surface. Directly punishes the failure state
         observed in v3 visuals (gripper aligns correctly but fingers descend
         to table level where they cannot physically close).
      2. grasp_activation scale boosted from v3's 1.0 -> 2.0. Makes the
         descent-and-close transition substantially more rewarding so PPO
         finds it via gradient (v3's grasp_act was too small relative to the
         750/episode the policy was earning from accumulated alignment + brief
         partial-grasp cycles).

    Reward balance (invariant: max non-success per-step accumulation < success):

        Per-step max sustained closing-and-holding state:
            grasp_act 2.0 + lift 10*0.09 + xy 0.5 + orient 0.3 + elbow 0.15 = 3.85/step
        Over 200 steps:                                                    = 770
        Success bonus:                                                     = 800
        Margin:                                                            = +30

        Anti-drag applies independently:
            v3-style dragging earns ~750/episode -> v4 same state: 750 - 200 = 550
            Closing-and-grasping (clear of table): ~770 + brief lift bursts
            Successful completion: ~770 path + 800 terminal = ~1500

    Logs land in grab-r4-run1.
    """

    # ---- Reward module: v4 (v3 shape + anti-drag) ----
    reward_module: str = "dense_grab_v4"

    # ---- Boost grasp_activation: 1.0 -> 2.0 ----
    # Stronger closing signal so PPO can find the descent-and-close path
    # via gradient instead of needing accidental successes (which v3
    # couldn't have because 10cm threshold is unreachable by chance).
    rew_scale_grasp_activation: float = 2.0

    # ---- NEW v4-only field: anti-drag penalty ----
    # Per-step penalty when either finger's body link frame is within
    # `drag_threshold_above_table` of the table surface. The reward uses
    # the inner_finger body link frame, whose origin is at the knuckle
    # (joint anchor) ~5 cm above the actual finger tip — so the threshold
    # needs to be ~6 cm above the table top to detect "tips on table".
    rew_scale_drag_penalty: float = -1.0
    # Threshold tuning history:
    #  - 0.005 (original): never fired — inner_finger link frame is much higher than tips
    #  - 0.06 : never fired — link frame ~1.56 m in home pose, ~1.15 m when tips on table
    #  - 0.20 (current): targets ~1.20 m, catches the "tips on table" state cleanly.
    #    Brief overlap with legitimate grasp descent is acceptable (the policy
    #    escapes by closing/lifting fast).
    drag_threshold_above_table: float = 0.20


# ============================================================================
# V5 cfg — conservative return to v0 scales + self-collision + elbow_up
# ============================================================================

@configclass
class HoloassistGrabEnvCfgV5(HoloassistGrabEnvCfg):
    """V5: v0's proven reward + PhysX self-collision + small elbow_up posture nudge.

    Strategic retreat after v1/v2/v3/v4 each failed to find grasping at the
    10cm threshold. v0 is the only design empirically proven to work (97%
    success at 5cm lift). v5 keeps v0's scales unchanged and adds only:

      1. PhysX self-collisions (prevents arm folding through itself, which
         was v0's only visible flaw)
      2. Tiny elbow_up posture nudge (max 30/episode — too small to break
         v0's reward balance, but enough to gently bias toward overhead reach)

    Inherits the BASE cfg (v0 scales) — NOT V1, V3, or V4 which all
    overrode reward scales in ways that broke v0's discovery path.

    Why 5cm threshold matters:
        v0 worked because random closures during exploration sometimes lift
        the cube enough to fire the +200 success bonus. The policy learns
        "close gripper -> good" from those accidental successes. At 10cm
        threshold this lucky path is closed (random closures don't lift
        that high), so PPO can never discover grasping. v1/v3/v4 all hit
        this wall in different forms.

    Reward-balance note:
        Per-step max under v0 scales (lift 50 at 4cm) reaches ~3.95/step
        which over 200 steps is ~790 -- theoretically more than the +200
        success bonus. v0 empirically didn't exploit this because the
        geometric grasped flag is strict (cube must actually be held +
        lifted, hard to maintain via random actions). v5 trusts the same.

    Logs land in grab-r5-run1.
    """

    # Override robot_cfg to enable PhysX self-collisions (same pattern as V1)
    robot_cfg: ArticulationCfg = UR_ONROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UR_ONROBOT_CFG.spawn.replace(
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
    )

    # Reward module: v5 = v0's 6 terms + elbow_up
    reward_module: str = "dense_grab_v5"

    # ---- NEW v5-only fields: elbow_up posture nudge ----
    # Rewards forearm_link's world-frame Z position when above threshold.
    # Same fields as V3; safe because magnitude is small.
    rew_scale_elbow_up: float = 0.5
    elbow_up_threshold_z: float = 1.1      # m, table top is at 1.0
    elbow_up_clamp_max: float = 0.3        # m, cap reward at forearm_z = 1.4

    # Everything else (rew_scale_reach, _xy_align, _orient_align,
    # _grasp_activation, _lift, _success, success_lift_height, etc.) inherits
    # from HoloassistGrabEnvCfg (the v0 baseline). NOT overridden here.

