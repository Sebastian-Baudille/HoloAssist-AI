"""Gripper grasp sanity test — can RG2 hold a cube in Isaac?

The URDF's RG2 mimic chain was stripped during Isaac's URDF import (it would
have lost finite limits otherwise — see scripts/prepare_urdf.py). Without
mimic, we drive 6 linkage joints independently. The reach task held them at
fixed closed values; the upcoming grab task needs to actually open and close
the gripper around a cube.

This script verifies the grasp is physically viable BEFORE we commit to
building the grab task package. The flow is intentionally minimal:

    1. Robot starts at PRE_GRASP_POSE (gripper open, hovering just above the
       expected cube position on the table). Descent is baked into the pose
       — no separate "reach down" phase. Tune via --descent to set how low
       the gripper hovers.
    2. Cube teleports to its starting position directly under the EE on the
       table top.
    3. GRASP — close the gripper (extended sim time).
    4. LIFT — raise shoulder_lift back up (extended sim time).
    5. Optional SHAKE — rotate the base ±30° to test hold robustness.
    6. Verdict.

Pass criteria: cube z increased by > 5 cm after lift -> grasp works.
Fail recovery: try --friction 2.0/3.0 for slip; --descent for height.

Bypasses kinematics — the FK model in common/kinematics.py has an unverified
frame convention inherited from the ROS stack and gave wrong joint targets
in earlier test versions. Reading the EE position directly from the
simulator is more reliable.

Run from the IsaacLab directory:

    cd C:\\Users\\sebas\\Github\\IsaacLab
    .\\isaaclab.bat -p "C:\\Users\\sebas\\Github\\41118 Artificial Intelligence in Robotics\\HoloAssist-AI\\isaac_rl\\scripts\\grasp_test_v0.py"
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="RG2 grasp sanity test")
parser.add_argument("--descent",     type=float, default=0.25,
                    help="Radians ADDED to shoulder_lift descent in the pre-grasp pose. "
                         "Larger = gripper hovers lower. Tune so EE z ends up ~3-5 cm above cube top. "
                         "Default 0.25 rad puts EE roughly 5 cm above table.")
parser.add_argument("--lift-up",     type=float, default=0.50,
                    help="Radians SUBTRACTED from pre-grasp shoulder_lift for lift phase. "
                         "Larger = higher lift. Default 0.50 clearly elevates above table.")
parser.add_argument("--cube-size",   type=float, default=0.04, help="Cube edge length (m)")
parser.add_argument("--cube-mass",   type=float, default=0.05, help="Cube mass (kg)")
parser.add_argument("--friction",    type=float, default=1.0,  help="Cube static + dynamic friction")
parser.add_argument("--close-steps", type=int,   default=360,
                    help="Sim steps to hold the close command (1 step = 1/120 s)")
parser.add_argument("--lift-steps",  type=int,   default=360,
                    help="Sim steps to execute the lift")
parser.add_argument("--no-shake",    action="store_true",
                    help="Skip the shake verification phase")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import omni.usd  # noqa: E402

from holoassist_tasks.common.gripper_coupling import apply_rg2_mimic_tendon  # noqa: E402
from holoassist_tasks.tasks.direct.reach.reach_env_cfg import UR_ONROBOT_CFG  # noqa: E402


# ---- Constants ----

TABLE_TOP_Z = 1.0
CUBE_TABLE_Z = TABLE_TOP_Z + args_cli.cube_size / 2.0

# Pre-grasp robot pose. Descent baked into shoulder_lift via --descent flag.
#
#   shoulder_pan  = +pi          — flips arm to face -Y (table direction).
#                                  Isaac's robot has yaw=0; Guy's Gazebo has
#                                  yaw=pi baked in. We compensate via pan.
#   shoulder_lift = -pi/2 - descent — Guy's elbow-up baseline minus extra
#                                  descent to bring the gripper close to the
#                                  table top. CLI-tunable.
#   elbow         = -pi/2        ) Guy's elbow-up top-down approach seed.
#   wrist_1       = -pi/2        ) wrist_2=+pi/2 is what makes the gripper
#   wrist_2       = +pi/2        ) point straight DOWN.
#   wrist_3       =  0           )
PRE_GRASP_POSE = (
    math.pi,
    -math.pi / 2 - args_cli.descent,
    -math.pi / 2,
    -math.pi / 2,
    math.pi / 2,
    0.0,
)

# RG2 mimic replication (approximates the stripped xacro mimic chain).
# 0.0 -> all linkages at +/-CLOSED_ANGLE rad; 1.0 -> all at 0 + finger_width 0.085 m.
#
# GRIPPER_CLOSED_ANGLE: don't command full geometric closure (0.78 rad =
# fingers touching) because at high drive stiffness the drives keep pushing
# past the cube, causing PhysX contact penetration. Instead, command a
# "just barely closed around the cube" angle that respects the cube width.
# For a 4 cm cube in an 8.5 cm gripper: target finger separation ~3.6 cm
# (slight squeeze) -> joint angle ~0.78 * (8.5-3.6)/8.5 = 0.45 rad.
# Round up to 0.50 for a bit of safety margin (firm contact, no over-push).
GRIPPER_CLOSED_ANGLE = 0.50
GRIPPER_MAX_WIDTH    = 0.085
GRIPPER_LINKAGE_SIGNS = {
    "finger_joint":                +1.0,
    "left_inner_knuckle_joint":    -1.0,
    "left_inner_finger_joint":     +1.0,
    "right_outer_knuckle_joint":   -1.0,
    "right_inner_knuckle_joint":   -1.0,
    "right_inner_finger_joint":    +1.0,
}

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


# ---- Scene config ----

@configclass
class GraspTestSceneCfg(InteractiveSceneCfg):
    """Single robot + table + cube + lights. Cube spawned far away and
    teleported into position after the arm settles (Phase 2)."""

    num_envs: int = 1
    env_spacing: float = 4.0

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    sky = AssetBaseCfg(
        prim_path="/World/SkyLight",
        spawn=sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.90)),
    )
    sun = AssetBaseCfg(
        prim_path="/World/SunLight",
        spawn=sim_utils.DistantLightCfg(intensity=1500.0, angle=0.53),
    )

    table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.7, 0.7, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.32)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.2, 0.5)),
    )

    robot = UR_ONROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(args_cli.cube_size, args_cli.cube_size, args_cli.cube_size),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
            mass_props=sim_utils.MassPropertiesCfg(mass=args_cli.cube_mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=args_cli.friction,
                dynamic_friction=args_cli.friction,
                restitution=0.0,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
        ),
        # Spawn far away — teleported into position in Phase 2.
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.5, 1.05)),
    )


# ---- Helpers ----

def gripper_pose_for_open_fraction(open_fraction: float) -> dict[str, float]:
    """Map a single open/close signal in [0, 1] to all 7 gripper joints."""
    open_fraction = float(np.clip(open_fraction, 0.0, 1.0))
    linkage_magnitude = GRIPPER_CLOSED_ANGLE * (1.0 - open_fraction)
    pose = {"finger_width": GRIPPER_MAX_WIDTH * open_fraction}
    for joint, sign in GRIPPER_LINKAGE_SIGNS.items():
        pose[joint] = sign * linkage_magnitude
    return pose


def step_and_settle(sim, scene, n_steps: int, label: str | None = None) -> None:
    if label is not None:
        print(f"\n=== {label} ===", flush=True)
    for _ in range(n_steps):
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


def read_ee_pos(robot, ee_idx) -> np.ndarray:
    """Read a single finger body's world position (left_inner_finger by convention)."""
    return robot.data.body_link_state_w[0, ee_idx, :3].cpu().numpy()


def read_gripper_center(robot, left_idx, right_idx) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read both inner-finger world positions and return (left, right, center).

    The MIDPOINT between left_inner_finger and right_inner_finger is the actual
    grasp point — the cube needs to be teleported here, not to a single finger.
    If we use just left_inner_finger (as the reach task does for its EE proxy),
    the cube ends up offset to one side of the gripper centerline, and when the
    gripper closes the fingers move toward the centerline AWAY from the cube.
    """
    left   = robot.data.body_link_state_w[0, left_idx,  :3].cpu().numpy()
    right  = robot.data.body_link_state_w[0, right_idx, :3].cpu().numpy()
    center = (left + right) / 2.0
    return left, right, center


# ---- Main ----

def main():
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    # Camera: behind-and-above the robot, looking at the workspace (-Y side).
    sim.set_camera_view(eye=(1.0, 0.6, 1.7), target=(0.0, -0.3, 1.05))

    print("=== grasp_test_v0: main() starting ===", flush=True)

    scene_cfg = GraspTestSceneCfg(num_envs=1, env_spacing=4.0)
    scene = InteractiveScene(scene_cfg)
    print("  Scene built", flush=True)

    # Couple the RG2 linkage joints with a PhysX fixed tendon BEFORE
    # sim.reset() so PhysX initialises the articulation with the constraint
    # in place. Replicates the URDF mimic chain that was stripped during
    # import. Wrapped in try/except so the test still runs if the tendon
    # API name doesn't match this Isaac Sim build — we'd just lose the
    # mimic coupling (visual artifacts under contact, but grasp still works).
    stage = omni.usd.get_context().get_stage()
    try:
        apply_rg2_mimic_tendon(stage, "/World/envs/env_0/Robot")
    except Exception as e:
        import traceback
        print(f"\n[gripper_coupling] FAILED to apply tendon: {type(e).__name__}: {e}", flush=True)
        print("[gripper_coupling] Traceback:", flush=True)
        traceback.print_exc()
        print("[gripper_coupling] Continuing WITHOUT tendon — gripper will be back to "
              "independent-drive behavior (visual artifacts under contact expected).\n", flush=True)

    sim.reset()
    print("  sim.reset() done", flush=True)

    # ---- HIGH-STIFFNESS drive coupling on all linkage joints ----
    # Previous test proved PhysxMimicJointAPI is metadata-only in this Isaac
    # Sim 5.1 build (zeroing follower drives caused only one side to close
    # — definitive proof that no PhysX-level mimic enforcement is happening).
    # So we fall back to control-level coupling: command coupled targets to
    # all 6 joints (already done by gripper_pose_for_open_fraction), and
    # bump the drive stiffness MUCH higher so contact forces can't deform
    # the linkage out of its commanded shape.
    #
    # At stiffness 500: ~0.1-0.2 rad position error under contact (visible asymmetry).
    # At stiffness 10000+: ~0.005-0.01 rad position error (visually clean).
    # Stiffness sweet spot: high enough to keep linkage geometry visually
    # correct under contact, low enough that drives don't slap past the cube
    # before contact resolves.
    #   500   -> grasp works, visual asymmetric
    #   10000 -> visual fixed, but fingers slap past cube (no grasp)
    #   2500  -> compromise (try this)
    HIGH_STIFFNESS = 2500.0
    HIGH_DAMPING   = 100.0
    robot = scene.articulations["robot"]
    linkage_joint_names = (
        "finger_joint",                # master
        "left_inner_knuckle_joint",
        "left_inner_finger_joint",
        "right_outer_knuckle_joint",
        "right_inner_knuckle_joint",
        "right_inner_finger_joint",
    )
    linkage_ids = torch.tensor(
        [robot.find_joints(name)[0][0] for name in linkage_joint_names],
        dtype=torch.long, device=sim.device,
    )
    stiff_tensor = torch.full((1, len(linkage_ids)), HIGH_STIFFNESS, device=sim.device)
    damp_tensor  = torch.full((1, len(linkage_ids)), HIGH_DAMPING,   device=sim.device)
    try:
        robot.write_joint_stiffness_to_sim(stiff_tensor, joint_ids=linkage_ids)
        robot.write_joint_damping_to_sim(damp_tensor,    joint_ids=linkage_ids)
        print(f"  Bumped linkage drive stiffness {500} -> {HIGH_STIFFNESS:.0f} "
              f"on {len(linkage_ids)} joints (control-level coupling under contact)", flush=True)
    except Exception as e:
        print(f"  WARN: couldn't update linkage drives: {type(e).__name__}: {e}", flush=True)
    robot = scene.articulations["robot"]
    cube = scene.rigid_objects["cube"]

    arm_indices = [robot.find_joints(name)[0][0] for name in ARM_JOINTS]
    shoulder_pan_idx, shoulder_lift_idx = arm_indices[0], arm_indices[1]
    gripper_indices = {
        name: robot.find_joints(name)[0][0]
        for name in ("finger_width", *GRIPPER_LINKAGE_SIGNS.keys())
    }
    # Both inner-finger bodies are needed: the cube must be placed at their
    # MIDPOINT (gripper center), not at one finger's position.
    left_finger_idx  = robot.find_bodies("left_inner_finger")[0][0]   # same EE proxy as reach env
    right_finger_idx = robot.find_bodies("right_inner_finger")[0][0]
    ee_idx = left_finger_idx   # kept for compat with reach-task EE proxy

    # ---- Phase 0: settle ----
    step_and_settle(sim, scene, n_steps=60, label="Phase 0: settle initial state")

    # Build the pre-grasp joint vector (arm + gripper-open)
    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)
    for i, idx in enumerate(arm_indices):
        joint_pos[:, idx] = PRE_GRASP_POSE[i]
    for name, value in gripper_pose_for_open_fraction(1.0).items():
        joint_pos[:, gripper_indices[name]] = value

    # ---- Phase 1: drive robot to PRE-GRASP starting position ----
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    step_and_settle(sim, scene, n_steps=300, label="Phase 1: robot to PRE-GRASP position, gripper OPEN")
    left_finger, right_finger, gripper_center = read_gripper_center(robot, left_finger_idx, right_finger_idx)
    print(f"  Left finger pos:    ({left_finger[0]:+.3f}, {left_finger[1]:+.3f}, {left_finger[2]:.3f})", flush=True)
    print(f"  Right finger pos:   ({right_finger[0]:+.3f}, {right_finger[1]:+.3f}, {right_finger[2]:.3f})", flush=True)
    print(f"  Gripper center:     ({gripper_center[0]:+.3f}, {gripper_center[1]:+.3f}, {gripper_center[2]:.3f})", flush=True)
    print(f"  Gripper center above table: {(gripper_center[2] - TABLE_TOP_Z)*100:+.1f} cm", flush=True)
    print(f"  Finger separation:  {float(np.linalg.norm(left_finger - right_finger))*100:.1f} cm "
          f"(should be near {GRIPPER_MAX_WIDTH*100:.1f} cm when open)", flush=True)

    # ---- Phase 2: teleport cube to STARTING position (under gripper CENTER on table) ----
    print(f"\n=== Phase 2: cube to STARTING position (under gripper CENTER on table) ===", flush=True)
    cube_target = torch.zeros((1, 7), device=sim.device, dtype=torch.float32)
    cube_target[:, 0] = float(gripper_center[0])     # under midpoint between fingers
    cube_target[:, 1] = float(gripper_center[1])
    cube_target[:, 2] = CUBE_TABLE_Z
    cube_target[:, 3] = 1.0   # quaternion w=1, identity rotation
    cube.write_root_pose_to_sim(cube_target)
    cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device, dtype=torch.float32))
    step_and_settle(sim, scene, n_steps=90)   # let cube fall/settle on table

    cube_pos_settled = cube.data.root_pos_w[0].cpu().numpy()
    _, _, gripper_center_check = read_gripper_center(robot, left_finger_idx, right_finger_idx)
    xy_offset = float(np.linalg.norm(gripper_center_check[:2] - cube_pos_settled[:2]))
    z_offset  = float(gripper_center_check[2] - cube_pos_settled[2])
    print(f"  Cube settled at: ({cube_pos_settled[0]:+.3f}, {cube_pos_settled[1]:+.3f}, {cube_pos_settled[2]:.3f})", flush=True)
    print(f"  Gripper-cube XY offset: {xy_offset*100:.1f} cm "
          f"({'OK' if xy_offset < 0.02 else 'WARN: cube not centered between fingers'})", flush=True)
    print(f"  Gripper-cube  Z offset: {z_offset*100:+.1f} cm "
          f"({'OK' if -0.03 < z_offset < 0.05 else 'WARN: adjust --descent (target gripper at cube height)'})", flush=True)

    # ---- Phase 3: GRASP (close gripper, extended time) ----
    for name, value in gripper_pose_for_open_fraction(0.0).items():
        joint_pos[:, gripper_indices[name]] = value
    robot.set_joint_position_target(joint_pos)
    step_and_settle(sim, scene, n_steps=args_cli.close_steps,
                    label=f"Phase 3: GRASP — close gripper ({args_cli.close_steps} steps = {args_cli.close_steps/120:.1f} sec)")
    cube_after_close = cube.data.root_pos_w[0].cpu().numpy()
    print(f"  Cube position after gripper close: ({cube_after_close[0]:+.3f}, {cube_after_close[1]:+.3f}, {cube_after_close[2]:.3f})", flush=True)

    # ---- Phase 4: LIFT ----
    joint_pos[:, shoulder_lift_idx] = PRE_GRASP_POSE[1] + args_cli.lift_up
    robot.set_joint_position_target(joint_pos)
    step_and_settle(sim, scene, n_steps=args_cli.lift_steps,
                    label=f"Phase 4: LIFT (shoulder_lift += {args_cli.lift_up:.2f} rad, {args_cli.lift_steps} steps)")
    cube_after_lift = cube.data.root_pos_w[0].cpu().numpy()
    ee_after_lift   = read_ee_pos(robot, ee_idx)
    print(f"  EE position after lift:   ({ee_after_lift[0]:+.3f}, {ee_after_lift[1]:+.3f}, {ee_after_lift[2]:.3f})", flush=True)
    print(f"  Cube position after lift: ({cube_after_lift[0]:+.3f}, {cube_after_lift[1]:+.3f}, {cube_after_lift[2]:.3f})", flush=True)

    # ---- Phase 5: SHAKE (optional, verifies hold under perturbation) ----
    cube_after_shake = cube_after_lift
    if not args_cli.no_shake:
        joint_pos[:, shoulder_pan_idx] = math.pi + math.pi / 6   # rotate base +30 deg
        robot.set_joint_position_target(joint_pos)
        step_and_settle(sim, scene, n_steps=120, label="Phase 5a: shake +30 deg pan")
        joint_pos[:, shoulder_pan_idx] = math.pi - math.pi / 6   # rotate base -30 deg
        robot.set_joint_position_target(joint_pos)
        step_and_settle(sim, scene, n_steps=120, label="Phase 5b: shake -30 deg pan")
        cube_after_shake = cube.data.root_pos_w[0].cpu().numpy()
        print(f"  Cube position after shake: ({cube_after_shake[0]:+.3f}, {cube_after_shake[1]:+.3f}, {cube_after_shake[2]:.3f})", flush=True)

    # ---- Verdict ----
    initial_cube_z = CUBE_TABLE_Z
    final_cube_z   = float(cube_after_shake[2])
    lift_achieved  = final_cube_z - initial_cube_z

    print("\n" + "=" * 64, flush=True)
    print("VERDICT", flush=True)
    print("=" * 64, flush=True)
    print(f"  Cube z initial (on table): {initial_cube_z:.3f} m", flush=True)
    print(f"  Cube z final (after all):  {final_cube_z:.3f} m", flush=True)
    print(f"  Lift achieved:             {lift_achieved*100:+.1f} cm", flush=True)
    if lift_achieved > 0.05:
        print(f"  RESULT: GRASP WORKS — cube was picked up and held", flush=True)
    elif lift_achieved > 0.01:
        print(f"  RESULT: WEAK GRASP — some lift but cube slipped (try --friction 2.0)", flush=True)
    elif lift_achieved > -0.01:
        print(f"  RESULT: NO GRASP — cube stayed on table", flush=True)
        if xy_offset > 0.03:
            print(f"         Likely cause: gripper not over cube (XY offset {xy_offset*100:.1f} cm). "
                  f"Adjust PRE_GRASP_POSE or --descent.", flush=True)
        elif z_offset > 0.07:
            print(f"         Likely cause: gripper too high (Z offset {z_offset*100:+.1f} cm above cube). "
                  f"Try larger --descent.", flush=True)
        elif z_offset < 0.02:
            print(f"         Likely cause: gripper too low / past cube (Z offset {z_offset*100:+.1f} cm). "
                  f"Try smaller --descent.", flush=True)
        else:
            print(f"         Likely cause: friction or actuator stiffness. Try --friction 2.0 or 3.0.", flush=True)
    else:
        print(f"  RESULT: CUBE FELL — cube was knocked off the table", flush=True)
    print("=" * 64, flush=True)

    if args_cli.headless:
        return
    print("\nGUI still open. Press Ctrl+C in terminal to exit.", flush=True)
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
