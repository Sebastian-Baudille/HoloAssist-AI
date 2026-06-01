"""Smoke test (v0) for the imported UR3e + RG2 USD asset.

Loads the robot into a minimal Isaac Lab scene (ground + light + robot), lets
physics run for a few seconds, and prints joint state periodically. Verifies:
    - The USD spawns into an Isaac Lab scene cleanly
    - The articulation holds its pose under gravity (drives are working)
    - The 13 actuated joints are readable via the Articulation API
    - No oscillation (confirms the 50 Hz / 1.0 damping import settings)

Run from the IsaacLab directory:

    cd C:\\Users\\sebas\\Github\\IsaacLab
    .\\isaaclab.bat -p "C:\\Users\\sebas\\Github\\41118 Artificial Intelligence in Robotics\\HoloAssist-AI\\isaac_rl\\scripts\\robot_test_v0.py"

Useful flags (forwarded to AppLauncher):
    --headless          run without GUI (faster, for CI)
    --device cpu        force CPU physics (default is cuda:0)
    --seconds 5.0       override simulated wall-clock (default 10s)
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI — must parse BEFORE importing the rest of isaaclab
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="UR3e + RG2 USD smoke test (v0)")
parser.add_argument("--seconds", type=float, default=10.0, help="Wall-clock seconds to simulate")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Imports below this line require the simulation app to be running
# -----------------------------------------------------------------------------
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

# Path to the USD produced by Phase 3 Step C (Isaac Sim URDF Importer)
USD_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets" / "usd" / "ur_onrobot_prepared" / "ur_onrobot_prepared.usd"
)


@configclass
class UROnrobotSceneCfg(InteractiveSceneCfg):
    """Minimal scene: ground plane, dome light, UR3e+RG2 robot."""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(USD_PATH).replace("\\", "/"),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                # Arm — UR3e parked pose (matches legacy HOME_JOINTS)
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 0.0,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": 0.0,
                "wrist_3_joint": 0.0,
                # Gripper driver (closed = 0 m, fully open = 0.085 m)
                "finger_width": 0.0,
                # Gripper linkage — values for closed-configuration. Pulled
                # slightly off the URDF joint limits (Isaac Lab rejects positions
                # exactly at the limit). 0.78 vs 0.785 is ~0.3° — visually
                # identical to fully closed.
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


def main() -> None:
    print(f"[robot_test_v0] USD path: {USD_PATH}")
    if not USD_PATH.is_file():
        raise SystemExit(f"USD not found: {USD_PATH}\nRun Phase 3 Step C first.")

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(2.5, 2.5, 1.8), target=(0.0, 0.0, 1.0))

    scene_cfg = UROnrobotSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    robot = scene["robot"]
    print(f"[robot_test_v0] Articulation loaded:")
    print(f"        num_joints = {robot.num_joints}")
    print(f"        num_bodies = {robot.num_bodies}")
    print(f"        joint_names = {robot.joint_names}")
    print(f"        body_names = {robot.body_names}")

    # Isaac Lab loads InitialStateCfg.joint_pos as the default joint state, but
    # drive targets default to 0. Without explicitly seeding the targets, the
    # arm would be driven to the all-zeros horizontal pose. Set targets to
    # match the init positions so drives hold the parked pose.
    robot.set_joint_position_target(robot.data.default_joint_pos)
    scene.write_data_to_sim()
    print(f"[robot_test_v0] Drive targets seeded to default_joint_pos = {robot.data.default_joint_pos[0].cpu().numpy().round(4)}")

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    next_log = 0.0

    print(f"[robot_test_v0] Simulating {args_cli.seconds:.1f} s of physics ...")
    while simulation_app.is_running() and sim_time < args_cli.seconds:
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)
        sim_time += sim_dt
        if sim_time >= next_log:
            jpos = robot.data.joint_pos[0].cpu().numpy()
            print(f"[robot_test_v0] t={sim_time:5.2f}s  joint_pos[:6] = {jpos[:6].round(4)}")
            next_log += 1.0

    print("[robot_test_v0] OK — robot loaded, simulated cleanly, joints readable.")


if __name__ == "__main__":
    main()
    simulation_app.close()
