"""
test_ik_visual.py — Visual IK alignment test.

Spawns the robot and random cubes, then smoothly drives the arm to the
IK reference pose (above the nearest cube, gripper pointing down).
Lets you verify that:
  - The robot spawns upright
  - The IK target lands the gripper directly above the cube
  - Coordinates are correctly aligned

Usage:
    python3 test_ik_visual.py
    python3 test_ik_visual.py --episodes 10 --speed 0.4
"""
from __future__ import annotations

# Fix HiDPI Wayland scaling — must be before any glfw/mujoco import
import os as _os, glob as _g
_x11 = next(iter(_g.glob(
    _os.path.expanduser("~/.local/lib/python*/site-packages/glfw/x11/libglfw.so")
)), None)
if _x11:
    _os.environ.setdefault("PYGLFW_LIBRARY", _x11)

import argparse
import time
import sys
from pathlib import Path
import numpy as np

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,   default=5,
                        help="Number of random cube spawns to test")
    parser.add_argument("--speed",    type=float, default=0.3,
                        help="Playback speed (1.0 = real time)")
    parser.add_argument("--hold",     type=float, default=2.0,
                        help="Seconds to hold at target before next episode")
    args = parser.parse_args()

    from ur3e_rl_env.envs.ur3e_mujoco_env import UR3eMuJoCoEnv, HOME_JOINTS
    from ur3e_rl_env.kinematics import forward_kinematics, fk_tcp_z_axis

    print("Opening MuJoCo viewer...")
    env = UR3eMuJoCoEnv(render_mode="human")

    # Step delay: 50 physics steps × 0.002 s = 0.1 s per RL step at real-time
    step_sleep = 0.1 / max(args.speed, 0.01)

    print(f"\nIK alignment test — {args.episodes} episodes at {args.speed}x speed")
    print("Watch the gripper: it should land directly above each cube.\n")

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset()
        target_cube = info["target_cube"]

        # Get IK reference for this episode's target cube
        ik_joints = env._ik_reference_joints
        cube_pos   = env._get_cube_pos(env._nearest_cube_idx(env._get_ee_pos()))
        tcp_start  = env._get_ee_pos()

        if ik_joints is None:
            print(f"Episode {ep}: IK failed for {target_cube} at {cube_pos.round(3)} — skipping")
            continue

        ik_tcp  = forward_kinematics(ik_joints.astype(float))
        ik_zax  = fk_tcp_z_axis(ik_joints.astype(float))
        approach = cube_pos + np.array([0, 0, 0.07])   # 7 cm above cube

        print(f"Episode {ep}/{args.episodes} — {target_cube}")
        print(f"  Cube pos:        {cube_pos.round(3)}")
        print(f"  IK approach tgt: {approach.round(3)}")
        print(f"  IK FK result:    {ik_tcp.round(3)}  (error {np.linalg.norm(ik_tcp-approach)*100:.1f} cm)")
        print(f"  IK Z-axis:       {ik_zax.round(3)}  (want [0,0,-1])")

        # --- Smoothly move from home to IK target ---
        # Interpolate joint angles over N steps
        current = env._get_joint_positions().astype(float)
        N_MOVE = 60   # steps to travel from home to target
        print(f"  Moving to IK pose ({N_MOVE} steps)...", end="", flush=True)

        for i in range(N_MOVE):
            if env._viewer is not None and not env._viewer.is_running():
                print("\nViewer closed.")
                env.close()
                return

            t = (i + 1) / N_MOVE          # 0→1
            t = t * t * (3 - 2 * t)        # smooth-step easing
            cmd = current + t * (ik_joints - current)

            # Build a 7-D action: first 6 = joint targets (denormalized to [-1,1])
            joint_range = (env._joint_upper - env._joint_lower) / 2.0
            joint_mid   = (env._joint_upper + env._joint_lower) / 2.0
            norm_cmd    = (cmd - joint_mid) / joint_range
            action = np.append(norm_cmd, 0.0).astype(np.float32)

            # Directly set ctrl so we bypass the slew-rate limit for this test
            env.data.ctrl[:6] = cmd
            import mujoco
            for _ in range(env.model.opt.timestep and 50 or 50):
                mujoco.mj_step(env.model, env.data)
            if env._viewer is not None:
                env._viewer.sync()
            time.sleep(step_sleep)

        # Read actual TCP after move
        actual_tcp = env._get_ee_pos()
        pos_err    = np.linalg.norm(actual_tcp - approach) * 100

        print(f" done")
        print(f"  Actual TCP:      {actual_tcp.round(3)}")
        print(f"  Position error:  {pos_err:.1f} cm  {'✓ GOOD' if pos_err < 5 else '✗ BAD'}")
        print(f"  Holding {args.hold:.0f}s — inspect in viewer...")
        print()

        # Hold at target so you can inspect in the viewer
        t_hold = time.monotonic()
        while time.monotonic() - t_hold < args.hold:
            if env._viewer is not None and not env._viewer.is_running():
                env.close()
                return
            env._viewer.sync()
            time.sleep(0.05)

    print("All episodes complete.")
    import os; os._exit(0)

if __name__ == "__main__":
    main()
