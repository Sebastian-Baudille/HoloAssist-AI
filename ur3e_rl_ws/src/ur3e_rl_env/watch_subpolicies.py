"""
watch_subpolicies.py — Run pan → pan_locked_reach sub-policies in sequence.

Usage:
    PYTHONPATH=. python3 watch_subpolicies.py \
        --pan   ../../rl_models/pan/best/best_model.zip \
        --reach ../../rl_models/pan_locked_reach/best/best_model.zip \
        --speed 1.0
"""
import argparse, time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

from ur3e_rl_env.envs.pan_env import UR3ePanEnv, _pan_err
from ur3e_rl_env.envs.pan_locked_reach_env import (
    LIFT_SCALE, ELBOW_SCALE, W1_SCALE, LIFT_LIM, ELBOW_LIM, W1_LIM,
    _FIXED_W2, PHYSICS_STEPS, TARGET_Z_ABOVE, _w3_from_cube_yaw,
)
from ur3e_rl_env.ik import build_ik_cache

PAN_THRESHOLD = np.radians(5)    # switch PAN → REACH
SIM_DT = 20 * 0.002              # 0.04 s per RL step (pan env)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pan",      required=True)
    ap.add_argument("--reach",    required=True)
    ap.add_argument("--speed",    type=float, default=1.0)
    ap.add_argument("--episodes", type=int,   default=0)
    args = ap.parse_args()

    print("Loading models...")
    pan_model   = PPO.load(args.pan,   device="cpu")
    reach_model = PPO.load(args.reach, device="cpu")

    env      = UR3ePanEnv(render_mode="human")
    ik_cache = env._ik_cache
    wall_per_step = SIM_DT / args.speed

    def get_joints():
        return np.array([env.data.qpos[a] for a in ik_cache["arm_qpos_addrs"]], dtype=np.float32)

    def get_ee():
        return env.data.xpos[ik_cache["tcp_body_id"]].astype(np.float32)

    def get_cube():
        return env.data.xpos[env._cube_body_ids[0]].astype(np.float32)

    episode    = 0
    ep_rew     = 0.0
    state      = "PAN"
    pan_steps  = reach_steps = grip_step = 0
    prev_state = None
    locked_pan = 0.0
    locked_w3  = np.pi / 2
    target     = np.zeros(3, dtype=np.float32)
    obs, _     = env.reset(seed=0)

    GRIP_STEPS   = 30    # steps to close gripper
    GRIP_HOLD    = 20    # steps to hold closed before lifting
    LIFT_STEPS   = 40    # steps to lift after gripping
    GRIP_MAX     = 1.3   # fully closed

    print(f"\nSub-policy viewer — {args.speed}x speed")
    print("States: [PAN] → [REACH] → [GRIP]\n")

    try:
        while env._viewer is not None and env._viewer.is_running():
            t0 = time.monotonic()

            cube_pos = get_cube()
            joints   = get_joints()
            ee_pos   = get_ee()
            pan      = float(joints[0])
            phi0     = env._phi0
            pan_err  = _pan_err(pan, cube_pos, phi0)
            err      = abs(pan_err)
            dist3d   = float(np.linalg.norm(ee_pos - target)) if state == "REACH" else 0.0

            # ── PAN ──────────────────────────────────────────────────────────
            if state == "PAN":
                pan_steps += 1
                pan_obs = np.array([np.cos(pan_err), np.sin(pan_err)], dtype=np.float32)
                action, _ = pan_model.predict(pan_obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                if err <= PAN_THRESHOLD:
                    locked_pan = float(joints[0])
                    target     = cube_pos + np.array([0.0, 0.0, TARGET_Z_ABOVE], dtype=np.float32)
                    cube_quat  = env.data.xquat[env._cube_body_ids[0]]
                    locked_w3  = _w3_from_cube_yaw(cube_quat)
                    state      = "REACH"

            # ── REACH ─────────────────────────────────────────────────────────
            elif state == "REACH":
                reach_steps += 1
                xy_dist = float(np.linalg.norm(ee_pos[:2] - target[:2]))
                z_err   = float(ee_pos[2] - target[2])
                r_obs = np.array([
                    np.clip(xy_dist          / 0.6,       0,  1),
                    np.clip(z_err            / 0.5,      -1,  1),
                    np.clip(joints[1] / np.pi,            -1,  1),
                    np.clip(joints[2] / np.pi,            -1,  1),
                    np.clip(joints[3] / (np.pi / 2),      -1,  1),
                ], dtype=np.float32)
                action, _ = reach_model.predict(r_obs, deterministic=True)
                delta     = np.clip(action, -1, 1)
                q         = joints.astype(np.float64).copy()
                q[0] = locked_pan
                q[1] = float(np.clip(joints[1] + delta[0]*LIFT_SCALE,  *LIFT_LIM))
                q[2] = float(np.clip(joints[2] + delta[1]*ELBOW_SCALE, *ELBOW_LIM))
                q[3] = float(np.clip(joints[3] + delta[2]*W1_SCALE,    *W1_LIM))
                q[4] = _FIXED_W2
                q[5] = locked_w3
                env.data.ctrl[:6] = q
                for _ in range(PHYSICS_STEPS):
                    mujoco.mj_step(env.model, env.data)
                env._viewer.sync()
                dist3d = float(np.linalg.norm(ee_pos - target))
                term   = dist3d < 0.02
                trunc  = reach_steps > 300
                reward = -dist3d
                info   = {"is_success": term, "pan_err_deg": np.degrees(err), "dist": dist3d}
                if term:
                    state     = "GRIP"
                    grip_step = 0

            # ── GRIP ─────────────────────────────────────────────────────────
            elif state == "GRIP":
                grip_step += 1
                joints = get_joints()
                q      = joints.astype(np.float64).copy()
                q[0]   = locked_pan
                q[4]   = _FIXED_W2
                q[5]   = locked_w3

                if grip_step <= GRIP_STEPS:
                    # Gradually close gripper
                    env.data.ctrl[6] = GRIP_MAX * (grip_step / GRIP_STEPS)
                elif grip_step <= GRIP_STEPS + GRIP_HOLD:
                    # Hold closed
                    env.data.ctrl[6] = GRIP_MAX
                else:
                    # Lift slightly to confirm grasp
                    q[1] = float(np.clip(joints[1] + 0.02, *LIFT_LIM))
                    env.data.ctrl[6] = GRIP_MAX

                env.data.ctrl[:6] = q
                for _ in range(PHYSICS_STEPS):
                    mujoco.mj_step(env.model, env.data)
                env._viewer.sync()

                done_grip = grip_step >= GRIP_STEPS + GRIP_HOLD + LIFT_STEPS
                term  = done_grip
                trunc = False
                reward = 0.0
                info  = {"is_success": done_grip, "pan_err_deg": np.degrees(err), "dist": 0.0}

            if state != prev_state:
                print(f"  → {state}  (pan_err={np.degrees(err):.1f}°)")
                prev_state = state

            ep_rew += reward

            elapsed = time.monotonic() - t0
            if (remaining := wall_per_step - elapsed) > 0:
                time.sleep(remaining)

            end_episode = (state == "GRIP" and term) or (state == "REACH" and trunc)
            if end_episode:
                result = "SUCCESS" if info.get("is_success") else "timeout"
                episode += 1
                print(f"Ep {episode:3d} | {result} | pan={pan_steps}s  reach={reach_steps}s | "
                      f"pan_err={info['pan_err_deg']:.1f}°  dist={info.get('dist', 0):.3f}m  rew={ep_rew:.1f}")
                if info.get("is_success"):
                    time.sleep(2.0 / args.speed)
                if args.episodes > 0 and episode >= args.episodes:
                    break
                obs, _ = env.reset()
                env.data.ctrl[6] = 0.0   # open gripper on reset
                ep_rew = 0.0
                state  = "PAN"
                pan_steps = reach_steps = grip_step = 0
                prev_state = None

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
