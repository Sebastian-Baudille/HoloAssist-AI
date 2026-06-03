"""
watch_pan_locked_reach.py — Watch the pan-locked reach policy in isolation.

Usage:
    PYTHONPATH=. python3 watch_pan_locked_reach.py \
        --model ../../rl_models/pan_locked_reach/best/best_model.zip \
        --speed 1.0
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import argparse, time
import mujoco
import numpy as np
from stable_baselines3 import PPO
from ur3e_rl_env.envs.pan_locked_reach_env import UR3ePanLockedReachEnv

SIM_DT     = 10 * 0.002
GRIP_STEPS = 30
GRIP_HOLD  = 20
LIFT_STEPS = 40
GRIP_MAX   = 1.3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True)
    ap.add_argument("--speed",    type=float, default=1.0)
    ap.add_argument("--episodes", type=int,   default=0)
    args = ap.parse_args()

    model = PPO.load(args.model, device="cpu")
    env   = UR3ePanLockedReachEnv(render_mode="human")
    wall_per_step = SIM_DT / args.speed

    episode = 0
    print(f"\nWatching pan-locked reach — {args.speed}x speed\n")

    try:
        while env._viewer is not None and env._viewer.is_running():
            obs, _ = env.reset()
            env.data.ctrl[6] = 0.0
            ep_rew, steps = 0.0, 0
            done = False

            # ── REACH phase ──────────────────────────────────────────────────
            while not done and env._viewer.is_running():
                t0 = time.monotonic()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
                ep_rew += reward
                steps  += 1
                done    = term or trunc
                if info.get("is_success"):  # only pace on success
                    remaining = wall_per_step - (time.monotonic() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

            episode += 1
            if not info.get("is_success"):
                continue

            print(f"Ep {episode:3d} | SUCCESS | {steps} steps | "
                  f"xy={info.get('xy_dist', 0):.3f}m  z={info.get('z_err', 0):+.3f}m  rew={ep_rew:.1f}")

            # ── GRIP phase ───────────────────────────────────────────────────
            if env._viewer.is_running():
                ik    = env._ik_cache
                def get_joints():
                    return np.array([env.data.qpos[a] for a in ik["arm_qpos_addrs"]], dtype=np.float64)

                for grip_step in range(1, GRIP_STEPS + GRIP_HOLD + LIFT_STEPS + 1):
                    t0 = time.monotonic()
                    q  = get_joints()
                    q[0] = env._locked_pan
                    q[4] = -np.pi / 2
                    q[5] = env._locked_w3
                    if grip_step <= GRIP_STEPS:
                        env.data.ctrl[6] = GRIP_MAX * (grip_step / GRIP_STEPS)
                    elif grip_step <= GRIP_STEPS + GRIP_HOLD:
                        env.data.ctrl[6] = GRIP_MAX
                    else:
                        from ur3e_rl_env.envs.pan_locked_reach_env import LIFT_LIM
                        q[1] = float(np.clip(q[1] + 0.02, *LIFT_LIM))
                        env.data.ctrl[6] = GRIP_MAX
                    env.data.ctrl[:6] = q
                    for _ in range(10):
                        mujoco.mj_step(env.model, env.data)
                    env._viewer.sync()
                    remaining = wall_per_step - (time.monotonic() - t0)
                    if remaining > 0:
                        time.sleep(remaining)

                time.sleep(1.0 / args.speed)

            if args.episodes > 0 and episode >= args.episodes:
                break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Done.")


if __name__ == "__main__":
    main()
