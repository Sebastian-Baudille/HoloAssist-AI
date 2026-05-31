#!/usr/bin/env python3
"""
watch_reach.py — Watch the reach model move the arm toward cubes.

Usage:
    PYTHONPATH=. python3 watch_reach.py \
        --model ../../rl_models/reach_best/best_model.zip

    # Slow motion:
    ... --speed 0.5

Controls: left-drag rotate, scroll zoom, close window to quit.
"""
import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco.viewer
from stable_baselines3 import PPO
from ur3e_rl_env.envs.reach_env import UR3eReachEnv

SIM_DT = 50 * 0.002  # 0.1 s sim time per RL step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    required=True, help="Path to reach model .zip")
    ap.add_argument("--episodes", type=int,   default=0,   help="Episodes (0=forever)")
    ap.add_argument("--speed",    type=float, default=1.0, help="Playback speed (1=real-time)")
    ap.add_argument("--seed",     type=int,   default=0)
    args = ap.parse_args()

    print(f"Loading: {args.model}")
    model = PPO.load(args.model, device="cpu")

    env     = UR3eReachEnv(render_mode="human")
    obs, _  = env.reset(seed=args.seed)
    episode = 0
    step    = 0
    ep_rew  = 0.0
    wall_per_step = SIM_DT / args.speed

    print(f"\nReach viewer — {args.speed}x speed  (arm → cube)")
    print("Close window to quit.\n")

    try:
        while env._viewer is not None and env._viewer.is_running():
            t0 = time.monotonic()

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_rew += reward
            step   += 1

            elapsed = time.monotonic() - t0
            if (remaining := wall_per_step - elapsed) > 0:
                time.sleep(remaining)

            if terminated or truncated:
                episode += 1
                result = "SUCCESS" if info["is_success"] else "timeout"
                print(f"Episode {episode:3d} | steps={step:3d} | "
                      f"reward={ep_rew:7.2f} | dist={info['dist_to_cube']:.3f}m | {result}")
                if args.episodes > 0 and episode >= args.episodes:
                    break
                obs, _ = env.reset()
                step   = 0
                ep_rew = 0.0

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
