#!/usr/bin/env python3
"""
watch_mujoco.py — Watch the MuJoCo sim run in a 3D viewer.

Usage:
    # Random actions (no model needed):
    python3 watch_mujoco.py

    # Watch a trained policy:
    python3 watch_mujoco.py --model rl_models/mujoco_best/best_model.zip

Controls in the viewer window:
    Left-drag   : rotate
    Right-drag  : pan
    Scroll      : zoom
    Space       : pause/resume
    Backspace   : reset episode
    Ctrl+Q / Esc: quit
"""
import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from ur3e_rl_env.envs.ur3e_mujoco_env import UR3eMuJoCoEnv

REPO = Path(__file__).parent.parent.parent.parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None,
                        help="Path to a .zip SB3 checkpoint to run. Omit for random actions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run (0 = run forever).")
    parser.add_argument("--slow", action="store_true",
                        help="Add a small sleep between steps so motion is easier to follow.")
    args = parser.parse_args()

    policy = None
    if args.model:
        from stable_baselines3 import PPO
        print(f"Loading policy: {args.model}")
        policy = PPO.load(args.model, device="cpu")
        print("Policy loaded.")
    else:
        print("No model specified — running random actions.")

    print("Opening MuJoCo viewer...")
    env = UR3eMuJoCoEnv(render_mode="human")
    obs, _ = env.reset(seed=args.seed)

    episode = 0
    step = 0
    ep_reward = 0.0

    print("Viewer open. Use mouse to rotate/zoom. Close window to quit.")
    print()

    try:
        while True:
            if env._viewer is not None and not env._viewer.is_running():
                break

            if policy is not None:
                action, _ = policy.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            step += 1

            if args.slow:
                time.sleep(0.05)

            if terminated or truncated:
                episode += 1
                result = "SUCCESS" if info.get("is_success") else "timeout"
                print(f"Episode {episode:3d} | steps={step:3d} | "
                      f"reward={ep_reward:7.2f} | {result}")
                obs, _ = env.reset()
                step = 0
                ep_reward = 0.0

                if args.episodes > 0 and episode >= args.episodes:
                    break

    except KeyboardInterrupt:
        pass
    finally:
        env.close()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
