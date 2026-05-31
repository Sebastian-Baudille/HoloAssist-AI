from __future__ import annotations

import argparse
import os

from ur3e_rl_env.envs.ur3e_pick_place_env import UR3ePickPlaceEnv


DEFAULT_MODEL_PATH = "./rl_models/ppo_ur3e_reach_object"
DEFAULT_EPISODES = 20


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO policy")
    parser.add_argument("--model",    default=DEFAULT_MODEL_PATH,
                        help="Path to model zip (without .zip extension)")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES,
                        help="Number of evaluation episodes")
    args = parser.parse_args()

    model_path = args.model.removesuffix(".zip")

    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required. Install with: "
            "pip install stable-baselines3 gymnasium"
        ) from exc

    if not os.path.exists(model_path + ".zip") and not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    print(f"Loading model: {model_path}")
    env = UR3ePickPlaceEnv()
    successes = 0
    try:
        if not env.wait_until_ready(timeout_sec=20.0):
            raise RuntimeError("ROS state is not ready. Start the Gazebo/RL topics first.")

        model = PPO.load(model_path, env=env)
        for episode_index in range(args.episodes):
            observation, _ = env.reset()
            terminated = False
            truncated = False
            info = {"is_success": False}

            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)

            success = bool(info.get("is_success", False))
            successes += int(success)
            dist = info.get("distance_to_cube", float("nan"))
            print(
                f"Episode {episode_index + 1:02d}/{args.episodes}: "
                f"{'SUCCESS' if success else 'FAIL   '}"
                f"  dist={dist:.3f} m"
            )

        success_rate = successes / args.episodes
        print(f"\nSuccess rate: {successes}/{args.episodes} ({success_rate:.1%})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
