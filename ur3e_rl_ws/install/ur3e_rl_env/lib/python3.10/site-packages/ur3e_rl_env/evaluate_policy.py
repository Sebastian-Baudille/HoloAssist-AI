from __future__ import annotations

import os

from ur3e_rl_env.envs.ur3e_pick_place_env import UR3ePickPlaceEnv


MODEL_PATH = "./rl_models/ppo_ur3e_reach_object"
EPISODES = 20


def main() -> None:
    try:
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 is required for evaluation. Install it with: "
            "python3 -m pip install stable-baselines3 gymnasium"
        ) from exc

    if not os.path.exists(MODEL_PATH + ".zip") and not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at {MODEL_PATH}")

    env = UR3ePickPlaceEnv()
    successes = 0
    try:
        if not env.wait_until_ready(timeout_sec=20.0):
            raise RuntimeError("ROS state is not ready. Start the Gazebo/RL topics first.")

        model = PPO.load(MODEL_PATH, env=env)
        for episode_index in range(EPISODES):
            observation, _ = env.reset()
            terminated = False
            truncated = False
            info = {"is_success": False}

            while not terminated and not truncated:
                action, _ = model.predict(observation, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)

            successes += int(bool(info.get("is_success", False)))
            print(
                f"Episode {episode_index + 1:02d}/{EPISODES}: "
                f"success={bool(info.get('is_success', False))}"
            )

        success_rate = successes / EPISODES
        print(f"Success rate: {successes}/{EPISODES} ({success_rate:.1%})")
    finally:
        env.close()


if __name__ == "__main__":
    main()

