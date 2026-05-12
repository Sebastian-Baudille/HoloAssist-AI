"""Pretrain a PPO policy from teleop demonstrations via behavior cloning.

Loads demo episodes recorded by record_demo, trains the policy network
with supervised learning (MSE on observation -> action), and saves the
result as an SB3 model that can be fine-tuned with train_ppo.

Usage:
    ros2 run ur3e_rl_env pretrain_from_demos

    # Custom settings:
    UR3E_BC_EPOCHS=200 UR3E_BC_LR=5e-4 ros2 run ur3e_rl_env pretrain_from_demos

Environment variables:
    UR3E_DEMO_INPUT_DIR     Demo directory (default: ./demo_data)
    UR3E_BC_OUTPUT_PATH     Output model path (default: ./rl_models/ppo_ur3e_pretrained)
    UR3E_BC_EPOCHS          Training epochs (default: 100)
    UR3E_BC_LR              Learning rate (default: 1e-3)
    UR3E_BC_BATCH_SIZE      Batch size (default: 256)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

DEMO_DIR = os.getenv("UR3E_DEMO_INPUT_DIR", "./demo_data")
OUTPUT_PATH = os.getenv("UR3E_BC_OUTPUT_PATH", "./rl_models/ppo_ur3e_pretrained")
EPOCHS = int(os.getenv("UR3E_BC_EPOCHS", "100"))
LR = float(os.getenv("UR3E_BC_LR", "1e-3"))
BATCH_SIZE = int(os.getenv("UR3E_BC_BATCH_SIZE", "256"))

OBSERVATION_SIZE = 29
ACTION_SIZE = 6
ACTION_SCALE = 0.03


def load_demos(demo_dir: str) -> tuple[np.ndarray, np.ndarray]:
    all_obs: list[np.ndarray] = []
    all_acts: list[np.ndarray] = []
    demo_path = Path(demo_dir)
    files = sorted(demo_path.glob("demo_episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No demo_episode_*.npz files in {demo_dir}")

    for f in files:
        data = np.load(f)
        all_obs.append(data["observations"])
        all_acts.append(data["actions"])
        n = len(data["observations"])
        terminals = data["terminals"].sum() if "terminals" in data else "?"
        print(f"  {f.name}: {n} transitions, {terminals} terminal(s)")

    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    acts = np.concatenate(all_acts, axis=0).astype(np.float32)
    print(f"  Total: {len(obs)} transitions from {len(files)} episodes\n")
    return obs, acts


def main() -> None:
    try:
        import torch
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PyTorch is required for pretraining. "
            "Install with: python3 -m pip install torch"
        ) from exc

    try:
        import gymnasium as gym
        from gymnasium import spaces
        from stable_baselines3 import PPO
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "stable-baselines3 and gymnasium are required. "
            "Install with: python3 -m pip install stable-baselines3 gymnasium"
        ) from exc

    print(f"Loading demos from {DEMO_DIR}...")
    obs_np, acts_np = load_demos(DEMO_DIR)
    acts_np = np.clip(acts_np / ACTION_SCALE, -1.0, 1.0).astype(np.float32)

    class _DummyEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.observation_space = spaces.Box(
                -np.inf, np.inf, (OBSERVATION_SIZE,), np.float32
            )
            self.action_space = spaces.Box(
                -1.0, 1.0, (ACTION_SIZE,), np.float32
            )

        def reset(self, **kw):
            return np.zeros(OBSERVATION_SIZE, dtype=np.float32), {}

        def step(self, action):
            return (
                np.zeros(OBSERVATION_SIZE, dtype=np.float32),
                0.0,
                True,
                False,
                {},
            )

    model = PPO("MlpPolicy", _DummyEnv(), verbose=0, device="cpu")
    policy = model.policy
    policy.set_training_mode(True)

    obs_tensor = torch.from_numpy(obs_np)
    acts_tensor = torch.from_numpy(acts_np)
    dataset = TensorDataset(obs_tensor, acts_tensor)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    print(
        f"Pretraining for {EPOCHS} epochs, "
        f"batch_size={BATCH_SIZE}, lr={LR}, "
        f"transitions={len(obs_np)}"
    )
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        n_batches = 0
        for obs_batch, act_batch in loader:
            features = policy.extract_features(
                obs_batch, policy.features_extractor
            )
            latent_pi, _ = policy.mlp_extractor(features)
            predicted = policy.action_net(latent_pi)

            loss = F.mse_loss(predicted, act_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: loss = {avg_loss:.6f}")

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    model.save(OUTPUT_PATH)
    print(f"\nSaved pretrained model → {OUTPUT_PATH}")
    print(
        f"Fine-tune with PPO:\n"
        f'  model = PPO.load("{OUTPUT_PATH}", env=your_env)\n'
        f"  model.learn(total_timesteps=...)"
    )


if __name__ == "__main__":
    main()
