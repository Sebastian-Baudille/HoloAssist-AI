"""
watch_mujoco.py — Live MuJoCo viewer for the UR3e environment.

Loads the latest checkpoint and renders episodes in real time.
Run this in a separate terminal while training is going.

Usage:
    python3 watch_mujoco.py                      # best/latest checkpoint
    python3 watch_mujoco.py --checkpoint <path>  # specific checkpoint
    python3 watch_mujoco.py --random             # random policy (no model)
    python3 watch_mujoco.py --speed 0.5          # 0.5x real-time (default 0.3x)

Viewer controls:
    Left-drag   — rotate
    Right-drag  — pan
    Scroll      — zoom
    Ctrl+C      — quit
"""
from __future__ import annotations

# Fix MuJoCo viewer scaling on HiDPI Wayland displays (2560×1600 @ 2× content scale).
# Must be set before glfw is imported — forces the X11 GLFW backend which reports 1.0
# content scale and renders full-window instead of bottom-left quarter.
import os as _os, glob as _g
_x11 = next(iter(_g.glob(
    _os.path.expanduser("~/.local/lib/python*/site-packages/glfw/x11/libglfw.so")
)), None)
if _x11:
    _os.environ.setdefault("PYGLFW_LIBRARY", _x11)

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

_WS       = Path(__file__).parent.parent.parent.parent.parent / "ur3e_rl_ws"
MODEL_DIR = _WS / "rl_models"
BEST_DIR  = MODEL_DIR / "mujoco_best"
CKPT_DIR  = MODEL_DIR / "mujoco_checkpoints"

# Real-time physics: 50 physics steps × 0.002 s = 0.1 s per RL step.
# At speed=1.0 we sleep 0.1 s per step (actual real time).
PHYSICS_DT_PER_RL_STEP = 0.10


def _find_model() -> tuple[str, float] | tuple[None, float]:
    """Returns (path, mtime) of the best model, or latest checkpoint as fallback."""
    best = BEST_DIR / "best_model.zip"
    if best.exists():
        return str(best), best.stat().st_mtime
    zips = sorted(glob.glob(str(CKPT_DIR / "*.zip")), key=lambda p: Path(p).stat().st_mtime)
    if zips:
        p = Path(zips[-1])
        return str(p), p.stat().st_mtime
    return None, 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,   default=None)
    parser.add_argument("--random",     action="store_true")
    parser.add_argument("--episodes",   type=int,   default=0, help="0 = forever")
    parser.add_argument("--speed",      type=float, default=0.3,
                        help="Playback speed multiplier (1.0 = real time, 0.3 = 30%%)")
    args = parser.parse_args()

    from ur3e_rl_env.envs.ur3e_mujoco_env import UR3eMuJoCoEnv

    print("Opening MuJoCo viewer window...")
    env = UR3eMuJoCoEnv(render_mode="human")

    model = None
    _last_mtime = 0.0
    if not args.random:
        ckpt, _last_mtime = _find_model()
        if ckpt:
            from stable_baselines3 import PPO
            print(f"Loaded: {Path(ckpt).name}")
            model = PPO.load(ckpt, device="cpu")
        else:
            print("No checkpoint yet — using random policy until training saves one.")

    step_sleep = PHYSICS_DT_PER_RL_STEP / max(args.speed, 0.01)
    print(
        f"\nSpeed: {args.speed}x real-time ({step_sleep*1000:.0f} ms/step)\n"
        "Viewer controls: left-drag=rotate  right-drag=pan  scroll=zoom\n"
        "Press Ctrl+C to quit.\n"
    )

    ep = 0
    try:
        while args.episodes == 0 or ep < args.episodes:
            # Reload best model if training has written a newer version
            if not args.random:
                new_ckpt, new_mtime = _find_model()
                if new_ckpt and new_mtime > _last_mtime:
                    from stable_baselines3 import PPO
                    model = PPO.load(new_ckpt, device="cpu")
                    _last_mtime = new_mtime
                    print(f"[ep {ep}] Loaded best model ({Path(new_ckpt).name}, updated {new_mtime:.0f})")

            obs, reset_info = env.reset()
            ep += 1
            ep_reward = 0.0
            step = 0
            target_cube = reset_info.get("target_cube", "?")

            done = False
            while not done:
                if env._viewer is not None and not env._viewer.is_running():
                    print("\nViewer closed.")
                    return

                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1
                done = terminated or truncated
                time.sleep(step_sleep)

            success   = info.get("is_success", False)
            dist      = info.get("distance_to_cube", float("nan"))
            orient    = info.get("orient_err", float("nan"))
            result    = "SUCCESS" if success else "timeout"
            indicator = "✓" if success else "✗"
            print(
                f"{indicator} ep={ep:4d}  {result:<8s}  "
                f"steps={step:3d}  reward={ep_reward:6.1f}  "
                f"dist={dist*100:.1f}cm  orient={orient:.2f}  "
                f"target={target_cube}"
            )

    except KeyboardInterrupt:
        print("\nStopped.")
        import os; os._exit(0)  # skip cleanup — MuJoCo viewer segfaults on Wayland close


if __name__ == "__main__":
    main()
