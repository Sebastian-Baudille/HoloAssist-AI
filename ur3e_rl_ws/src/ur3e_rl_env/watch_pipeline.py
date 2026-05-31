#!/usr/bin/env python3
"""
watch_pipeline.py — Watch the full Reach → Grasp → Transport pipeline.

Usage:
    PYTHONPATH=. python3 watch_pipeline.py \
        --reach     ../../rl_models/reach_best/best_model.zip \
        --transport ../../rl_models/transport_best/best_model.zip

    # Slow motion (easier to follow):
    ... --speed 0.5

Controls: left-drag rotate, scroll zoom, close window to quit.
"""
import sys, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import mujoco.viewer  # must import before coordinator
from ur3e_rl_env.coordinator import MuJoCoCoordinator, Stage

SIM_DT = 50 * 0.002  # 0.1 s sim time per RL step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reach",     required=True, help="Path to reach model .zip")
    ap.add_argument("--transport", required=True, help="Path to transport model .zip")
    ap.add_argument("--episodes",  type=int, default=0, help="Episodes (0=forever)")
    ap.add_argument("--speed",     type=float, default=1.0,
                    help="Playback speed relative to real-time (1=real-time, 0.5=half-speed)")
    ap.add_argument("--seed",      type=int, default=0)
    args = ap.parse_args()

    coord = MuJoCoCoordinator(
        reach_model_path=args.reach,
        transport_model_path=args.transport,
        render_mode="human",
        rng_seed=args.seed,
    )

    wall_per_step = SIM_DT / args.speed
    episode = 0

    print(f"\nPipeline viewer — {args.speed}x speed")
    print("Reach → Grasp → Transport → Release\n")

    try:
        coord.reset()
        while coord.is_running():
            t0    = time.monotonic()
            stage = coord.step()

            elapsed   = time.monotonic() - t0
            remaining = wall_per_step - elapsed
            if remaining > 0:
                time.sleep(remaining)

            if stage == Stage.DONE:
                episode += 1
                print(f"Episode {episode} complete\n")
                if args.episodes > 0 and episode >= args.episodes:
                    break
                coord.reset()

    except KeyboardInterrupt:
        pass
    finally:
        coord.close()
        print("Viewer closed.")


if __name__ == "__main__":
    main()
