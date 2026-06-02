#!/usr/bin/env python3
"""Randomise cube positions in sim_params.yaml.

Places k cubes on the table surface using rejection sampling, enforcing a
minimum centre-to-centre separation so K-Means clustering stays reliable.
Writes the updated config in-place; re-launch the sim to apply.

Usage (from repo root):
    python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py
    python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py -k 3
    python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py --seed 42
    python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py --preview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml


DEFAULT_PARAMS = Path(__file__).parent.parent / "config/sim_params.yaml"

# Fixed colour palette — names stay stable regardless of cube count.
_COLOURS = [
    ("cube_red",    [0.85, 0.10, 0.10, 1.0]),
    ("cube_green",  [0.10, 0.75, 0.10, 1.0]),
    ("cube_blue",   [0.10, 0.20, 0.85, 1.0]),
    ("cube_yellow", [0.95, 0.85, 0.10, 1.0]),
    ("cube_orange", [0.90, 0.50, 0.05, 1.0]),
    ("cube_purple", [0.65, 0.10, 0.80, 1.0]),
]

MAX_TRIES = 20_000


def _place_cubes(
    k: int,
    cube_size: float,
    xy_min: float,
    xy_max: float,
    min_sep: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (k, 2) array of valid [x, y] positions via rejection sampling."""
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < k:
        attempts += 1
        if attempts > MAX_TRIES:
            raise RuntimeError(
                f"Could not place {k} cubes with min_sep={min_sep:.3f} m inside "
                f"[{xy_min}, {xy_max}] in {MAX_TRIES} attempts. "
                f"Try fewer cubes or a smaller --min-sep."
            )
        candidate = rng.uniform(xy_min, xy_max, size=2)
        if all(float(np.linalg.norm(candidate - p)) >= min_sep for p in positions):
            positions.append(candidate)
    return np.array(positions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-k", "--cubes", type=int, default=None,
                        help="Number of cubes to place (default: keep existing count)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument("--min-sep", type=float, default=0.10,
                        help="Minimum centre-to-centre separation in metres (default: 0.10)")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS),
                        help="Path to sim_params.yaml")
    parser.add_argument("--preview", action="store_true",
                        help="Print new positions but do NOT write the file")
    args = parser.parse_args()

    params_path = Path(args.params)
    with open(params_path) as f:
        params = yaml.safe_load(f)

    existing_cubes = params["cubes"]
    k = args.cubes if args.cubes is not None else len(existing_cubes)
    if k < 1 or k > len(_COLOURS):
        sys.exit(f"k must be 1–{len(_COLOURS)}, got {k}")

    cube_size  = float(existing_cubes[0]["size"][0])
    table_half = float(params["table"]["size"][0]) / 2
    table_top_z = float(params["table"]["pose"][2]) + float(params["table"]["size"][2]) / 2
    cube_z     = round(table_top_z + cube_size / 2, 4)

    # Keep cubes away from table edge by at least half their size + 1 cm margin.
    margin = cube_size / 2 + 0.01
    xy_min = round(-(table_half - margin), 4)
    xy_max = round( (table_half - margin), 4)

    min_sep = max(args.min_sep, cube_size * 2.5)  # never less than 2.5× cube size

    rng = np.random.default_rng(args.seed)
    positions = _place_cubes(k, cube_size, xy_min, xy_max, min_sep, rng)

    # Build new cube list
    new_cubes = []
    for i in range(k):
        name, colour = _COLOURS[i]
        x = float(round(positions[i, 0], 4))
        y = float(round(positions[i, 1], 4))
        template = existing_cubes[i] if i < len(existing_cubes) else existing_cubes[0]
        new_cubes.append({
            "name":  name,
            "size":  template["size"],
            "pose":  [x, y, cube_z, 0.0, 0.0, 0.0],
            "color": colour,
            "mass":  float(template["mass"]),
        })

    # Print summary
    print(f"Randomised {k} cube positions  (seed={args.seed}, min_sep={min_sep:.3f} m)")
    print(f"Placement window: x/y ∈ [{xy_min}, {xy_max}] m\n")
    col_w = max(len(c["name"]) for c in new_cubes)
    for c in new_cubes:
        print(f"  {c['name']:{col_w}}  x={c['pose'][0]:+.4f}  y={c['pose'][1]:+.4f}  z={cube_z:.4f}")

    print("\nSeparations:")
    for i in range(k):
        for j in range(i + 1, k):
            pi = np.array(new_cubes[i]["pose"][:2])
            pj = np.array(new_cubes[j]["pose"][:2])
            d = float(np.linalg.norm(pi - pj))
            ok = "✓" if d >= min_sep else "✗"
            print(f"  {new_cubes[i]['name']} ↔ {new_cubes[j]['name']}: {d:.4f} m  {ok}")

    if args.preview:
        print("\n--preview: file not written.")
        return

    params["cubes"] = new_cubes
    # yaml.dump strips comments; that's acceptable for a training workflow.
    with open(params_path, "w") as f:
        yaml.dump(params, f, default_flow_style=None, sort_keys=False, allow_unicode=True)

    print(f"\nWritten → {params_path}")
    print("Re-launch the sim to apply:")
    print("  ros2 launch holoassist_sim sim.launch.py")


if __name__ == "__main__":
    main()
