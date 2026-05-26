"""verify_detection.py — Evaluate cube detection accuracy against the labelled dataset.

Loads every scene_NNNN.ply + scene_NNNN.labels.json from ~/holoassist_dataset/,
runs the K-Means detection pipeline on each, matches detections to ground truth
using the Hungarian algorithm, and reports centroid accuracy statistics.

Run with the clustering venv (Python 3.11):
  cd /path/to/HoloAssist-AI
  source clustering/.venv/bin/activate
  python3 clustering/verify_detection.py
  python3 clustering/verify_detection.py --dataset ~/holoassist_dataset --no-report-file
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DEFAULT_DATASET = Path.home() / "holoassist_dataset"
DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"

Z_MARGIN_BELOW = 0.015   # metres above table to start crop (3× sensor noise)
Z_MARGIN_ABOVE = 0.010   # metres above cube top


# ── Camera → world transform (matches detect_cubes.py exactly) ───────────────

def _camera_to_world(points: np.ndarray, camera_pose: list) -> np.ndarray:
    x, y, z, roll, pitch, yaw = camera_pose
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0            ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,            1]])
    R = Rz @ Ry @ Rx
    return (R @ points.T).T + np.array([x, y, z])


# ── PLY loader ────────────────────────────────────────────────────────────────

def _load_ply(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    """Return (xyz float32 (N,3), rgb uint8 (N,3) or None)."""
    with open(path, "rb") as f:
        # Parse header
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        has_rgb = any("red" in l or "rgb" in l for l in header_lines)
        n_verts = int(next(l for l in header_lines if l.startswith("element vertex")).split()[-1])

        if has_rgb:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

        data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    if has_rgb:
        rgb = np.stack([data["r"], data["g"], data["b"]], axis=1)
    else:
        rgb = None
    return xyz, rgb


# ── Detection pipeline ────────────────────────────────────────────────────────

def detect_centroids(
    ply_path: Path,
    labels: dict,
    camera_pose: list,
) -> np.ndarray:
    """Run K-Means detection pipeline. Returns (k, 3) array of world-frame centroids."""
    xyz_cam, _ = _load_ply(ply_path)

    # Transform to world frame
    xyz_world = _camera_to_world(xyz_cam, camera_pose)

    # Z crop — use actual cube sizes from labels for tight bounds
    table_top = labels["table_top_z"]
    cube_size_max = max((c.get("size_m", 0.04) for c in labels["cubes"]), default=0.04)
    z_min = table_top + Z_MARGIN_BELOW
    z_max = table_top + cube_size_max + Z_MARGIN_ABOVE
    mask = (xyz_world[:, 2] > z_min) & (xyz_world[:, 2] < z_max)
    cropped = xyz_world[mask]

    if len(cropped) < labels["cube_count"] * 10:
        # Not enough points — return zeros
        return np.zeros((labels["cube_count"], 3))

    k = labels["cube_count"]
    xyz_scaled = StandardScaler().fit_transform(cropped)
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    cluster_labels = km.fit_predict(xyz_scaled)

    centroids = np.array([
        cropped[cluster_labels == i].mean(axis=0) for i in range(k)
    ])
    return centroids


# ── Hungarian matching ────────────────────────────────────────────────────────

def match_detections(
    detected: np.ndarray,
    ground_truth: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Match detected centroids to ground truth using Hungarian algorithm.

    Returns (per_pair_errors_m, mean_error_m).
    """
    # Cost matrix: pairwise Euclidean distances
    cost = np.linalg.norm(
        detected[:, None, :] - ground_truth[None, :, :], axis=2
    )
    row_idx, col_idx = linear_sum_assignment(cost)
    errors = cost[row_idx, col_idx]
    return errors, float(errors.mean())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify cube detection against dataset")
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help=f"Dataset directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--params", default=str(DEFAULT_PARAMS),
        help="sim_params.yaml for camera pose",
    )
    parser.add_argument(
        "--no-report-file", action="store_true",
        help="Skip writing accuracy_report.json",
    )
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    if not dataset_dir.exists():
        print(f"ERROR: Dataset directory not found: {dataset_dir}")
        print("Run dataset_capture.py first to generate data.")
        return 1

    with open(args.params) as f:
        params = yaml.safe_load(f)
    camera_pose = params["camera"]["pose"]

    # Collect all scene files
    scene_files = sorted(dataset_dir.glob("scene_*.ply"))
    if not scene_files:
        print(f"No scene_NNNN.ply files found in {dataset_dir}")
        return 1

    print(f"Found {len(scene_files)} scenes in {dataset_dir}")
    print(f"Camera pose: {camera_pose}\n")

    results = {"train": [], "val": []}
    failed = 0

    for ply_path in scene_files:
        labels_path = ply_path.with_suffix(".labels.json")
        if not labels_path.exists():
            print(f"  SKIP {ply_path.name} — no labels file")
            continue

        with open(labels_path) as f:
            labels = json.load(f)

        split = labels.get("split", "train")
        k = labels["cube_count"]
        gt_positions = np.array([
            [c["position"]["x"], c["position"]["y"], c["position"]["z"]]
            for c in labels["cubes"]
        ])

        try:
            detected = detect_centroids(ply_path, labels, camera_pose)
        except Exception as e:
            print(f"  FAIL {ply_path.name}: {e}")
            failed += 1
            continue

        errors, mean_err = match_detections(detected, gt_positions)

        result = {
            "scene_id": labels["scene_id"],
            "split": split,
            "cube_count": k,
            "mean_error_m": round(float(mean_err), 5),
            "per_cube_errors_m": [round(float(e), 5) for e in errors],
            "max_error_m": round(float(errors.max()), 5),
        }
        results[split].append(result)

        status = "OK" if mean_err < 0.03 else "WARN"
        print(
            f"  [{status}] {labels['scene_id']} ({split}, k={k}) "
            f"mean={mean_err*100:.1f} cm  "
            f"max={errors.max()*100:.1f} cm"
        )

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACCURACY REPORT")
    print("=" * 60)

    report = {}
    for split in ("train", "val"):
        split_results = results[split]
        if not split_results:
            print(f"\n{split.upper()}: no scenes")
            continue

        all_errors = [r["mean_error_m"] for r in split_results]
        by_k: dict[int, list] = {}
        for r in split_results:
            by_k.setdefault(r["cube_count"], []).append(r["mean_error_m"])

        mean_e = np.mean(all_errors)
        std_e = np.std(all_errors)
        max_e = np.max(all_errors)
        detection_rate = len(all_errors) / (len(all_errors) + failed)

        print(f"\n{split.upper()} ({len(split_results)} scenes)")
        print(f"  Mean centroid error : {mean_e*100:.2f} cm")
        print(f"  Std dev             : {std_e*100:.2f} cm")
        print(f"  Worst scene         : {max_e*100:.2f} cm")
        print(f"  Detection rate      : {detection_rate*100:.0f}%")
        print(f"  Breakdown by k:")
        for k_val in sorted(by_k):
            k_errors = by_k[k_val]
            print(
                f"    k={k_val}: {len(k_errors)} scenes, "
                f"mean={np.mean(k_errors)*100:.2f} cm"
            )

        report[split] = {
            "scene_count": len(split_results),
            "mean_error_m": round(float(mean_e), 5),
            "std_error_m": round(float(std_e), 5),
            "max_error_m": round(float(max_e), 5),
            "detection_rate": round(detection_rate, 4),
            "by_cube_count": {
                str(k_val): {
                    "count": len(errs),
                    "mean_error_m": round(float(np.mean(errs)), 5),
                }
                for k_val, errs in by_k.items()
            },
            "scenes": split_results,
        }

    if failed:
        print(f"\nFailed scenes: {failed}")

    target_cm = 3.0
    all_means = [r["mean_error_m"] for s in results.values() for r in s]
    if all_means:
        overall_mean = np.mean(all_means) * 100
        verdict = "PASS" if overall_mean < target_cm else "FAIL"
        print(f"\nOverall mean: {overall_mean:.2f} cm  (target < {target_cm} cm) → {verdict}")

    if not args.no_report_file:
        report_path = dataset_dir / "accuracy_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved → {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
