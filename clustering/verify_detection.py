"""verify_detection.py — Evaluate cube detection accuracy against the labelled dataset.

Loads every scene_NNNN.ply + scene_NNNN.labels.json from ~/holoassist_dataset/,
runs the DBSCAN detection pipeline on each (matching detect_cubes.py exactly),
matches detections to ground truth using the Hungarian algorithm, and reports
centroid accuracy statistics.

Because DBSCAN finds clusters automatically (no fixed k), the detected count
may differ from the ground-truth cube count. The matching handles this:
  - Missed cubes (detected < GT): a configurable miss_penalty distance is used
    for unmatched ground-truth cubes so the mean error reflects misses honestly.
  - Extra detections (detected > GT): extra clusters are unmatched false positives
    and are counted but don't inflate the error metric.

Run with the clustering venv (Python 3.11):
  cd /path/to/HoloAssist-AI
  source clustering/.venv/bin/activate
  python3 clustering/verify_detection.py
  python3 clustering/verify_detection.py --dataset ~/holoassist_dataset --no-report-file
  python3 clustering/verify_detection.py --eps 0.015 --min-samples 20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import yaml
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN

DEFAULT_DATASET = Path.home() / "holoassist_dataset"
DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"

Z_MARGIN_BELOW = 0.015   # metres above table to start crop (3× sensor noise)
Z_MARGIN_ABOVE = 0.010   # metres above cube top

# DBSCAN / size-filter defaults — must match detect_cubes.py
DBSCAN_EPS         = 0.015
DBSCAN_MIN_SAMPLES = 20
MIN_CLUSTER_PTS    = 50
MAX_CLUSTER_PTS    = 1500

# Distance assigned to a missed cube so single-cube misses register as a large error
MISS_PENALTY_M = 0.50


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
    """Return (xyz float32 (N,3), rgb float32 (N,3) or None)."""
    with open(path, "rb") as f:
        header_lines = []
        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        has_rgb = any("red" in l or "rgb" in l for l in header_lines)
        n_verts = int(
            next(l for l in header_lines if l.startswith("element vertex")).split()[-1]
        )

        if has_rgb:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4")])

        data = np.frombuffer(f.read(n_verts * dtype.itemsize), dtype=dtype)

    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float32)
    rgb = (np.stack([data["r"], data["g"], data["b"]], axis=1).astype(np.float32) / 255.0
           if has_rgb else None)
    return xyz, rgb


# ── DBSCAN detection pipeline (mirrors detect_cubes.py) ──────────────────────

def _remove_outliers(points: np.ndarray,
                     nb_neighbors: int = 20,
                     std_ratio: float = 2.0) -> np.ndarray:
    """Statistical outlier removal — kills flying pixels and depth noise."""
    if len(points) < nb_neighbors:
        return points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    return points[np.asarray(inlier_idx)]


def detect_centroids(
    ply_path: Path,
    labels: dict,
    camera_pose: list,
    eps: float = DBSCAN_EPS,
    min_samples: int = DBSCAN_MIN_SAMPLES,
    min_pts: int = MIN_CLUSTER_PTS,
    max_pts: int = MAX_CLUSTER_PTS,
) -> np.ndarray:
    """Run the DBSCAN detection pipeline.

    Returns an (m, 3) array of world-frame centroids where m may differ from
    labels['cube_count'] — DBSCAN finds however many clusters are present.
    Returns an empty (0, 3) array if no valid clusters are found.
    """
    xyz_cam, _ = _load_ply(ply_path)

    # 1. Transform to world frame
    xyz_world = _camera_to_world(xyz_cam, camera_pose)

    # 2. Z crop — use actual cube sizes from labels for tight bounds
    table_top = labels["table_top_z"]
    cube_size_max = max((c.get("size_m", 0.04) for c in labels["cubes"]), default=0.04)
    z_min = table_top + Z_MARGIN_BELOW
    z_max = table_top + cube_size_max + Z_MARGIN_ABOVE
    mask = (xyz_world[:, 2] > z_min) & (xyz_world[:, 2] < z_max)
    cropped = xyz_world[mask]

    if len(cropped) < min_pts:
        return np.empty((0, 3), dtype=np.float32)

    # 3. Statistical outlier removal
    cropped = _remove_outliers(cropped)
    if len(cropped) < min_pts:
        return np.empty((0, 3), dtype=np.float32)

    # 4. DBSCAN
    cluster_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(cropped)

    # 5. Size filter
    valid_ids = []
    for cid in sorted(set(cluster_labels)):
        if cid == -1:
            continue
        n = int(np.sum(cluster_labels == cid))
        if min_pts <= n <= max_pts:
            valid_ids.append(cid)

    if not valid_ids:
        return np.empty((0, 3), dtype=np.float32)

    centroids = np.array([
        cropped[cluster_labels == cid].mean(axis=0) for cid in valid_ids
    ])
    return centroids


# ── Hungarian matching — handles variable detected count ──────────────────────

def match_detections(
    detected: np.ndarray,
    ground_truth: np.ndarray,
    miss_penalty: float = MISS_PENALTY_M,
) -> tuple[np.ndarray, float, int, int]:
    """Match detected centroids to ground truth using Hungarian algorithm.

    Handles the case where detected count != ground-truth count:
      - Missed GT cubes get an error of miss_penalty so they are counted honestly.
      - Extra (false-positive) detections are counted but don't affect the error.

    Returns:
        per_gt_errors  — length-k array, one error per GT cube (miss_penalty for misses)
        mean_error_m   — mean over all GT cubes including misses
        n_matched      — number of GT cubes successfully matched
        n_fp           — number of false-positive detections
    """
    k_gt = len(ground_truth)

    if len(detected) == 0:
        return (
            np.full(k_gt, miss_penalty),
            float(miss_penalty),
            0,
            0,
        )

    # Cost matrix shape: (n_detected, k_gt)
    cost = np.linalg.norm(
        detected[:, None, :] - ground_truth[None, :, :], axis=2
    )  # (n_det, k_gt)

    # linear_sum_assignment minimises over min(n_det, k_gt) pairs
    row_idx, col_idx = linear_sum_assignment(cost)
    # row_idx: detection indices matched; col_idx: GT indices matched

    matched_errors = cost[row_idx, col_idx]
    n_matched = len(row_idx)
    n_fp = max(0, len(detected) - n_matched)

    # Build per-GT error array; unmatched GT cubes get miss_penalty
    per_gt = np.full(k_gt, miss_penalty)
    per_gt[col_idx] = matched_errors

    return per_gt, float(per_gt.mean()), n_matched, n_fp


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify cube detection accuracy against dataset (DBSCAN pipeline)"
    )
    parser.add_argument(
        "--dataset", default=str(DEFAULT_DATASET),
        help=f"Dataset directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument(
        "--params", default=str(DEFAULT_PARAMS),
        help="sim_params.yaml for camera pose",
    )
    parser.add_argument(
        "--eps", type=float, default=DBSCAN_EPS,
        help=f"DBSCAN neighbourhood radius in metres (default {DBSCAN_EPS})",
    )
    parser.add_argument(
        "--min-samples", type=int, default=DBSCAN_MIN_SAMPLES,
        help=f"DBSCAN min points to form a core (default {DBSCAN_MIN_SAMPLES})",
    )
    parser.add_argument(
        "--min-points", type=int, default=MIN_CLUSTER_PTS,
        help=f"Min points for a valid cube cluster (default {MIN_CLUSTER_PTS})",
    )
    parser.add_argument(
        "--max-points", type=int, default=MAX_CLUSTER_PTS,
        help=f"Max points for a valid cube cluster (default {MAX_CLUSTER_PTS})",
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

    scene_files = sorted(dataset_dir.glob("scene_*.ply"))
    if not scene_files:
        print(f"No scene_NNNN.ply files found in {dataset_dir}")
        return 1

    print(f"Found {len(scene_files)} scenes in {dataset_dir}")
    print(f"Camera pose : {camera_pose}")
    print(f"DBSCAN eps={args.eps} m  min_samples={args.min_samples}")
    print(f"Cluster pts : {args.min_points}–{args.max_points}\n")

    results: dict[str, list] = {"train": [], "val": []}
    failed = 0
    total_gt_cubes = 0
    total_matched  = 0
    total_fp       = 0

    for ply_path in scene_files:
        labels_path = ply_path.with_suffix(".labels.json")
        if not labels_path.exists():
            print(f"  SKIP {ply_path.name} — no labels file")
            continue

        with open(labels_path) as f:
            labels = json.load(f)

        split  = labels.get("split", "train")
        k_gt   = labels["cube_count"]
        gt_pos = np.array([
            [c["position"]["x"], c["position"]["y"], c["position"]["z"]]
            for c in labels["cubes"]
        ])

        try:
            detected = detect_centroids(
                ply_path, labels, camera_pose,
                eps=args.eps, min_samples=args.min_samples,
                min_pts=args.min_points, max_pts=args.max_points,
            )
        except Exception as e:
            print(f"  FAIL {ply_path.name}: {e}")
            failed += 1
            continue

        per_gt_errors, mean_err, n_matched, n_fp = match_detections(detected, gt_pos)

        total_gt_cubes += k_gt
        total_matched  += n_matched
        total_fp       += n_fp

        count_ok = (len(detected) == k_gt)
        status = "OK" if (mean_err < 0.03 and count_ok) else "WARN"
        print(
            f"  [{status}] {labels['scene_id']}  ({split}, GT={k_gt}, det={len(detected)})  "
            f"mean={mean_err*100:.1f} cm  max={per_gt_errors.max()*100:.1f} cm"
            + (f"  FP={n_fp}" if n_fp else "")
        )

        results[split].append({
            "scene_id":         labels["scene_id"],
            "split":            split,
            "cube_count":       k_gt,
            "detected_count":   int(len(detected)),
            "n_matched":        int(n_matched),
            "n_fp":             int(n_fp),
            "mean_error_m":     round(float(mean_err), 5),
            "per_cube_errors_m":[round(float(e), 5) for e in per_gt_errors],
            "max_error_m":      round(float(per_gt_errors.max()), 5),
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("ACCURACY REPORT  (DBSCAN pipeline)")
    print("=" * 60)

    report: dict = {}
    for split in ("train", "val"):
        split_results = results[split]
        if not split_results:
            print(f"\n{split.upper()}: no scenes")
            continue

        all_errors   = [r["mean_error_m"] for r in split_results]
        n_scenes     = len(split_results)
        perfect      = sum(1 for r in split_results if r["detected_count"] == r["cube_count"])
        split_gt     = sum(r["cube_count"]       for r in split_results)
        split_match  = sum(r["n_matched"]        for r in split_results)
        split_fp     = sum(r["n_fp"]             for r in split_results)
        cube_recall  = split_match / split_gt if split_gt else 0.0

        by_k: dict[int, list] = {}
        for r in split_results:
            by_k.setdefault(r["cube_count"], []).append(r["mean_error_m"])

        mean_e = float(np.mean(all_errors))
        std_e  = float(np.std(all_errors))
        max_e  = float(np.max(all_errors))

        print(f"\n{split.upper()} ({n_scenes} scenes)")
        print(f"  Mean centroid error : {mean_e*100:.2f} cm")
        print(f"  Std dev             : {std_e*100:.2f} cm")
        print(f"  Worst scene         : {max_e*100:.2f} cm")
        print(f"  Exact count correct : {perfect}/{n_scenes}  ({100*perfect/n_scenes:.0f}%)")
        print(f"  Cube recall         : {split_match}/{split_gt}  ({100*cube_recall:.0f}%)")
        if split_fp:
            print(f"  False positives     : {split_fp}")
        print(f"  Breakdown by GT k:")
        for k_val in sorted(by_k):
            k_errors = by_k[k_val]
            print(
                f"    k={k_val}: {len(k_errors)} scenes, "
                f"mean={np.mean(k_errors)*100:.2f} cm"
            )

        report[split] = {
            "scene_count":    n_scenes,
            "mean_error_m":   round(mean_e, 5),
            "std_error_m":    round(std_e, 5),
            "max_error_m":    round(max_e, 5),
            "exact_count_rate": round(perfect / n_scenes, 4),
            "cube_recall":    round(cube_recall, 4),
            "false_positives": split_fp,
            "by_cube_count":  {
                str(k_val): {
                    "count":        len(errs),
                    "mean_error_m": round(float(np.mean(errs)), 5),
                }
                for k_val, errs in by_k.items()
            },
            "scenes": split_results,
        }

    if failed:
        print(f"\nFailed scenes: {failed}")

    print(f"\nOverall cube recall: {total_matched}/{total_gt_cubes}  "
          f"({100*total_matched/total_gt_cubes:.0f}%)"
          if total_gt_cubes else "")
    if total_fp:
        print(f"Total false positives: {total_fp}")

    target_cm = 3.0
    all_means = [r["mean_error_m"] for s in results.values() for r in s]
    if all_means:
        overall_mean = float(np.mean(all_means)) * 100
        verdict = "PASS" if overall_mean < target_cm else "FAIL"
        print(f"\nOverall mean error: {overall_mean:.2f} cm  "
              f"(target < {target_cm} cm) → {verdict}")

    if not args.no_report_file:
        report_path = dataset_dir / "accuracy_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nFull report saved → {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
