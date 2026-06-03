"""
Detect and localise cubes from a HoloAssist-AI point cloud.

Pipeline:
  1. Load PLY file (XYZ + RGB, in camera frame)
  2. Transform points to world frame using camera pose from sim_params.yaml
  3. Crop to cube layer using world Z (table top to cube top)
  4. Statistical outlier removal — kills flying pixels and depth noise blobs
  5. DBSCAN clustering — finds clusters automatically, no k needed
  6. Size filter — reject clusters too small or large to be a cube
  7. PCA per valid cluster → centroid + orientation axes
  8. Print world-frame centroids for the PPO observation/state contract
  9. Visualise in Polyscope

Usage:
  python clustering/detect_cubes.py
  python clustering/detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
  python clustering/detect_cubes.py --no-viz
  python clustering/detect_cubes.py --eps 0.015 --min-samples 20
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import polyscope as ps
import yaml
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"

# Cluster size bounds — a 4 cm cube at ~0.7 m range produces ~200–900 points.
# Anything outside these bounds is noise or a large non-cube object.
MIN_CLUSTER_PTS = 50
MAX_CLUSTER_PTS = 1500


# ── 1. Transform from camera body frame to world frame ───────────────────────

def camera_to_world(points: np.ndarray, camera_pose: list) -> np.ndarray:
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


# ── 2. Crop to cube layer ─────────────────────────────────────────────────────

def crop_cube_layer(points: np.ndarray, colors: np.ndarray,
                    z_min: float, z_max: float):
    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    return points[mask], colors[mask]


# ── 3. Statistical outlier removal ───────────────────────────────────────────

def remove_outliers(points: np.ndarray, nb_neighbors: int = 20,
                    std_ratio: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    """Remove points whose mean neighbour distance is more than std_ratio
    standard deviations above the global mean. Kills flying pixels and
    depth-discontinuity artefacts. Returns (clean_points, inlier_indices)."""
    if len(points) < nb_neighbors:
        return points, np.arange(len(points))
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    _, inlier_idx = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    inlier_idx = np.asarray(inlier_idx)
    return points[inlier_idx], inlier_idx


# ── 4. DBSCAN clustering ──────────────────────────────────────────────────────

def dbscan_cluster(points: np.ndarray, eps: float = 0.015,
                   min_samples: int = 20) -> np.ndarray:
    """Cluster points with DBSCAN. Returns label array where -1 = noise."""
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(points)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(f"  DBSCAN: {n_clusters} clusters found, {n_noise} noise points discarded")
    return labels


# ── 5. Size filter ────────────────────────────────────────────────────────────

def filter_cube_clusters(labels: np.ndarray,
                         min_pts: int = MIN_CLUSTER_PTS,
                         max_pts: int = MAX_CLUSTER_PTS) -> list[int]:
    """Return cluster IDs whose point count falls within cube-sized bounds."""
    valid = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        n = int(np.sum(labels == cid))
        if min_pts <= n <= max_pts:
            valid.append(cid)
        else:
            print(f"  Cluster {cid}: {n} pts — rejected "
                  f"({'too few' if n < min_pts else 'too many'})")
    return valid


# ── 6. PCA per cluster → centroid + orientation ───────────────────────────────

def compute_pca(points: np.ndarray, labels: np.ndarray,
                valid_ids: list[int]) -> list[dict]:
    results = []
    for i, cid in enumerate(valid_ids):
        cluster_pts = points[labels == cid]
        pca = PCA(n_components=3)
        pca.fit(cluster_pts)
        results.append({
            "id":             i,
            "centroid":       pca.mean_,
            "axes":           pca.components_,
            "extents":        np.sqrt(pca.explained_variance_),
            "variance_ratio": pca.explained_variance_ratio_,
            "n_points":       len(cluster_pts),
        })
    return results


# ── 7. Polyscope visualisation ────────────────────────────────────────────────

def visualise(world_points: np.ndarray, colors: np.ndarray,
              cube_points: np.ndarray, labels: np.ndarray,
              valid_ids: list[int], cube_poses: list[dict]) -> None:
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Full scene
    scene = ps.register_point_cloud("Scene", world_points, radius=0.0003)
    scene.add_color_quantity("RGB", colors, enabled=True)
    scene.set_enabled(False)

    # Cube layer — colour by cluster, grey for noise
    if len(cube_points) > 0:
        np.random.seed(0)
        palette = np.random.rand(max(len(valid_ids), 1), 3)
        cluster_cols = np.full((len(cube_points), 3), 0.4)  # grey default (noise)
        for i, cid in enumerate(valid_ids):
            cluster_cols[labels == cid] = palette[i]
        cubes = ps.register_point_cloud("Cube layer", cube_points, radius=0.001)
        cubes.add_color_quantity("Cluster", cluster_cols, enabled=True)
        cubes.set_point_render_mode("sphere")

    # Centroid markers + PCA axes
    if cube_poses:
        centroids = np.array([p["centroid"] for p in cube_poses])
        pc1 = np.array([p["axes"][0] * p["extents"][0] for p in cube_poses])
        pc2 = np.array([p["axes"][1] * p["extents"][1] for p in cube_poses])
        pc3 = np.array([p["axes"][2] * p["extents"][2] for p in cube_poses])
        centers = ps.register_point_cloud("Centroids", centroids, radius=0.003)
        centers.add_vector_quantity("PC1", pc1, enabled=True,
                                    color=(1, 0, 0), vectortype="ambient")
        centers.add_vector_quantity("PC2", pc2, enabled=True,
                                    color=(0, 1, 0), vectortype="ambient")
        centers.add_vector_quantity("PC3", pc3, enabled=True,
                                    color=(0, 0, 1), vectortype="ambient")

    ps.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Detect cube positions from a HoloAssist point cloud using DBSCAN"
    )
    parser.add_argument("path",          nargs="?",  default=None)
    parser.add_argument("--params",      default=str(DEFAULT_PARAMS))
    parser.add_argument("--eps",         type=float, default=0.015,
                        help="DBSCAN neighbourhood radius in metres (default 0.015)")
    parser.add_argument("--min-samples", type=int,   default=20,
                        help="DBSCAN min points to form a core (default 20)")
    parser.add_argument("--min-points",  type=int,   default=MIN_CLUSTER_PTS,
                        help="Min points for a valid cube cluster")
    parser.add_argument("--max-points",  type=int,   default=MAX_CLUSTER_PTS,
                        help="Max points for a valid cube cluster")
    parser.add_argument("--no-viz",      action="store_true")
    args = parser.parse_args()

    # Auto-pick most recent capture, fall back to bundled sample
    if args.path is None:
        captures = sorted((Path.home() / "holoassist_pointclouds").glob("*.ply"))
        if captures:
            args.path = str(captures[-1])
        else:
            args.path = str(
                Path(__file__).parent / "sample_data/default_4cubes_40mm_v001.ply"
            )
            print("No captures found in ~/holoassist_pointclouds — using bundled sample")
        print(f"Using: {args.path}")

    # Load scene parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)

    camera_pose = params["camera"]["pose"]
    table_top_z = params["table"]["pose"][2] + params["table"]["size"][2] / 2
    cube_height = params["cubes"][0]["size"][2]
    z_min = table_top_z + 0.015
    z_max = table_top_z + cube_height + 0.01
    ground_truth = {c["name"]: c["pose"][:3] for c in params["cubes"]}

    print(f"Table top Z  = {table_top_z:.3f} m")
    print(f"Cube Z crop  : {z_min:.3f} → {z_max:.3f} m")
    print(f"DBSCAN eps={args.eps} m, min_samples={args.min_samples}")

    # Load PLY
    pcd    = o3d.io.read_point_cloud(args.path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    print(f"\nLoaded {len(points):,} points")

    # 1. Transform to world frame
    world_points = camera_to_world(points, camera_pose)
    print(f"World Z range: {world_points[:,2].min():.3f} → {world_points[:,2].max():.3f} m")

    # 2. Crop to cube layer
    cube_pts, cube_col = crop_cube_layer(world_points, colors, z_min, z_max)
    print(f"After Z crop : {len(cube_pts):,} points")

    # 3. Statistical outlier removal
    cube_pts_clean, inlier_idx = remove_outliers(cube_pts)
    cube_col_clean = cube_col[inlier_idx]
    print(f"After outlier removal: {len(cube_pts_clean):,} points "
          f"({len(cube_pts) - len(cube_pts_clean)} removed)")

    # 4. DBSCAN
    print(f"\nRunning DBSCAN ...")
    labels = dbscan_cluster(cube_pts_clean, eps=args.eps, min_samples=args.min_samples)

    # 5. Size filter
    valid_ids = filter_cube_clusters(labels, args.min_points, args.max_points)
    print(f"Valid cube clusters after size filter: {len(valid_ids)}")

    if not valid_ids:
        print("No cubes detected.")
        return

    # 6. PCA per cluster
    cube_poses = compute_pca(cube_pts_clean, labels, valid_ids)

    # 7. Print results
    print("\n── Detected cube positions (world frame) ───────────────────────────")
    print(f"  {'#':>3}  {'X':>7}  {'Y':>7}  {'Z':>7}  {'Points':>7}  "
          f"{'Nearest GT':>18}  {'Error (m)':>10}")
    centroids = np.array([p["centroid"] for p in cube_poses])
    for p in cube_poses:
        cx, cy, cz = p["centroid"]
        nearest_name, nearest_pos = min(
            ground_truth.items(),
            key=lambda item: np.linalg.norm(np.array(item[1]) - p["centroid"])
        )
        error = np.linalg.norm(np.array(nearest_pos) - p["centroid"])
        print(f"  {p['id']:>3}  {cx:>7.4f}  {cy:>7.4f}  {cz:>7.4f}  "
              f"{p['n_points']:>7}  {nearest_name:>18}  {error:>10.4f}")

    print(f"\nDetected {len(cube_poses)} cube(s)")
    print(f"Perception state vector: {centroids.flatten().round(4)}")

    if not args.no_viz:
        visualise(world_points, colors, cube_pts_clean, labels, valid_ids, cube_poses)


if __name__ == "__main__":
    main()
