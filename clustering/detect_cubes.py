"""
Detect and localise cubes from a HoloAssist-AI point cloud.

Pipeline:
  1. Load PLY file (XYZ + RGB, in camera frame)
  2. Transform points to world frame using camera pose from sim_params.yaml
  3. Crop to cube layer using world Z (table top to cube top)
  4. K-Means clustering on XYZ + RGB
  5. PCA per cluster to get centroid (cube position) and orientation
  6. Print world-frame centroids for DQN use
  7. Visualise in Polyscope

Usage:
  python clustering/detect_cubes.py
  python clustering/detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
  python clustering/detect_cubes.py -k 4 --no-viz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import polyscope as ps
import yaml
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"


# ── 1. Transform from camera body frame to world frame ───────────────────────

def camera_to_world(points, camera_pose):
    x, y, z, roll, pitch, yaw = camera_pose

    # Rotation matrices for each axis
    Rx = np.array([[1, 0,            0           ],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])

    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [ 0,             1, 0            ],
                   [-np.sin(pitch), 0, np.cos(pitch)]])

    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,            1]])

    R = Rz @ Ry @ Rx
    translation = np.array([x, y, z])

    # Apply rotation then translation to every point
    world_points = (R @ points.T).T + translation
    return world_points


# ── 2. Keep only points in the cube layer ─────────────────────────────────────

def crop_cube_layer(points, colors, z_min, z_max):
    mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
    return points[mask], colors[mask]


# ── 3. K-Means clustering on XYZ + RGB ───────────────────────────────────────

def cluster_points(points, colors, k, color_weight):
    # Scale XYZ and RGB independently so neither dominates
    xyz_scaled = StandardScaler().fit_transform(points)
    rgb_scaled = StandardScaler().fit_transform(colors)

    # Combine into one feature vector per point
    features = np.hstack([xyz_scaled, rgb_scaled * color_weight])

    kmeans = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = kmeans.fit_predict(features)
    return labels


# ── 4. PCA per cluster → centroid + orientation ───────────────────────────────

def compute_pca(points, labels, k):
    results = []
    for cluster_id in range(k):
        cluster_pts = points[labels == cluster_id]

        pca = PCA(n_components=3)
        pca.fit(cluster_pts)

        results.append({
            "id":             cluster_id,
            "centroid":       pca.mean_,               # (x, y, z) in world frame
            "axes":           pca.components_,          # principal axes (rows)
            "extents":        np.sqrt(pca.explained_variance_),
            "variance_ratio": pca.explained_variance_ratio_,
            "n_points":       len(cluster_pts),
        })
    return results


# ── 5. Polyscope visualisation ─────────────────────────────────────────────────

def visualise(world_points, colors, cube_points, cube_labels, k, cube_poses):
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    # Full scene with RGB colours
    scene = ps.register_point_cloud("Scene", world_points, radius=0.0003)
    scene.add_color_quantity("RGB", colors, enabled=True)
    scene.set_enabled(False)  # hide by default — enable manually if needed

    # Cube layer coloured by cluster
    if len(cube_points) > 0:
        np.random.seed(0)
        palette      = np.random.rand(k, 3)
        cluster_cols = palette[cube_labels]

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
        centers.add_vector_quantity("PC1", pc1, enabled=True, color=(1, 0, 0), vectortype="ambient")
        centers.add_vector_quantity("PC2", pc2, enabled=True, color=(0, 1, 0), vectortype="ambient")
        centers.add_vector_quantity("PC3", pc3, enabled=True, color=(0, 0, 1), vectortype="ambient")

    ps.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default=None)
    parser.add_argument("-k", "--clusters",     type=int,   default=4)
    parser.add_argument("--color-weight",       type=float, default=0.0)
    parser.add_argument("--params",             default=str(DEFAULT_PARAMS))
    parser.add_argument("--no-viz",             action="store_true")
    args = parser.parse_args()

    # Auto-pick most recent capture, fall back to bundled sample
    if args.path is None:
        captures = sorted((Path.home() / "holoassist_pointclouds").glob("*.ply"))
        if captures:
            args.path = str(captures[-1])
        else:
            args.path = str(Path(__file__).parent / "sample_data/default_4cubes_40mm_v001.ply")
            print("No captures found in ~/holoassist_pointclouds — using bundled sample")
        print(f"Using: {args.path}")

    # Load scene parameters
    with open(args.params) as f:
        params = yaml.safe_load(f)

    camera_pose = params["camera"]["pose"]
    table_top_z = params["table"]["pose"][2] + params["table"]["size"][2] / 2
    cube_height = params["cubes"][0]["size"][2]
    z_min = table_top_z + 0.015   # 3× sensor noise (σ=0.005 m) to exclude table surface
    z_max = table_top_z + cube_height + 0.01
    ground_truth = {c["name"]: c["pose"][:3] for c in params["cubes"]}

    print(f"Table top Z = {table_top_z:.3f} m")
    print(f"Cube layer crop: {z_min:.3f} m → {z_max:.3f} m")

    # Load PLY
    pcd    = o3d.io.read_point_cloud(args.path)
    points = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    print(f"\nLoaded {len(points):,} points")

    # Transform to world frame
    world_points = camera_to_world(points, camera_pose)
    print(f"World Z range: {world_points[:, 2].min():.3f} → {world_points[:, 2].max():.3f} m")

    # Crop to cube layer
    cube_pts, cube_col = crop_cube_layer(world_points, colors, z_min, z_max)
    print(f"Cube layer: {len(cube_pts):,} points")

    # Cluster
    k = args.clusters
    print(f"\nRunning K-Means (k={k}) ...")
    labels = cluster_points(cube_pts, cube_col, k, args.color_weight)
    for i in range(k):
        print(f"  Cluster {i}: {np.sum(labels == i)} points")

    # PCA per cluster
    cube_poses = compute_pca(cube_pts, labels, k)

    # Print results
    print("\n── Estimated cube positions (world frame) ──────────────────────────")
    print(f"  {'Cluster':>7}  {'X':>7}  {'Y':>7}  {'Z':>7}  {'Nearest GT':>18}  {'Error (m)':>10}")
    centroids = np.zeros((k, 3))
    for p in cube_poses:
        cx, cy, cz = p["centroid"]
        centroids[p["id"]] = p["centroid"]
        nearest_name, nearest_pos = min(ground_truth.items(),
                                        key=lambda item: np.linalg.norm(np.array(item[1]) - p["centroid"]))
        error = np.linalg.norm(np.array(nearest_pos) - p["centroid"])
        print(f"  {p['id']:>7}  {cx:>7.4f}  {cy:>7.4f}  {cz:>7.4f}  {nearest_name:>18}  {error:>10.4f}")

    print(f"\nDQN state vector: {centroids.flatten().round(4)}")

    if not args.no_viz:
        visualise(world_points, colors, cube_pts, labels, k, cube_poses)


if __name__ == "__main__":
    main()
