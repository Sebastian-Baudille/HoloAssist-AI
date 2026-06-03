"""
analyse_scene.py — Point cloud analysis: Polyscope visualisation + confusion matrix.

Usage:
  python3 clustering/analyse_scene.py
  python3 clustering/analyse_scene.py ~/holoassist_pointclouds/default_4cubes_40mm_v005.ply
  python3 clustering/analyse_scene.py <ply> --confusion   # show confusion matrix first
  python3 clustering/analyse_scene.py <ply> --no-viz      # confusion matrix only
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import polyscope as ps
import yaml
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"
MIN_CLUSTER_PTS = 50
MAX_CLUSTER_PTS = 2000

# Distinct colours for up to 6 cube clusters
CUBE_COLOURS = [
    [0.95, 0.25, 0.25],   # red
    [0.25, 0.75, 0.25],   # green
    [0.25, 0.45, 0.95],   # blue
    [0.95, 0.85, 0.10],   # yellow
    [0.95, 0.55, 0.10],   # orange
    [0.75, 0.25, 0.90],   # purple
]


# ── transform ─────────────────────────────────────────────────────────────────

def camera_to_world(points: np.ndarray, pose: list) -> np.ndarray:
    x, y, z, roll, pitch, yaw = pose
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,              1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,            0,            1]])
    return (Rz @ Ry @ Rx @ points.T).T + np.array([x, y, z])


# ── helpers ───────────────────────────────────────────────────────────────────

def _solid(n: int, colour: list) -> np.ndarray:
    return np.tile(colour, (n, 1)).astype(np.float32)


def _height_colour(pts: np.ndarray) -> np.ndarray:
    z = pts[:, 2]
    t = np.clip((z - z.min()) / max(z.max() - z.min(), 1e-6), 0, 1)
    # cold (blue) → warm (red) heat map
    r = t
    g = 1 - np.abs(t - 0.5) * 2
    b = 1 - t
    return np.stack([r, g, b], axis=1).astype(np.float32)


# ── confusion matrix ──────────────────────────────────────────────────────────

def _show_confusion_matrix(world_points, cube_mask, clean_pts, labels,
                            valid_ids, gt_positions, cube_size):
    """
    Binary confusion matrix evaluated only within the Z-crop (detectable zone).

    We restrict evaluation to points that passed the Z-crop (cube_mask=True)
    so that FN only counts things the algorithm could have detected but missed —
    not cube-body points that were intentionally excluded below z_min.

    Ground truth: a Z-crop point is 'actual cube' if within the cube bounding
    sphere (cube_size * sqrt(3)/2 + 5 mm margin) of any GT centroid.
    Predicted:    a Z-crop point is 'predicted cube' if in a valid DBSCAN cluster.
    """
    from sklearn.neighbors import KDTree

    gt_radius = cube_size * (3 ** 0.5) / 2 + 0.005   # bounding sphere + 5 mm margin

    # Restrict to detectable zone (Z-crop points only)
    zone_pts = world_points[cube_mask]

    # ── Ground truth for Z-crop points ────────────────────────────────────────
    gt_tree  = KDTree(gt_positions)
    gt_dists, _ = gt_tree.query(zone_pts, k=1)
    actual_cube  = gt_dists[:, 0] < gt_radius

    # ── Predicted for Z-crop points ───────────────────────────────────────────
    predicted_cube = np.zeros(len(zone_pts), dtype=bool)
    if valid_ids:
        all_cluster_pts = np.vstack([clean_pts[labels == cid] for cid in valid_ids])
        tree = KDTree(all_cluster_pts)
        snap = 0.003  # 3 mm snap radius
        dists, _ = tree.query(zone_pts, k=1)
        predicted_cube = dists[:, 0] < snap

    # ── Build 2×2 matrix ──────────────────────────────────────────────────────
    TP = int(np.sum( actual_cube &  predicted_cube))
    FP = int(np.sum(~actual_cube &  predicted_cube))
    FN = int(np.sum( actual_cube & ~predicted_cube))
    TN = int(np.sum(~actual_cube & ~predicted_cube))
    total = TP + FP + FN + TN

    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (TP + TN) / total if total else 0.0

    print("\n── Confusion matrix ───────────────────────────────────────────────────")
    print(f"  GT radius used    : {gt_radius*100:.1f} cm per cube")
    print(f"  TP {TP:6d}   FP {FP:6d}")
    print(f"  FN {FN:6d}   TN {TN:6d}")
    print(f"  Precision : {precision:.4f}   Recall : {recall:.4f}")
    print(f"  F1        : {f1:.4f}   Accuracy: {accuracy:.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    matrix = np.array([[TP, FN],
                       [FP, TN]], dtype=float)
    labels_txt = np.array([[f"TP\n{TP:,}\n({100*TP/total:.1f}%)",
                             f"FN\n{FN:,}\n({100*FN/total:.1f}%)"],
                            [f"FP\n{FP:,}\n({100*FP/total:.1f}%)",
                             f"TN\n{TN:,}\n({100*TN/total:.1f}%)"]])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5),
                             gridspec_kw={"width_ratios": [2, 1]})
    fig.patch.set_facecolor("#1a1a2e")

    # Left — heatmap
    ax = axes[0]
    ax.set_facecolor("#1a1a2e")
    norm_matrix = matrix / matrix.sum()
    im = ax.imshow(norm_matrix, cmap="Blues", vmin=0, vmax=1)

    for r in range(2):
        for c in range(2):
            ax.text(c, r, labels_txt[r, c], ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if norm_matrix[r, c] > 0.4 else "#cccccc")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted: Cube", "Predicted: Not-cube"],
                       color="white", fontsize=11)
    ax.set_yticklabels(["Actual: Cube", "Actual: Not-cube"],
                       color="white", fontsize=11)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444444")
    ax.set_title("Point-level Confusion Matrix", color="white", fontsize=13, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    # Right — metrics bar chart
    ax2 = axes[1]
    ax2.set_facecolor("#1a1a2e")
    metrics      = [precision, recall, f1, accuracy]
    metric_names = ["Precision", "Recall", "F1", "Accuracy"]
    colours      = ["#4fc3f7", "#81c784", "#ffb74d", "#ce93d8"]
    bars = ax2.barh(metric_names, metrics, color=colours, height=0.5)
    ax2.set_xlim(0, 1.15)
    ax2.set_facecolor("#1a1a2e")
    ax2.tick_params(colors="white")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for spine in ["bottom", "left"]:
        ax2.spines[spine].set_edgecolor("#444444")
    ax2.xaxis.label.set_color("white")
    ax2.yaxis.label.set_color("white")
    for bar, val in zip(bars, metrics):
        ax2.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{val:.4f}", va="center", color="white", fontsize=12,
                 fontweight="bold")
    ax2.set_title("Detection Metrics", color="white", fontsize=13, pad=12)
    plt.setp(ax2.get_xticklabels(), color="white")
    plt.setp(ax2.get_yticklabels(), color="white", fontsize=11)

    plt.tight_layout()
    plt.show()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path",    nargs="?", default=None)
    parser.add_argument("--params", default=str(DEFAULT_PARAMS))
    parser.add_argument("--eps",         type=float, default=0.015)
    parser.add_argument("--min-samples", type=int,   default=20)
    parser.add_argument("--confusion",   action="store_true",
                        help="Show confusion matrix before Polyscope")
    parser.add_argument("--no-viz",      action="store_true",
                        help="Show confusion matrix only, skip Polyscope")
    args = parser.parse_args()
    if args.no_viz:
        args.confusion = True

    # ── auto-pick PLY ────────────────────────────────────────────────────────
    if args.path is None:
        plys = sorted((Path.home() / "holoassist_pointclouds").glob("*.ply"))
        if plys:
            args.path = str(plys[-1])
        else:
            args.path = str(Path(__file__).parent / "sample_data/default_4cubes_40mm_v001.ply")
            print("No captures found — using bundled sample")
        print(f"Using: {args.path}")

    # ── load params ──────────────────────────────────────────────────────────
    with open(args.params) as f:
        params = yaml.safe_load(f)

    camera_pose  = params["camera"]["pose"]
    table_top_z  = params["table"]["pose"][2] + params["table"]["size"][2] / 2
    cube_height  = params["cubes"][0]["size"][2]
    z_min        = table_top_z + 0.015
    z_max        = table_top_z + cube_height + 0.010
    gt_positions = np.array([c["pose"][:3] for c in params["cubes"]])

    print(f"\nCamera pose  : {camera_pose}")
    print(f"Table top Z  : {table_top_z:.3f} m")
    print(f"Cube Z crop  : {z_min:.3f} → {z_max:.3f} m")
    print(f"DBSCAN       : eps={args.eps} m  min_samples={args.min_samples}")
    print(f"GT cubes     : {len(gt_positions)}")

    # ── load + transform ─────────────────────────────────────────────────────
    pcd    = o3d.io.read_point_cloud(args.path)
    pts    = np.asarray(pcd.points, dtype=np.float32)
    colors = np.asarray(pcd.colors, dtype=np.float32)
    if colors.shape[0] != pts.shape[0]:
        colors = np.ones((len(pts), 3), dtype=np.float32) * 0.6

    world = camera_to_world(pts, camera_pose)
    print(f"\nLoaded        {len(world):,} points")
    print(f"World Z range : {world[:,2].min():.3f} → {world[:,2].max():.3f} m")

    # ── Z-crop ───────────────────────────────────────────────────────────────
    cube_mask   = (world[:, 2] > z_min) & (world[:, 2] < z_max)
    table_mask  = ~cube_mask
    cube_layer  = world[cube_mask]
    cube_colors = colors[cube_mask]
    table_pts   = world[table_mask]
    print(f"Cube layer    : {len(cube_layer):,} points  (table/bg: {len(table_pts):,})")

    # ── outlier removal ───────────────────────────────────────────────────────
    pcd_layer = o3d.geometry.PointCloud()
    pcd_layer.points = o3d.utility.Vector3dVector(cube_layer)
    _, inlier_idx = pcd_layer.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    inlier_idx   = np.asarray(inlier_idx)
    outlier_mask = np.ones(len(cube_layer), dtype=bool)
    outlier_mask[inlier_idx] = False
    outlier_pts  = cube_layer[outlier_mask]
    clean_pts    = cube_layer[inlier_idx]
    clean_colors = cube_colors[inlier_idx]
    print(f"After outlier removal: {len(clean_pts):,}  ({len(outlier_pts)} removed)")

    # ── DBSCAN ───────────────────────────────────────────────────────────────
    labels     = DBSCAN(eps=args.eps, min_samples=args.min_samples).fit_predict(clean_pts)
    all_ids    = sorted(set(labels) - {-1})
    noise_pts  = clean_pts[labels == -1]
    print(f"DBSCAN        : {len(all_ids)} clusters, {len(noise_pts)} noise pts")

    # ── size filter ───────────────────────────────────────────────────────────
    valid_ids    = [c for c in all_ids if MIN_CLUSTER_PTS <= np.sum(labels==c) <= MAX_CLUSTER_PTS]
    rejected_ids = [c for c in all_ids if c not in valid_ids]
    rejected_pts = np.vstack([clean_pts[labels==c] for c in rejected_ids]) if rejected_ids else np.empty((0,3))
    print(f"Valid clusters: {len(valid_ids)}   Rejected: {len(rejected_ids)}")

    # ── centroids + comparison ────────────────────────────────────────────────
    centroids = []
    print("\n── Detection results ──────────────────────────────────────────────────")
    print(f"  {'Cube':>4}  {'X':>7}  {'Y':>7}  {'Z':>7}  {'Points':>7}  {'Nearest GT':>12}  {'Error':>8}")
    for i, cid in enumerate(valid_ids):
        c_pts = clean_pts[labels == cid]
        cen   = c_pts.mean(axis=0)
        centroids.append(cen)
        errors = np.linalg.norm(gt_positions - cen, axis=1)
        nearest_idx = errors.argmin()
        nearest_gt  = params["cubes"][nearest_idx]["name"]
        print(f"  {i:>4}  {cen[0]:>7.4f}  {cen[1]:>7.4f}  {cen[2]:>7.4f}  "
              f"{len(c_pts):>7}  {nearest_gt:>12}  {errors[nearest_idx]:>7.4f} m")

    centroids = np.array(centroids) if centroids else np.empty((0, 3))

    if len(centroids):
        mean_err = np.mean([
            np.linalg.norm(gt_positions - c, axis=1).min() for c in centroids
        ])
        print(f"\nMean centroid error: {mean_err*100:.2f} cm")
        print(f"DQN state vector: {centroids.flatten().round(4)}")

    # ── Confusion matrix ─────────────────────────────────────────────────────
    if args.confusion:
        _show_confusion_matrix(
            world_points   = world,
            cube_mask      = cube_mask,
            clean_pts      = clean_pts,
            labels         = labels,
            valid_ids      = valid_ids,
            gt_positions   = gt_positions,
            cube_size      = cube_height,
        )

    if args.no_viz:
        return

    # ── Polyscope ─────────────────────────────────────────────────────────────
    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")
    ps.set_background_color((0.12, 0.12, 0.12))

    # ── Main view: non-cube (grey) + cube (cyan) ──────────────────────────────

    # All non-cube points — grey, small
    ps.register_point_cloud("Non-cube points", world[~cube_mask], radius=0.00015) \
      .add_color_quantity("col", _solid(int((~cube_mask).sum()), [0.40, 0.40, 0.40]),
                          enabled=True)

    # All valid cube points — single cyan, slightly larger
    if len(valid_ids):
        all_cube_pts = np.vstack([clean_pts[labels == cid] for cid in valid_ids])
        ps.register_point_cloud("Cube points (detected)", all_cube_pts, radius=0.0008) \
          .add_color_quantity("col", _solid(len(all_cube_pts), [0.20, 0.85, 0.95]),
                              enabled=True)

    # ── Centroid anchors + 3 PCA axis arrows ─────────────────────────────────
    if len(centroids):
        pc1_vecs, pc2_vecs, pc3_vecs = [], [], []
        for cid in valid_ids:
            c_pts = clean_pts[labels == cid]
            pca   = PCA(n_components=3).fit(c_pts)
            scale = np.sqrt(pca.explained_variance_)
            pc1_vecs.append(pca.components_[0] * scale[0])
            pc2_vecs.append(pca.components_[1] * scale[1])
            pc3_vecs.append(pca.components_[2] * scale[2])

        cen_cloud = ps.register_point_cloud("Centroids", centroids, radius=0.0025)
        cen_cloud.add_color_quantity("col", _solid(len(centroids), [1.0, 1.0, 1.0]),
                                     enabled=True)
        cen_cloud.add_vector_quantity("PC1 (longest axis)", np.array(pc1_vecs),
                                      enabled=True, color=(1.0, 0.25, 0.25),
                                      vectortype="ambient", radius=0.003)
        cen_cloud.add_vector_quantity("PC2", np.array(pc2_vecs),
                                      enabled=True, color=(0.25, 1.0, 0.25),
                                      vectortype="ambient", radius=0.003)
        cen_cloud.add_vector_quantity("PC3 (normal)", np.array(pc3_vecs),
                                      enabled=True, color=(0.25, 0.50, 1.0),
                                      vectortype="ambient", radius=0.003)

    # ── GT positions — small white markers ───────────────────────────────────
    gt_cloud = ps.register_point_cloud("GT positions", gt_positions, radius=0.006)
    gt_cloud.add_color_quantity("col", _solid(len(gt_positions), [1.0, 1.0, 1.0]),
                                 enabled=True)
    gt_cloud.set_point_render_mode("sphere")

    # ── Error lines: centroid → nearest GT ───────────────────────────────────
    if len(centroids):
        nodes, edges, edge_cols = [], [], []
        for i, cen in enumerate(centroids):
            gt = gt_positions[np.linalg.norm(gt_positions - cen, axis=1).argmin()]
            err = np.linalg.norm(cen - gt)
            nodes += [cen, gt]
            edges.append([i * 2, i * 2 + 1])
            edge_cols.append([0.2, 0.9, 0.2] if err < 0.02 else
                             [0.9, 0.9, 0.1] if err < 0.05 else [0.9, 0.2, 0.2])
        net = ps.register_curve_network("Error lines (det → GT)",
                                        np.array(nodes), np.array(edges), radius=0.002)
        net.add_color_quantity("error", np.array(edge_cols),
                               defined_on="edges", enabled=True)

    # ── Diagnostic layers (hidden by default, toggle in left panel) ───────────
    if len(outlier_pts):
        out = ps.register_point_cloud("[diag] Removed outliers", outlier_pts, radius=0.002)
        out.add_color_quantity("col", _solid(len(outlier_pts), [0.95, 0.15, 0.15]), enabled=True)
        out.set_enabled(False)
    if len(noise_pts):
        ns = ps.register_point_cloud("[diag] DBSCAN noise", noise_pts, radius=0.002)
        ns.add_color_quantity("col", _solid(len(noise_pts), [0.55, 0.55, 0.55]), enabled=True)
        ns.set_enabled(False)
    if len(rejected_pts):
        rj = ps.register_point_cloud("[diag] Rejected clusters", rejected_pts, radius=0.002)
        rj.add_color_quantity("col", _solid(len(rejected_pts), [0.95, 0.55, 0.05]), enabled=True)
        rj.set_enabled(False)

    ps.show()


if __name__ == "__main__":
    main()
