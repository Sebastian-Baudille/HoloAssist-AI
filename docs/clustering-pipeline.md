# RGB-D Point Cloud Clustering & PCA Reference

> Reference document for implementing K-means clustering + PCA centroid/orientation extraction on RGB-D point clouds, specifically for distinguishing **cubes** from a **table surface** and other environmental clutter.

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Pipeline Overview](#2-pipeline-overview)
3. [Dependencies](#3-dependencies)
4. [Stage 1 – Preprocessing](#4-stage-1--preprocessing)
5. [Stage 2 – Feature Construction](#5-stage-2--feature-construction)
6. [Stage 3 – K-Means Clustering](#6-stage-3--k-means-clustering)
7. [Stage 4 – Cluster Filtering (Cube Identification)](#7-stage-4--cluster-filtering-cube-identification)
8. [Stage 5 – PCA per Cube Cluster](#8-stage-5--pca-per-cube-cluster)
9. [Alternative: DBSCAN Instead of K-Means](#9-alternative-dbscan-instead-of-k-means)
10. [Visualization (Polyscope / Open3D)](#10-visualization-polyscope--open3d)
11. [Parameter Tuning Cheat Sheet](#11-parameter-tuning-cheat-sheet)
12. [Common Pitfalls](#12-common-pitfalls)
13. [Full End-to-End Example](#13-full-end-to-end-example)

---

## 1. Problem Statement

**Input:** RGB-D point cloud of a tabletop scene containing:
- A planar **table surface** (large, dominant in point count)
- One or more **cubes** sitting on the table
- Possibly other clutter (background walls, floor, miscellaneous objects)

**Goal:**
1. Cluster the points into meaningful groups.
2. Identify which clusters correspond to cubes.
3. Compute each cube's **centroid** (3D position) and **orientation** (principal axes) using PCA.

**Why this is non-trivial:**
- Raw K-means on XYZ alone will waste clusters on the dominant table surface.
- Without normalization, spatial scale (meters) drowns out color (0–1).
- Cubes often share spatial proximity but differ in color — or vice versa.

---

## 2. Pipeline Overview

```
RGB-D point cloud (XYZ + RGB)
        │
        ▼
[1] RANSAC plane removal       ← isolate table, remove it
        │
        ▼
[2] Height crop                ← remove floor / ceiling / far clutter
        │
        ▼
[3] Feature scaling            ← normalize XYZ + RGB independently
        │
        ▼
[4] K-Means (or DBSCAN)        ← group remaining points
        │
        ▼
[5] Cluster filtering          ← keep only cube-shaped clusters
        │
        ▼
[6] PCA per cluster            ← centroid + orientation + extents
        │
        ▼
Pose estimates for each cube
```

**Critical insight:** Plane removal (step 1) is the single biggest practical win. Without it, K-means will collapse one of its clusters onto the table and the cube clusters will be polluted with table points near the cube bases.

---

## 3. Dependencies

```bash
pip install numpy open3d scikit-learn polyscope
```

| Library | Used for |
|---|---|
| `numpy` | Array operations |
| `open3d` | PLY/PCD I/O, RANSAC plane segmentation, point cloud structures |
| `scikit-learn` | K-Means, PCA, DBSCAN, StandardScaler |
| `polyscope` | Interactive 3D visualization |

---

## 4. Stage 1 – Preprocessing

### 4.1 Load the point cloud

```python
import numpy as np
import open3d as o3d

pcd = o3d.io.read_point_cloud("scene.ply")
points = np.asarray(pcd.points)        # (N, 3)
colors = np.asarray(pcd.colors)        # (N, 3) in [0, 1]
```

### 4.2 RANSAC plane removal (remove the table)

```python
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.01,   # 1 cm tolerance — tighten for clean data, loosen for noisy
    ransac_n=3,                # 3 points define a plane
    num_iterations=1000        # more iterations = more reliable but slower
)

# plane_model = [a, b, c, d] for plane ax + by + cz + d = 0
table_cloud   = pcd.select_by_index(inliers)
objects_cloud = pcd.select_by_index(inliers, invert=True)
```

**Tuning notes:**
- `distance_threshold` is the most important parameter. Start at 1 cm. Too tight → plane fragments; too loose → cube bases get absorbed into the plane.
- If the table is not the largest plane (e.g. a big floor is visible), call `segment_plane` repeatedly and keep the one with the right normal direction or expected height.

### 4.3 Height crop

After removing the table, crop above the table to keep only objects:

```python
# table_height typically derived from plane_model: z = -(a*x + b*y + d) / c
# For a roughly horizontal table you can also just take the median table z
table_z = np.asarray(table_cloud.points)[:, 2].mean()

obj_points = np.asarray(objects_cloud.points)
obj_colors = np.asarray(objects_cloud.colors)

z_min = table_z + 0.005    # 5 mm above table to avoid residual plane noise
z_max = table_z + 0.30     # 30 cm above — adjust based on cube size

mask = (obj_points[:, 2] > z_min) & (obj_points[:, 2] < z_max)
points = obj_points[mask]
colors = obj_colors[mask]
```

### 4.4 Optional: voxel downsample

If you have hundreds of thousands of points, K-means becomes slow:

```python
filtered = o3d.geometry.PointCloud()
filtered.points = o3d.utility.Vector3dVector(points)
filtered.colors = o3d.utility.Vector3dVector(colors)
filtered = filtered.voxel_down_sample(voxel_size=0.005)  # 5 mm voxels
points = np.asarray(filtered.points)
colors = np.asarray(filtered.colors)
```

### 4.5 Optional: statistical outlier removal

```python
filtered, _ = filtered.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

---

## 5. Stage 2 – Feature Construction

K-Means uses **Euclidean distance**, so feature scales must be comparable. XYZ in meters (~0.5 m range) and RGB in [0, 1] are not — without scaling, geometry will dominate completely.

```python
from sklearn.preprocessing import StandardScaler

spatial = points[:, :3]    # XYZ in meters
color   = colors[:, :3]    # RGB in [0, 1]

# Scale each block to zero mean, unit variance independently
spatial_scaled = StandardScaler().fit_transform(spatial)
color_scaled   = StandardScaler().fit_transform(color)

# Weight controls relative influence
color_weight = 2.0   # raise if cubes are color-distinct, lower if same-colored
features = np.hstack([spatial_scaled, color_scaled * color_weight])
```

**`color_weight` guidance:**

| Scenario | Recommended weight |
|---|---|
| Cubes are visually distinct (red, blue, green) | 2.0 – 4.0 |
| Cubes are similar colors but well separated spatially | 0.5 – 1.0 |
| Cubes touch or overlap spatially, distinct colors | 3.0 – 5.0 |
| All cubes same color | 0.0 (use spatial only) |

---

## 6. Stage 3 – K-Means Clustering

```python
from sklearn.cluster import KMeans

k = 4    # number of cubes + 1 for residual clutter
kmeans = KMeans(
    n_clusters=k,
    init='k-means++',     # smart init — avoids most local minima
    n_init=10,            # run 10 times, keep best
    random_state=42,      # reproducibility
    max_iter=300
)
labels = kmeans.fit_predict(features)
```

### Choosing k

If the number of cubes is unknown:

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(features)
    scores.append((k, silhouette_score(features, km.labels_)))
print(scores)   # pick k with highest silhouette
```

The **elbow method** on `kmeans.inertia_` is the other standard approach, but silhouette is more reliable for irregular-shaped clusters typical of point clouds.

---

## 7. Stage 4 – Cluster Filtering (Cube Identification)

Not every K-means cluster is a cube — some will be clutter. Filter on geometric priors:

```python
cube_clusters = []

for cluster_id in range(k):
    cluster_pts = points[labels == cluster_id]

    if len(cluster_pts) < 50:          # too small → noise
        continue

    bbox_min = cluster_pts.min(axis=0)
    bbox_max = cluster_pts.max(axis=0)
    dims     = bbox_max - bbox_min

    # Size filter: each side within expected cube range
    is_cube_sized = all(0.03 < d < 0.15 for d in dims)   # 3 cm – 15 cm

    # Aspect ratio: cubes have roughly equal dimensions
    is_cubic = (dims.max() / max(dims.min(), 1e-6)) < 1.8

    if is_cube_sized and is_cubic:
        cube_clusters.append((cluster_id, cluster_pts))

print(f"Identified {len(cube_clusters)} cube candidates")
```

**Adjust thresholds to match your application:**
- Cube edge length range → `0.03 < d < 0.15`
- Aspect ratio cap → `1.8` (allow some slack for partial occlusion)
- Minimum point count → `50` (depends on sensor resolution and downsampling)

---

## 8. Stage 5 – PCA per Cube Cluster

PCA on each cube gives the centroid, the three principal axes, and the spread along each axis. For an idealized cube observed fully, the three eigenvalues should be approximately equal.

```python
from sklearn.decomposition import PCA

cube_poses = []

for cluster_id, cluster_pts in cube_clusters:
    pca = PCA(n_components=3)
    pca.fit(cluster_pts)

    centroid     = pca.mean_                           # (3,) — 3D centroid
    axes         = pca.components_                     # (3, 3) — rows are PC1, PC2, PC3
    eigenvalues  = pca.explained_variance_             # (3,) — variance along each axis
    extents      = np.sqrt(eigenvalues)                # (3,) — std along each axis
    var_ratio    = pca.explained_variance_ratio_       # (3,) — ~[0.33, 0.33, 0.33] for cube

    cube_poses.append({
        'cluster_id'   : cluster_id,
        'centroid'     : centroid,
        'axes'         : axes,
        'extents'      : extents,
        'variance_ratio': var_ratio,
    })

    print(f"Cube {cluster_id}:")
    print(f"  Centroid:        {centroid}")
    print(f"  PC1 (longest):   {axes[0]}")
    print(f"  PC2:             {axes[1]}")
    print(f"  PC3 (normal):    {axes[2]}")
    print(f"  Extents:         {extents}")
    print(f"  Variance ratio:  {var_ratio}")
```

### Interpreting the output

| Output | Meaning |
|---|---|
| `pca.mean_` | True 3D centroid (more accurate than bbox center for partially occluded cubes) |
| `pca.components_[0]` | Primary axis — direction of greatest variance |
| `pca.components_[2]` | Normal to the flattest face (useful for grasp planning) |
| `pca.explained_variance_ratio_` | Should be roughly `[0.33, 0.33, 0.33]` for a well-observed cube |

### Cube-shape confidence score

A simple sanity check on whether a cluster is *actually* cube-shaped:

```python
def cubeness_score(variance_ratio):
    """1.0 = perfect cube, 0.0 = degenerate (line or plane)."""
    expected = np.array([1/3, 1/3, 1/3])
    return 1.0 - np.linalg.norm(variance_ratio - expected) / np.linalg.norm(expected)
```

Reject candidates with score below ~0.7.

### Building a 4×4 pose matrix

```python
def pose_matrix(centroid, axes):
    """Return a 4x4 SE(3) pose. axes rows = PC1, PC2, PC3."""
    R = axes.T                           # columns are basis vectors
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3]  = centroid
    return T
```

---

## 9. Alternative: DBSCAN Instead of K-Means

DBSCAN is often **better than K-means** for this kind of problem because:
- You don't need to specify the number of clusters.
- It naturally handles noise points (label = `-1`).
- It separates spatially disconnected groups even when their centroids are close in feature space.

```python
from sklearn.cluster import DBSCAN

# DBSCAN on spatial coordinates only (in meters)
db = DBSCAN(
    eps=0.02,         # 2 cm — points within this distance are neighbors
    min_samples=50    # cluster must have at least 50 points
)
labels = db.fit_predict(points[:, :3])

# Process each cluster (skip noise label -1)
unique_labels = [l for l in np.unique(labels) if l != -1]
for cluster_id in unique_labels:
    cluster_pts = points[labels == cluster_id]
    # ... same filtering and PCA as K-means path
```

**When to choose which:**

| Use **K-means** when | Use **DBSCAN** when |
|---|---|
| You know the cube count | Cube count varies / unknown |
| Cubes are spatially close together | Cubes are well separated |
| You want color to influence grouping | Spatial proximity is enough |
| Density is uniform | Density varies (some near, some far) |

---

## 10. Visualization (Polyscope / Open3D)

### Polyscope (recommended for inspection)

```python
import polyscope as ps

ps.init()
ps.set_up_dir("z_up")
ps.set_ground_plane_mode("none")

cloud = ps.register_point_cloud("Scene", points, radius=0.0015)
cloud.add_scalar_quantity("Cluster ID", labels.astype(float), enabled=True)

# Random color per cluster
np.random.seed(0)
palette = np.random.rand(int(labels.max()) + 1, 3)
cluster_colors = palette[labels]
cloud.add_color_quantity("Cluster Colors", cluster_colors, enabled=False)

# Centroids and PCA axes
centroids = np.array([p['centroid'] for p in cube_poses])
pc1 = np.array([p['axes'][0] * p['extents'][0] for p in cube_poses])
pc2 = np.array([p['axes'][1] * p['extents'][1] for p in cube_poses])
pc3 = np.array([p['axes'][2] * p['extents'][2] for p in cube_poses])

centers_pc = ps.register_point_cloud("Cube Centroids", centroids, radius=0.005)
centers_pc.add_vector_quantity("PC1", pc1, enabled=True, color=(1, 0, 0), vectortype="ambient")
centers_pc.add_vector_quantity("PC2", pc2, enabled=True, color=(0, 1, 0), vectortype="ambient")
centers_pc.add_vector_quantity("PC3", pc3, enabled=True, color=(0, 0, 1), vectortype="ambient")

ps.show()
```

**Key Polyscope conventions:**
- Use `vectortype="ambient"` to pass world-space vectors with absolute lengths (so `axis * sqrt(eigenvalue)` renders proportionally).
- Use `vectortype="standard"` if you only care about direction.
- `radius` is in world units; for typical RGB-D scenes 0.001–0.005 is reasonable.

### Open3D (alternative)

```python
viz = o3d.geometry.PointCloud()
viz.points = o3d.utility.Vector3dVector(points)
viz.colors = o3d.utility.Vector3dVector(palette[labels])
o3d.visualization.draw_geometries([viz])
```

---

## 11. Parameter Tuning Cheat Sheet

| Parameter | Stage | Default | Increase if… | Decrease if… |
|---|---|---|---|---|
| `distance_threshold` (RANSAC) | Plane removal | 0.01 m | Plane fragments / table not fully removed | Cube bases get absorbed into plane |
| `num_iterations` (RANSAC) | Plane removal | 1000 | Plane detection unreliable | Need faster runtime |
| `voxel_size` | Downsample | 0.005 m | Too slow / too many points | Losing fine detail |
| `color_weight` | Features | 2.0 | Cubes color-distinct but spatially close | Cubes same color |
| `k` (KMeans) | Clustering | #cubes + 1 | Cubes being merged | Cubes being split |
| `n_init` (KMeans) | Clustering | 10 | Results unstable across runs | Need faster runtime |
| `eps` (DBSCAN) | Clustering | 0.02 m | Cubes split into pieces | Adjacent cubes merge |
| `min_samples` (DBSCAN) | Clustering | 50 | Too much noise labeled as clusters | Real cubes labeled as noise |
| Cube size range | Filtering | 0.03–0.15 m | Cubes outside range | Picking up clutter |
| Aspect ratio cap | Filtering | 1.8 | Partially occluded cubes rejected | Non-cube objects accepted |

---

## 12. Common Pitfalls

1. **Forgetting to remove the plane.** K-means will dedicate one or more clusters to the (huge) table surface, leaving cubes underclustered.
2. **Not scaling features.** RGB in [0, 1] versus XYZ in meters means RGB contributes essentially nothing.
3. **Using `random_state=None` while debugging.** Different seeds give different results — set a fixed seed until the pipeline is stable.
4. **Confusing `pca.components_` rows vs columns.** `components_[i]` is the *i*-th principal axis as a row vector of length 3.
5. **Treating PCA axes as a rotation matrix without checking handedness.** PCA may give a left-handed frame; check `det(R)` and flip an axis if it's −1.
6. **Forgetting that PCA sign is arbitrary.** The direction of each principal component can flip from frame to frame; if you need temporal consistency, align signs against a reference (e.g. pick PC3 to point "up").
7. **Setting `eps` too small in DBSCAN.** Even 1 mm of sensor noise will fragment a cube into many small clusters. Start at 2× the voxel size.
8. **Dropping ground-truth labels too early.** Keep the label array aligned with `points` through every filtering step or downstream evaluation breaks.
9. **Running PCA on raw clusters that include outliers.** A few stray points pull the centroid and skew the eigenvectors. Use statistical outlier removal *per cluster* before PCA if needed.

---

## 13. Full End-to-End Example

```python
import numpy as np
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def process_rgbd_scene(ply_path: str, n_cubes_estimate: int = 3):
    # ---- 1. Load ----
    pcd = o3d.io.read_point_cloud(ply_path)

    # ---- 2. Plane removal ----
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01, ransac_n=3, num_iterations=1000
    )
    a, b, c, d = plane_model
    objects = pcd.select_by_index(inliers, invert=True)

    # ---- 3. Height crop above the table ----
    pts = np.asarray(objects.points)
    cols = np.asarray(objects.colors)
    # Plane equation: ax+by+cz+d=0; signed distance from plane:
    signed_dist = (pts @ np.array([a, b, c]) + d) / np.linalg.norm([a, b, c])
    mask = (signed_dist > 0.005) & (signed_dist < 0.30)
    pts = pts[mask]
    cols = cols[mask]

    # ---- 4. Feature scaling ----
    spatial_scaled = StandardScaler().fit_transform(pts)
    color_scaled   = StandardScaler().fit_transform(cols)
    features = np.hstack([spatial_scaled, color_scaled * 2.0])

    # ---- 5. K-Means ----
    k = n_cubes_estimate + 1   # +1 for residual clutter
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(features)

    # ---- 6. Filter cube clusters ----
    cubes = []
    for cid in range(k):
        cluster_pts = pts[labels == cid]
        if len(cluster_pts) < 50:
            continue
        dims = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
        if not all(0.03 < d < 0.15 for d in dims):
            continue
        if dims.max() / max(dims.min(), 1e-6) > 1.8:
            continue
        cubes.append((cid, cluster_pts))

    # ---- 7. PCA per cube ----
    poses = []
    for cid, cluster_pts in cubes:
        pca = PCA(n_components=3).fit(cluster_pts)
        poses.append({
            'cluster_id'    : cid,
            'centroid'      : pca.mean_,
            'axes'          : pca.components_,
            'extents'       : np.sqrt(pca.explained_variance_),
            'variance_ratio': pca.explained_variance_ratio_,
        })

    return pts, labels, poses


if __name__ == "__main__":
    points, labels, cube_poses = process_rgbd_scene("scene.ply", n_cubes_estimate=3)
    print(f"Found {len(cube_poses)} cubes")
    for p in cube_poses:
        print(f"  Cluster {p['cluster_id']}: centroid={p['centroid']}, "
              f"extents={p['extents']}")
```

---

## Quick Reference Card

```
PREPROCESS  → plane removal, height crop, voxel downsample
FEATURES    → StandardScaler on XYZ and RGB independently, weight RGB
CLUSTER     → KMeans (k known) or DBSCAN (k unknown)
FILTER      → bbox dimensions + aspect ratio + point count
PCA         → mean = centroid, components = axes, sqrt(variance) = extents
VALIDATE    → variance_ratio close to [0.33, 0.33, 0.33] for cubes
VISUALIZE   → Polyscope point cloud + vector quantities for axes
```
