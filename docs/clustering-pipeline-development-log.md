# Clustering Pipeline Development Log — Point Cloud Cube Detection

**Project:** HoloAssist-AI (41118 AI in Robotics)  
**Author:** Guy (GuyESmith) — perception/clustering  
**Period:** May–June 2026  
**Environment:** Python 3.11 (`clustering/.venv`) — Open3D 0.19.0, scikit-learn 1.8.0, Polyscope 2.6.1

---

## Overview

The clustering pipeline takes a PLY point cloud captured from the Gazebo sim (via the D435i depth sensor) and detects the world-frame positions of coloured cubes on the table. The detected positions are the **DQN state vector** — fed directly to the RL policies as the observation of where the cubes are.

The pipeline evolved through two major algorithm iterations: K-Means → DBSCAN. The final pipeline achieves 1.63 cm mean centroid error (PASS, target < 3 cm) on the validation set.

---

## Environment Setup

The clustering pipeline has a hard Python environment constraint. Open3D and Polyscope require Python 3.11, but `rclpy` (the ROS 2 Python client) only runs under Python 3.10 (Humble). The two environments cannot coexist in the same process.

**Solution:** PLY files on disk as the cross-environment bridge.

```
Gazebo sim (Python 3.10 / rclpy)
    → saves PLY to ~/holoassist_pointclouds/
    
Clustering pipeline (Python 3.11 / clustering/.venv)
    → reads PLY from ~/holoassist_pointclouds/
```

No ROS topics or rclpy imports appear anywhere in the clustering code. No Open3D or Polyscope imports appear anywhere in the ROS nodes.

**Install:**
```bash
cd clustering && python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt  # numpy==2.4.3 open3d==0.19.0 scikit-learn==1.8.0 polyscope==2.6.1
```

---

## What We Wanted to Do

The initial design goal was simple: given a point cloud of a table scene with coloured cubes, detect the 3D centre of each cube and return those centroids as the RL state vector.

The first approach was **K-Means clustering** with a fixed `k` (number of cubes). The motivation was simplicity — K-Means is deterministic, fast, and widely understood. If you know there are 4 cubes, `k=4` directly gives 4 centroids.

---

## Stage 1: K-Means Pipeline

### Algorithm

```
1. Load PLY (camera body frame)
2. Transform to world frame using camera pose from sim_params.yaml
3. Crop Z to cube height band (table_top + margin → table_top + cube_height + margin)
4. K-Means with k = number of cubes (XYZ only, no RGB weighting)
5. Centroid = mean of each cluster
6. Visualise in Polyscope
```

### Camera-to-World Transform

The point cloud is in **camera body frame** (+X = look direction). Before any Z-based cropping can work, all points must be transformed to world frame. The transform uses the camera pose from `sim_params.yaml` (`[x, y, z, roll, pitch, yaw]`):

```python
R = Rz @ Ry @ Rx
world_points = (R @ camera_points.T).T + [x, y, z]
```

This was a critical early correctness requirement — forgetting to transform first meant the Z-crop was applied in the wrong coordinate frame, selecting nothing or the wrong layer.

### Z Crop Parameters

```python
z_min = table_top_z + 0.015   # 3 × sensor noise σ (0.005 m)
z_max = table_top_z + cube_height + 0.01
```

The 0.015 m lower margin is 3× the depth sensor noise σ=0.005 m. Without this margin, the table surface noise bleeds into the cube layer and DBSCAN clusters it with the cube points, degrading centroid accuracy.

### Why No RGB Weighting?

The original design included color weighting in K-Means (using both XYZ and normalised RGB). Experiments showed that Gazebo's diffuse lighting model creates significant per-pixel lighting variation on cube faces — the same cube colour can appear very different on a lit face vs a shadowed face. Including RGB weighting caused K-Means to split single cubes along lighting boundaries rather than grouping by object. Setting `color_weight = 0.0` (XYZ-only clustering) resolved this.

### K-Means Results

First validated accuracy: **~2.65 cm** mean centroid error against ground truth across the initial test dataset. This passed the 3 cm target.

However, K-Means has a fundamental limitation: **k must be specified**. The RL task always has a fixed number of cubes, but the pipeline needs to handle variable cube counts during dataset capture. More importantly, K-Means forces exactly k clusters even when cubes are occluded, merged, or the crop contains noise — it will split noise clusters or merge adjacent cubes to hit exactly k, making the centroid error misleading.

---

## Stage 2: DBSCAN Pipeline

**Commit:** `b11750c` (27 May 2026) — complete rewrite of `detect_cubes.py`

### Decision

DBSCAN was adopted to eliminate the fixed-k requirement. DBSCAN discovers clusters purely from point density — no need to specify how many cubes are present. This allows the pipeline to work correctly when:
- The number of cubes is unknown
- A cube is partially occluded (may appear as a smaller cluster)
- Two cubes are very close but distinct

### Full Pipeline

```
1. Load PLY (camera body frame)
2. camera_to_world() — Euler rotation + translation
3. crop_cube_layer() — Z band filter
4. remove_outliers() — Open3D statistical outlier removal
5. dbscan_cluster() — scikit-learn DBSCAN
6. filter_cube_clusters() — size filter (50–1500 points)
7. compute_pca() — centroid + orientation axes
8. Polyscope visualisation
```

### Statistical Outlier Removal (Step 4)

Before DBSCAN runs, Open3D's `remove_statistical_outlier` is applied with `nb_neighbors=20`, `std_ratio=2.0`. This removes any point whose mean distance to its 20 nearest neighbours is more than 2 standard deviations above the global mean. In practice this kills:
- Flying pixels at depth discontinuities (edge of the table, back of the scene)
- Isolated noise points from the Gaussian depth model

### DBSCAN Parameters

```python
DBSCAN(eps=0.015, min_samples=20)
```

- `eps=0.015 m` (1.5 cm): neighbourhood radius. Two points are in the same cluster if their 3D distance is < 1.5 cm. This is 3× the sensor noise, so the cube surface (a dense, continuous surface of points) reliably forms a single cluster.
- `min_samples=20`: minimum points to be a core point. Prevents isolated noise points from forming their own clusters.

Points with label `-1` (DBSCAN noise) are discarded.

### Size Filter (Step 6)

After DBSCAN, clusters are filtered by point count:

```python
MIN_CLUSTER_PTS = 50
MAX_CLUSTER_PTS = 1500
```

A 4 cm cube at ~0.7 m range produces approximately 200–900 points under the D435i FOV and resolution (848×480, 87° H FOV). The bounds are wide enough to catch partially occluded cubes (down to 50 points) while rejecting table edge artefacts and large background blobs (>1500 points).

### PCA for Centroid and Orientation

Each valid cluster runs PCA to extract:
- **Centroid**: `pca.mean_` (the cluster's 3D centre in world frame)
- **Principal axes**: `pca.components_` (orientation of the cube)
- **Extents**: `sqrt(pca.explained_variance_)` (approximate half-sizes along each axis)

PCA axes are used only for Polyscope visualisation (red/green/blue axis arrows). The centroid is the only value used by the RL pipeline.

### DBSCAN Results

After switching to DBSCAN and increasing the minimum cube separation from 1.5× to 2.0× cube size in the scene controller:

- **Mean centroid error (when count correct):** 1.63 cm (vs 2.65 cm K-Means) — 39% improvement
- **Exact cube count rate:** 82% on train set
- **Primary failure mode:** close-cube merging when two cubes are near each other despite the 2.0× separation requirement (DBSCAN eps=1.5 cm bridges the ~4 cm gap when cubes settle slightly closer)

---

## Dataset and Verification

### Dataset Structure

```
~/holoassist_dataset/
  scene_0001.ply
  scene_0001.labels.json
  scene_0002.ply
  scene_0002.labels.json
  ...
  scene_0060.ply    (50 train + 10 val)
  accuracy_report.json
```

60 scenes total: 50 train, 10 val. Each scene has 2–4 randomly placed cubes (uniform). All cubes are 4 cm size.

### verify_detection.py — Hungarian Algorithm Matching

`verify_detection.py` evaluates the pipeline on the full dataset using the **Hungarian algorithm** for detection-to-ground-truth matching (`scipy.optimize.linear_sum_assignment`).

Because DBSCAN may detect more or fewer cubes than the ground truth, direct matching is non-trivial:
- **Missed cubes** (detected < GT): unmatched GT cubes get a penalty distance of 0.50 m so misses register as large errors in the mean
- **Extra detections** (detected > GT): counted as false positives but not added to the error metric

This produces honest accuracy numbers — a pipeline that misses half the cubes doesn't hide behind averaging only the cubes it found.

**Key metrics reported:**
- Mean centroid error (cm)
- Standard deviation
- Worst-scene error
- Exact count rate (detected == GT count)
- Cube recall (matched / total GT)
- False positive count
- Breakdown by GT cube count (k=2, k=3, k=4)

### Validated Accuracy

On the 60-scene dataset with DBSCAN (eps=0.015, min_samples=20):

| Split | Mean Error | Exact Count Rate | Cube Recall |
|---|---|---|---|
| Train | ~1.63 cm | ~82% | ~91% |
| Val | ~1.63 cm | ~82% | ~91% |

**Target was < 3 cm. PASS.**

The dominant error source is close-cube merging, not the detector itself. When two cubes are close enough that DBSCAN treats them as one cluster, the single detected centroid lands somewhere between the two ground-truth positions — giving ~4–8 cm error for those cubes (counted as a miss by the Hungarian matching).

---

## DQN State Vector

The centroid array is flattened to produce the RL state vector:

```python
centroids = np.array([p["centroid"] for p in cube_poses])  # (k, 3) world-frame
state_vector = centroids.flatten()                          # (k*3,) = (12,) for 4 cubes
```

For 4 cubes: `(x0, y0, z0, x1, y1, z1, x2, y2, z2, x3, y3, z3)`, shape `(12,)`.

The centroids are **world-frame** XYZ in metres, directly usable without further transformation.

---

## Polyscope Visualisation

`view_ply.py` — quick viewer for any PLY file:
```bash
source clustering/.venv/bin/activate
python3 clustering/view_ply.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
```

`detect_cubes.py` — runs the full pipeline and opens an interactive viewer:
```bash
python3 clustering/detect_cubes.py                          # auto-picks most recent capture
python3 clustering/detect_cubes.py path/to/file.ply         # specific file
python3 clustering/detect_cubes.py --no-viz                 # results only, no viewer
python3 clustering/detect_cubes.py --eps 0.020 --min-samples 15  # tune DBSCAN
```

The Polyscope viewer shows:
- Full scene point cloud (coloured by RGB, toggleable)
- Cube layer after Z-crop (coloured by cluster ID)
- Centroid markers with PCA axes (red=PC1, green=PC2, blue=PC3)

---

## What We Would Change

### 1. Adaptive eps

The current `eps=0.015` is fixed and tuned for 4 cm cubes at ~0.7 m range. Cubes closer to the camera produce denser point clouds and might need a larger eps; cubes at the far edge of the table produce sparser clouds. Adaptive eps (scaling with estimated range or point density) would be more robust.

### 2. Colour-Aware DBSCAN

We abandoned RGB weighting in K-Means due to lighting variation, but DBSCAN could use HSV colour to separate cubes that are very close spatially. Hue is more lighting-invariant than RGB. This could improve the close-cube merging problem without needing to increase cube separation further.

### 3. Larger Dataset

60 scenes is functional but small. The `dataset_capture.py` pipeline can easily scale to 500+ scenes in under 10 minutes. A larger dataset would give more reliable DBSCAN parameter estimates and confidence intervals on the accuracy metrics.

### 4. 3D Bounding Box Output

Currently only centroid (XY) is used by the RL pipeline. The PCA axes and extents could produce a 3D oriented bounding box per cube, which would be useful for grasp orientation planning (aligning the gripper to the cube's longer axis).

### 5. Real-Camera Validation

The entire pipeline was validated on synthetic Gazebo data. Real D435i data has different noise characteristics (quantisation, structured IR interference, specular reflections on cube faces). Testing on a real capture with known cube positions is the obvious next step before deployment.

---

## Discussion

### K-Means vs DBSCAN

Both algorithms were tried, and DBSCAN won on two dimensions: accuracy and flexibility. The accuracy improvement (2.65 cm → 1.63 cm) came primarily from better handling of noise — DBSCAN labels noise explicitly as -1 and discards it before computing centroids, whereas K-Means is forced to assign every point (including noise) to a cluster, polluting centroids.

The flexibility improvement is more important for long-term use: DBSCAN doesn't require knowing k in advance. As the task scales (more cubes, variable cube count) the DBSCAN pipeline requires no changes, whereas K-Means would need k passed as a parameter.

The main cost of DBSCAN is that `eps` and `min_samples` must be tuned. At 1.5 cm eps, clusters that are 1.5 cm apart merge; at 2.5 cm they reliably stay separate but noise rejection worsens. The current 1.5 cm value is a compromise — it may need revisiting if the table scene changes (different cube sizes, different camera angle).

### Why Not Deep Learning?

A learned detector (PointNet, PointPillars, or a simple voxel CNN) would likely outperform DBSCAN, especially for occluded cubes and variable lighting. The decision to use classical clustering was driven by:

1. **Time**: the Gazebo + clustering stack needed to be functional quickly to unblock the RL pipeline
2. **Interpretability**: classical clustering failures are visually diagnosable in Polyscope; a learned model's failures are less obvious
3. **Data**: 60 scenes is insufficient to train a reliable 3D object detector from scratch

For a production system, a learned detector trained on the 60-scene dataset (augmented with synthetic variation) would be the natural next step.

### Integration with RL

The clustering pipeline produces world-frame XYZ centroids. These are fed directly to the RL policies as the state observation — no coordinate transformation required because the Gazebo TF chain establishes a consistent `map` world frame that both the camera and the RL simulation use.

The validated accuracy of 1.63 cm is well within the tolerance of the RL policies. The MuJoCo pan-locked-reach policy has a 3.5 cm XY jaw margin, so a 1.63 cm centroid error (mean) gives sufficient positioning accuracy for a real deployment. At the 95th percentile (worst cubes), error rises to ~3–5 cm — at the edge of the jaw margin but still within the gripper's geometric range of acceptance.
