# HoloAssist-AI — Perception Stack: Full Technical Analysis

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Layer 1 — Simulation & Sensor](#3-layer-1--simulation--sensor)
4. [Layer 2 — ROS Bridge & TF](#4-layer-2--ros-bridge--tf)
5. [Layer 3 — Point Cloud Capture](#5-layer-3--point-cloud-capture)
6. [Layer 4 — Dataset Generation](#6-layer-4--dataset-generation)
7. [Layer 5 — Clustering & Detection](#7-layer-5--clustering--detection)
8. [Layer 6 — RL Observation Interface](#8-layer-6--rl-observation-interface)
9. [Configuration Reference](#9-configuration-reference)
10. [How to Run](#10-how-to-run)
11. [Validated Performance](#11-validated-performance)
12. [Short Summary Extract](#12-short-summary-extract)

---

## 1. System Overview

The perception stack converts a simulated depth camera scene (Gazebo Fortress) into a structured state vector of cube centroids for downstream reinforcement learning. It operates in two isolated Python environments:

| Environment | Python | Role |
|---|---|---|
| `ros2_ws/` | 3.10 (ROS 2 Humble) | Simulation, sensor bridging, PLY capture |
| `clustering/.venv` | 3.11 | Clustering, cube detection, visualisation |

The bridge between them is a set of PLY files written to `~/holoassist_pointclouds/` (raw capture) or `~/holoassist_dataset/` (labelled dataset). No ROS imports ever enter the clustering environment; no open3d/polyscope imports ever enter the ROS environment.

---

## 2. Architecture & Data Flow

```
┌──────────────────────────────────────────────────────────────┐
│  Gazebo Ignition Fortress (sim)                              │
│  RGBD sensor → /camera/points (PointCloud2, body frame)      │
└──────────────────┬───────────────────────────────────────────┘
                   │ ros_gz_bridge
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ROS 2 (Python 3.10)                                        │
│  save_pointcloud.py — subscribe, unpack, downsample → PLY   │
│  dataset_capture.py — randomise scene, capture, label       │
└──────────────────┬───────────────────────────────────────────┘
                   │ ~/holoassist_pointclouds/*.ply
                   │ ~/holoassist_dataset/*.ply + *.labels.json
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  Clustering (Python 3.11 venv)                              │
│  detect_cubes.py                                            │
│  1. Camera→World transform (ZYX Euler)                      │
│  2. Z-crop to cube layer                                    │
│  3. Statistical outlier removal                             │
│  4. DBSCAN clustering                                       │
│  5. Size filter                                             │
│  6. PCA per cluster → centroid, axes                        │
│  → centroids.flatten() → DQN/PPO state vector              │
└──────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — Simulation & Sensor

### Files
- `ros2_ws/src/holoassist_sim/config/sim_params.yaml` — single source of truth
- `ros2_ws/src/holoassist_sim/worlds/table_cubes.sdf.jinja2` — Jinja2 world template
- `ros2_ws/src/holoassist_sim/scripts/render_world.py` — template renderer
- `ros2_ws/src/holoassist_sim/launch/sim.launch.py` — orchestration

### Camera Model

The camera simulates an Intel D435i RGBD sensor. It is mounted as a static link (`d435i_link`) in the world.

```
Camera pose (sim_params.yaml):
  position:  x=-0.60 m,  y=0.0 m,  z=0.85 m
  rotation:  roll=0.0,   pitch=0.5281 rad (≈30.25°),  yaw=0.0

  → positioned 60 cm in front of the table, 85 cm high
  → tilted 30° downward to view the table surface
  → +X axis is the look direction (body frame convention)
```

Sensor parameters:

| Parameter | Value |
|---|---|
| Resolution | 848 × 480 px |
| Horizontal FOV | 87° |
| Depth range | 0.105 – 10 m |
| Update rate | 30 Hz |
| Depth noise | Gaussian σ = 0.005 m |
| Render engine | ogre2 |
| Frame ID | `d435i/d435i_link/rgbd` (Ignition scoped) |

### World Template Rendering

At launch, `sim.launch.py` invokes `render_world.py` twice — **before** Gazebo starts:

```
sim_params.yaml + table_cubes.sdf.jinja2       →  /tmp/.../table_cubes.sdf
sim_params.yaml + ros_gz_bridge.yaml.jinja2    →  /tmp/.../ros_gz_bridge.yaml
```

The renderer loads the YAML, passes it as a Jinja2 context, and writes the output. The output SDF must never be edited directly; it is regenerated every launch.

### Scene Physics

- **Table**: static, 0.5 m cube at z=0.25 m (top face at z=0.5 m)
- **Cubes**: dynamic, 40 mm, mass=0.05 kg, friction μ=0.8
- **Cube centre Z**: default 0.52 m (table top 0.5 + half-cube 0.02)
- **Physics step**: 0.001 s, real-time factor 1.0
- **Inertia**: computed inline in Jinja2 as `I = m/12 × (a² + b²)`

### Scene Controller

`ros2_ws/src/holoassist_sim/scripts/scene_controller.py` is a ROS 2 node that randomises the scene for dataset collection.

**Services:**
- `/scene/randomize_cubes` — spawns N cubes (2–4) with random pose/size/color
- `/scene/reset` — restores default layout from sim_params.yaml

**Spawning constraint:** Minimum centre-to-centre distance = `2.0 × cube_size`. At 40 mm this enforces a ≥4 cm surface gap — the minimum inter-cube separation required for DBSCAN `eps=0.015 m` to reliably separate clusters.

**Mechanism:** Writes per-cube SDF to `/tmp/holoassist_*.sdf`, then calls `ign service /world/{world_name}/create`. Removal calls `ign service /world/{world_name}/remove`.

---

## 4. Layer 2 — ROS Bridge & TF

### Files
- `ros2_ws/src/holoassist_sim/config/ros_gz_bridge.yaml.jinja2`

### Bridged Topics

| Ignition topic | ROS type | Notes |
|---|---|---|
| `/camera/image` | `sensor_msgs/Image` | RGB |
| `/camera/depth_image` | `sensor_msgs/Image` | Depth |
| `/camera/points` | `sensor_msgs/PointCloud2` | **Key perception topic** |
| `/camera/camera_info` | `sensor_msgs/CameraInfo` | Intrinsics |
| `/imu` | `sensor_msgs/Imu` | Conditional on `camera.imu.enabled` |
| `/clock` | `rosgraph_msgs/Clock` | Sim time |

### TF Chain

```
map
 └── d435i_link          (static, from camera.pose in sim_params.yaml)
      └── d435i/d435i_link/rgbd   (identity transform, same physical location)
```

Two static `transform_publisher` nodes are launched. The second (identity) transform reconciles the Ignition Fortress scoped frame ID `d435i/d435i_link/rgbd` with the link frame `d435i_link`.

---

## 5. Layer 3 — Point Cloud Capture

### File
- `ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py`

### What It Does

Subscribes to `/camera/points` (PointCloud2, BEST_EFFORT QoS, depth=5). On each message:

1. **Unpack fields** — reads `x`, `y`, `z` plus optional `rgb` from the PointCloud2 field descriptors
2. **RGB decode** — `rgb` is stored as a float32 bit-cast of a uint32 packed as `0x00RRGGBB`:
   ```python
   rgb_int = rgb.view(np.uint32)
   r = (rgb_int >> 16) & 0xFF
   g = (rgb_int >> 8)  & 0xFF
   b =  rgb_int        & 0xFF
   ```
3. **Drop non-finite** — removes any rows where x/y/z are NaN or Inf
4. **Voxel downsample** (if `capture.voxel_size > 0`) — hash-grid deduplication:
   ```python
   keys = np.floor(pts / voxel_size).astype(np.int64)
   hash = 73856093*keys[:,0] ^ 19349663*keys[:,1] ^ 83492791*keys[:,2]
   ```
   Keeps one point per voxel cell.
5. **Write binary PLY** — little-endian, header includes a timestamp comment

### Output Filename Convention

```
{description}_{N}cubes_{size}mm_v{NNN}.ply

Example: default_4cubes_40mm_v001.ply
```

Version number auto-increments by scanning the output directory. Functions: `_build_base_name(params)`, `_next_version(dir, base)`.

### One-shot Mode

When `capture.one_shot: true` (default), the node exits after writing the first PLY. Run it as a one-off command rather than a persistent service.

---

## 6. Layer 4 — Dataset Generation

### File
- `ros2_ws/src/holoassist_sim/scripts/dataset_capture.py`

### What It Does

Automates 60 labelled captures (50 train + 10 val) for clustering evaluation. Per scene:

1. Set `cube_count` parameter on `scene_controller` (range 2–4)
2. Call `/scene/randomize_cubes`
3. Wait 1.0 s for physics to settle
4. Query actual settled poses via Gazebo service:
   ```
   ign topic -e -n 1 /world/{world_name}/dynamic_pose/info
   ```
   Parses the protobuf text output with regex to extract x, y, z, qx, qy, qz, qw per cube
5. Subscribe to `/camera/points`, wait up to 10 s for a frame
6. Save PLY (voxel_size=0.0, no downsampling for ground truth quality)
7. Write `labels.json`:
   ```json
   {
     "scene_id": "scene_0001",
     "split": "train",
     "cube_count": 3,
     "world_name": "table_cubes",
     "table_top_z": 0.5,
     "cubes": [
       {
         "name": "cube_0",
         "position": {"x": 0.1, "y": 0.1, "z": 0.52},
         "orientation": {"qx": 0, "qy": 0, "qz": 0, "qw": 1},
         "size_m": 0.04
       }
     ]
   }
   ```
   Note: `size_m` is back-calculated from settled z: `size = (z - table_top_z) × 2`

### Output Layout

```
~/holoassist_dataset/
  scene_0001.ply
  scene_0001.labels.json
  ...
  scene_0060.ply
  scene_0060.labels.json
  accuracy_report.json        ← written by verify_detection.py
```

---

## 7. Layer 5 — Clustering & Detection

### Files
- `clustering/detect_cubes.py` — main detection pipeline
- `clustering/view_ply.py` — raw PLY viewer
- `clustering/verify_detection.py` — accuracy evaluation

### Dependencies

```
numpy==2.4.3
open3d==0.19.0          # PLY I/O, statistical outlier removal
scikit-learn==1.8.0     # DBSCAN, PCA, KMeans
polyscope==2.6.1        # 3D visualisation
```

### Detection Pipeline — Step by Step

#### Step 1: Load PLY

```python
pcd = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points)   # (N, 3) float32, camera body frame
colors = np.asarray(pcd.colors)   # (N, 3) float32, 0..1, if present
```

#### Step 2: Camera → World Transform

The PLY is in the camera body frame (`+X = look direction`). Transform to world frame using ZYX Euler angles from `sim_params.yaml`:

```python
roll, pitch, yaw = camera.pose[3], camera.pose[4], camera.pose[5]

Rx = [[1, 0,          0         ],
      [0, cos(roll), -sin(roll)  ],
      [0, sin(roll),  cos(roll)  ]]

Ry = [[ cos(pitch), 0, sin(pitch)],
      [  0,         1, 0         ],
      [-sin(pitch), 0, cos(pitch)]]

Rz = [[cos(yaw), -sin(yaw), 0],
      [sin(yaw),  cos(yaw), 0],
      [0,         0,        1]]

R = Rz @ Ry @ Rx
world_points = (R @ camera_points.T).T + [x, y, z]
```

**This transform must be applied before any Z-based cropping**, since the camera is tilted 30° and the raw Z axis is not aligned with the vertical world axis.

#### Step 3: Z-Crop to Cube Layer

```python
z_min = table_top_z + 0.015   # 0.515 m  (3× depth noise σ=0.005 m)
z_max = table_top_z + cube_height + 0.010  # 0.550 m

mask = (world_z >= z_min) & (world_z <= z_max)
cube_layer = world_points[mask]
```

This isolates the cube surfaces from the table surface and from objects above.

#### Step 4: Statistical Outlier Removal

```python
pcd_layer = o3d.geometry.PointCloud()
pcd_layer.points = o3d.utility.Vector3dVector(cube_layer)
pcd_clean, inlier_idx = pcd_layer.remove_statistical_outlier(
    nb_neighbors=20, std_ratio=2.0
)
```

Removes points whose mean distance to 20 nearest neighbours exceeds `mean + 2.0 × std_dev`. Eliminates flying pixels and depth discontinuity noise at cube edges.

#### Step 5: DBSCAN Clustering

```python
from sklearn.cluster import DBSCAN

labels = DBSCAN(eps=0.015, min_samples=20).fit_predict(clean_points)
# labels: -1 = noise, 0..k-1 = cluster IDs
```

Parameter rationale:
- `eps=0.015 m` (1.5 cm): scene_controller enforces ≥4 cm surface gap, so distinct cubes will never be within 1.5 cm of each other
- `min_samples=20`: sufficient to reject isolated noise patches while accepting real cube surfaces

#### Step 6: Size Filter

```python
MIN_CLUSTER_PTS = 50
MAX_CLUSTER_PTS = 1500

valid = [i for i in cluster_ids
         if MIN_CLUSTER_PTS <= count(i) <= MAX_CLUSTER_PTS]
```

A 40 mm cube at ~0.7 m distance produces approximately 200–900 points after outlier removal. Clusters outside [50, 1500] are rejected as non-cube objects or residual noise.

#### Step 7: PCA per Cluster

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=3).fit(cluster_points)
centroid   = cluster_points.mean(axis=0)      # world frame (x, y, z)
axes       = pca.components_                  # (3, 3) — principal axes
extents    = np.sqrt(pca.explained_variance_) # rough half-extents
```

The centroid is what feeds the RL state vector. The PCA axes are used for Polyscope visualisation only (orientation arrows).

#### Output: DQN/PPO State Vector

```python
state = centroids.flatten().round(4)
# shape: (k*3,) e.g. (12,) for 4 cubes
# values: world-frame x, y, z per cube in metres
```

### View-Only Pipeline (view_ply.py)

No clustering — just transforms and visualises:

1. Load PLY
2. Apply camera→world transform (same as above)
3. Launch Polyscope with `Z_up` orientation
4. Optionally colour by height scalar

### Verification Pipeline (verify_detection.py)

Runs the full detect_cubes pipeline over `~/holoassist_dataset/`, matches detections against ground truth labels using the **Hungarian algorithm**:

```python
from scipy.optimize import linear_sum_assignment

cost_matrix = cdist(detected_centroids, gt_centroids)
row_ind, col_ind = linear_sum_assignment(cost_matrix)
```

Missed cubes (unmatched GT) receive `miss_penalty_m = 0.50 m`. Extra detections (false positives) are counted separately but do not inflate mean error.

---

## 8. Layer 6 — RL Observation Interface

### File
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/ur3e_mujoco_env.py`

### Observation Space (13D)

The perception output is consumed as part of a normalised 13-dimensional state vector:

| Index | Content | Range | Source |
|---|---|---|---|
| 0–2 | EE position (normalised) | −1..+1 | MuJoCo FK |
| 3–5 | Cube position (normalised) | −1..+1 | **Perception centroid** |
| 6–8 | Bin position (normalised) | −1..+1 | Fixed/known |
| 9 | Grasped flag | 0 or 1 | EE within 3 cm + gripper closed |
| 10 | Gripper state | 0 or 1 | Actuator state |
| 11 | EE height (normalised) | 0..1 | ee_z / WORKSPACE_HEIGHT_M |
| 12 | Timestep (normalised) | 0..1 | step / MAX_EPISODE_STEPS |

Indices 3–5 map directly to `centroids[0]` from `detect_cubes.py` after normalisation against workspace bounds. The normalisation scheme matches `ros_interface.py` exactly, enabling sim-to-real weight transfer.

### Cube State in MuJoCo (Training)

During training, ground truth cube position is read directly from MuJoCo (`mjData.xpos`). At deployment, this slot is replaced by the DBSCAN centroid from the real D435i.

---

## 9. Configuration Reference

All perception parameters live in a single file:

`ros2_ws/src/holoassist_sim/config/sim_params.yaml`

```yaml
camera:
  name: d435i
  pose: [-0.6, 0.0, 0.85, 0.0, 0.5281, 0.0]   # [x,y,z,roll,pitch,yaw]
  topic: /camera
  update_rate: 30
  width: 848
  height: 480
  horizontal_fov_deg: 87
  near_clip: 0.105
  far_clip: 10.0
  imu:
    enabled: true
    update_rate: 200
    topic: /imu

capture:
  output_dir: ~/holoassist_pointclouds/
  description: default
  one_shot: true
  voxel_size: 0.0         # 0 = no downsampling

table:
  size: [1.0, 1.0, 0.5]
  pose: [0.0, 0.0, 0.25, 0, 0, 0]

cubes:
  - name: cube_red
    size: [0.04, 0.04, 0.04]
    pose: [0.10, 0.10, 0.52, 0, 0, 0]
    color: [0.9, 0.1, 0.1, 1.0]
    mass: 0.05
  # ... (cube_green, cube_blue, cube_yellow)
```

**Clustering parameters** (hardcoded in `clustering/detect_cubes.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `z_min` | `table_top_z + 0.015` | 3σ noise margin above table |
| `z_max` | `table_top_z + cube_height + 0.01` | Top of cube layer |
| `eps` | `0.015` m | DBSCAN neighbourhood radius |
| `min_samples` | `20` | DBSCAN core point threshold |
| `nb_neighbors` | `20` | Statistical outlier removal |
| `std_ratio` | `2.0` | Outlier removal threshold |
| `MIN_CLUSTER_PTS` | `50` | Minimum valid cluster size |
| `MAX_CLUSTER_PTS` | `1500` | Maximum valid cluster size |

---

## 10. How to Run

### Prerequisites

```bash
# ROS 2 environment
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash

# Clustering environment
cd clustering
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step A: Launch Simulation

```bash
ros2 launch holoassist_sim sim.launch.py
# Optional: override params file
ros2 launch holoassist_sim sim.launch.py params_file:=/abs/path/to/custom.yaml
```

Starts Gazebo, the ROS-Gazebo bridge, static TF publishers, RViz, scene_controller, and rqt_reconfigure.

### Step B: Capture a Single Point Cloud

```bash
ros2 run holoassist_sim save_pointcloud.py --params \
  ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

Output: `~/holoassist_pointclouds/default_4cubes_40mm_v001.ply`

### Step C: View the Raw Point Cloud

```bash
cd clustering && source .venv/bin/activate
python view_ply.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
```

Opens an interactive Polyscope window. Point cloud is transformed to world frame. Optional height-scalar colouring.

### Step D: Detect Cubes

```bash
cd clustering && source .venv/bin/activate
python detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
```

Prints centroid table and state vector. Opens Polyscope showing clustered cube layer with PCA axes.

### Step E: Generate Full Labelled Dataset

```bash
# With simulation running (Step A):
ros2 run holoassist_sim dataset_capture.py
```

Writes 60 scenes to `~/holoassist_dataset/`.

### Step F: Evaluate Detection Accuracy

```bash
cd clustering && source .venv/bin/activate
python verify_detection.py
```

Reads `~/holoassist_dataset/`, runs full detection pipeline, prints per-split accuracy report, writes `accuracy_report.json`.

---

## 11. Validated Performance

Baseline evaluation over 60 scenes (2–4 cubes, 40 mm fixed size):

| Metric | Value |
|---|---|
| Train scenes | 50 |
| Val scenes | 10 |
| Exact count accuracy | **100%** (60/60) |
| Cube recall | **100%** (185/185) |
| Mean centroid error | **1.63 cm** |
| Std deviation | 0.04 cm |
| Worst single scene | 1.71 cm |
| Target threshold | 3.0 cm |
| **Verdict** | **PASS** |

Error is within 3× the depth sensor noise floor (σ=0.005 m), consistent with the expected lower bound for this sensor model.

---

## 12. Short Summary Extract

A simulated Intel D435i RGBD camera is mounted at `[-0.6, 0.0, 0.85]` metres, tilted 30° downward over a tabletop scene in Gazebo Fortress. It publishes a PointCloud2 on `/camera/points` at 30 Hz with Gaussian depth noise σ=0.005 m. A ROS 2 node (`save_pointcloud.py`) subscribes to this topic, unpacks the point cloud, optionally voxel-downsamples it, and writes a binary PLY file. A separate dataset node (`dataset_capture.py`) automates 60 labelled captures by randomising cube layouts via the scene controller and saving ground truth poses queried directly from Gazebo.

In an isolated Python 3.11 environment, `detect_cubes.py` processes each PLY in five stages: (1) apply the camera-to-world ZYX Euler rotation so the point cloud is in world frame; (2) Z-crop to the cube layer (table_top + 1.5 cm to table_top + cube_height + 1 cm); (3) remove statistical outliers (20-neighbour, 2σ threshold); (4) run DBSCAN (eps=1.5 cm, min_samples=20) to form per-cube clusters; (5) filter by cluster size [50–1500 pts] and compute the mean centroid of each surviving cluster. The output is a flat array of world-frame (x, y, z) centroids — one per detected cube — fed directly as the cube-position slots in the RL agent's 13-dimensional observation vector. Validated accuracy across 60 scenes is 1.63 cm mean centroid error with 100% cube recall, well within the 3 cm target.
