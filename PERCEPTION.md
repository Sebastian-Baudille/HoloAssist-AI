# HoloAssist-AI — Perception Stack: Full Technical Analysis

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Layer 1 — Simulation & Sensor](#3-layer-1--simulation--sensor)
4. [Layer 2 — ROS Bridge & TF](#4-layer-2--ros-bridge--tf)
5. [Layer 3 — Point Cloud Capture](#5-layer-3--point-cloud-capture)
6. [Layer 4 — Scene Randomisation](#6-layer-4--scene-randomisation)
7. [Layer 5 — DBSCAN Detection Pipeline](#7-layer-5--dbscan-detection-pipeline)
8. [Layer 6 — Dataset Generation & Verification](#8-layer-6--dataset-generation--verification)
9. [Layer 7 — RL Observation Interface](#9-layer-7--rl-observation-interface)
10. [Configuration Reference](#10-configuration-reference)
11. [How to Run](#11-how-to-run)
12. [Validated Performance](#12-validated-performance)
13. [Design Decisions & Failure History](#13-design-decisions--failure-history)
14. [Short Summary Extract](#14-short-summary-extract)

---

## 1. System Overview

The perception stack converts a simulated depth camera scene (Gazebo Fortress) into a structured array of cube centroids for downstream reinforcement learning. It operates across two isolated Python environments that communicate only through PLY files on disk.

| Environment | Python | Role |
|---|---|---|
| `ros2_ws/` | 3.10 (ROS 2 Humble) | Gazebo sim, sensor bridging, PLY capture, scene control |
| `clustering/.venv` | 3.10 (isolated venv) | DBSCAN detection, visualisation, accuracy evaluation |

No ROS imports ever enter the clustering environment. No open3d/polyscope imports ever enter the ROS environment.

---

## 2. Architecture & Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Gazebo Ignition Fortress                                        │
│  Overhead RGBD camera → /camera/points (PointCloud2, body frame) │
└──────────────────────┬──────────────────────────────────────────┘
                       │ ros_gz_bridge
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  ROS 2 (Python 3.10)                                            │
│  scene_controller.py — randomise cube positions via ign service  │
│  save_pointcloud.py  — subscribe, unpack, write binary PLY       │
│  dataset_capture.py  — automate N scenes with ground-truth labels│
└──────────────────────┬──────────────────────────────────────────┘
                       │ ~/holoassist_pointclouds/*.ply
                       │ ~/holoassist_dataset/scene_NNNN.{ply,labels.json}
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│  Clustering venv (Python 3.10)                                   │
│  detect_cubes.py                                                 │
│    1. Camera → World transform (ZYX Euler from sim_params.yaml)  │
│    2. Z-crop to cube layer                                       │
│    3. Statistical outlier removal (Open3D)                       │
│    4. DBSCAN clustering                                          │
│    5. Size filter                                                │
│    6. Mean centroid per cluster                                  │
│    → centroids.flatten() → RL agent state vector                │
│                                                                  │
│  verify_detection.py — Hungarian-matched accuracy evaluation     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Layer 1 — Simulation & Sensor

### Files
- `ros2_ws/src/holoassist_sim/config/sim_params.yaml` — single source of truth
- `ros2_ws/src/holoassist_sim/worlds/table_cubes.sdf.jinja2` — Jinja2 world template
- `ros2_ws/src/holoassist_sim/scripts/render_world.py` — template renderer
- `ros2_ws/src/holoassist_sim/launch/sim.launch.py` — orchestration

### Camera Model

The camera simulates an Intel D435i RGBD sensor mounted **directly above the table**, looking straight down.

```
Camera pose (sim_params.yaml):
  position:  x=0.0 m,  y=0.0 m,  z=1.2 m
  rotation:  roll=0.0,  pitch=π/2 (1.5708 rad),  yaw=0.0

  → directly above the table centre
  → pitch=π/2 points camera +X axis straight down (−Z world)
  → all cubes at ~0.66 m from lens — uniform depth accuracy
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

### Why Overhead?

Two earlier configurations were tried before settling on the overhead position. See [Section 13](#13-design-decisions--failure-history) for the full failure history.

### World Template Rendering

At launch, `sim.launch.py` invokes `render_world.py` twice before Gazebo starts:

```
sim_params.yaml + table_cubes.sdf.jinja2       →  /tmp/.../table_cubes.sdf
sim_params.yaml + ros_gz_bridge.yaml.jinja2    →  /tmp/.../ros_gz_bridge.yaml
```

Never edit the output SDF directly — it is regenerated every launch.

### Scene Physics

- **Table**: static, 0.5 m × 0.5 m × 0.5 m, top face at Z = 0.5 m
- **Cubes**: dynamic, 40 mm, mass = 0.05 kg, friction μ = 0.8
- **Cube centre Z**: 0.52 m (table top 0.5 + half-cube 0.02)
- **Physics step**: 0.001 s, real-time factor 1.0

---

## 4. Layer 2 — ROS Bridge & TF

### File
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
 └── d435i_link          (static — from camera.pose in sim_params.yaml)
      └── d435i/d435i_link/rgbd   (identity transform)
```

Two static `transform_publisher` nodes are launched. The second (identity) transform reconciles the Ignition scoped frame ID with the link frame.

---

## 5. Layer 3 — Point Cloud Capture

### File
- `ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py`

### What It Does

Subscribes to `/camera/points` (PointCloud2, BEST_EFFORT QoS, depth=5). On each message:

1. **Unpack fields** — reads `x`, `y`, `z` plus optional `rgb` from PointCloud2 field descriptors
2. **RGB decode** — `rgb` is stored as a float32 bit-cast of uint32 `0x00RRGGBB`:
   ```python
   r = (rgb_int >> 16) & 0xFF
   g = (rgb_int >>  8) & 0xFF
   b =  rgb_int        & 0xFF
   ```
3. **Drop non-finite** — removes rows where x/y/z are NaN or Inf
4. **Voxel downsample** (if `capture.voxel_size > 0`) — hash-grid deduplication
5. **Write binary PLY** — little-endian, with timestamp in header comment

### Output Filename Convention

```
{description}_{N}cubes_{size}mm_v{NNN}.ply

Example: topdown_4cubes_40mm_v001.ply
         topdown_4cubes_40mm_v002.ply  ← auto-increments, never overwrites
```

### One-shot Mode

When `capture.one_shot: true` (default), the node captures one frame, writes the PLY, and exits.

---

## 6. Layer 4 — Scene Randomisation

### Files
- `ros2_ws/src/holoassist_sim/scripts/scene_controller.py` — live ROS node
- `ros2_ws/src/holoassist_sim/scripts/randomize_scene.py` — offline YAML editor

### scene_controller.py

A ROS 2 node that runs alongside the sim and exposes two services:

| Service | Effect |
|---|---|
| `/scene/randomize_cubes` | Remove current cubes, spawn N new ones at random positions |
| `/scene/reset` | Remove current cubes, restore default layout from sim_params.yaml |

**Spawning logic:**
- Placement window: within ±18 cm of table centre
- Cube size: fixed at 40 mm (`cube_size_min = cube_size_max = 0.04`)
- Minimum centre-to-centre distance: 2.5 × cube size = **10 cm**
- This guarantees a ≥ 6 cm surface gap, well above DBSCAN `eps = 1.5 cm`

**First-call behaviour:** On the first `/scene/randomize_cubes` call, removes the static cubes baked into the SDF world file (`cube_red`, `cube_green`, etc.) before spawning random ones. Subsequent calls only remove previously spawned cubes. Tracked via `_sdf_defaults_cleared` flag.

**Mechanism:** Writes per-cube SDF to `/tmp/holoassist_*.sdf`, calls `ign service /world/{world}/create`. Removal calls `ign service /world/{world}/remove`.

### randomize_scene.py

Offline alternative — modifies `sim_params.yaml` directly for the next launch. Useful when the sim is not running.

```bash
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py           # 4 cubes
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py -k 3      # 3 cubes
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py --seed 42  # reproducible
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py --preview  # dry run
```

---

## 7. Layer 5 — DBSCAN Detection Pipeline

### File
- `clustering/detect_cubes.py`

### Dependencies

```
numpy==1.26.4       # Python 3.10 compatible
open3d==0.19.0      # PLY I/O, statistical outlier removal
scikit-learn==1.5.2 # DBSCAN, PCA
polyscope==2.3.0    # 3D interactive visualisation
```

### Step 1 — Load PLY

```python
pcd    = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points, dtype=np.float32)   # (N, 3), camera body frame
colors = np.asarray(pcd.colors, dtype=np.float32)   # (N, 3), 0..1
```

### Step 2 — Camera → World Transform

The PLY is in camera body frame (`+X = look direction`). Convert to world frame using ZYX Euler angles from `sim_params.yaml`:

```python
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
world_points = (R @ camera_points.T).T + [x, y, z]
```

With the overhead camera (pitch = π/2), this is equivalent to:
- camera `+X` (depth axis) → world `−Z` (down)
- camera `+Y` (horizontal) → world `+Y`
- camera `+Z` (up in body frame) → world `+X`

**Must be applied before Z-based cropping.**

### Step 3 — Z-Crop to Cube Layer

```python
z_min = table_top_z + 0.015   # 0.515 m — 3× sensor noise σ
z_max = table_top_z + cube_height + 0.010  # 0.550 m

mask = (world_z > z_min) & (world_z < z_max)
```

Isolates the cube top surfaces from the table (below) and anything above. The 3σ margin ensures table surface noise (σ = 0.005 m) is excluded.

### Step 4 — Statistical Outlier Removal

```python
pcd_layer.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

Removes any point whose mean distance to its 20 nearest neighbours exceeds `mean + 2σ` of the global neighbour-distance distribution. Eliminates flying pixels and depth-discontinuity artefacts at cube edges **before** clustering — these would otherwise form spurious small clusters or distort density estimates.

### Step 5 — DBSCAN Clustering

```python
labels = DBSCAN(eps=0.015, min_samples=20).fit_predict(clean_points)
# labels: −1 = noise,  0..k−1 = cluster IDs
```

**`eps = 0.015 m` (1.5 cm):**
The scene controller enforces ≥ 10 cm centre-to-centre separation between cubes, leaving a ≥ 6 cm surface gap. Since 6 cm >> 2 × eps = 3 cm, DBSCAN cannot bridge the gap between two separate cubes. Noise points need to form a continuous chain within 1.5 cm of each other to be merged into a cube cluster — the outlier removal step eliminates isolated noise, preventing this.

**`min_samples = 20`:**
A 40 mm cube at ~0.66 m from the overhead camera produces ~800–1600 points. A real cluster has orders of magnitude more points than a noise patch, so min_samples=20 is a very conservative threshold that cleanly rejects sparse artefacts.

**No fixed k:**
DBSCAN finds however many valid clusters exist without being told the cube count. This is the key advantage over K-Means, which would force exactly k clusters even when k was wrong.

### Step 6 — Size Filter

```python
MIN_CLUSTER_PTS = 50
MAX_CLUSTER_PTS = 2000

valid_ids = [c for c in clusters if MIN_CLUSTER_PTS <= count(c) <= MAX_CLUSTER_PTS]
```

A 40 mm cube at 0.66 m produces ~800–1600 points with the overhead camera. The bounds [50, 2000] reject:
- **Below 50**: isolated noise patches that survived outlier removal
- **Above 2000**: merged objects or large non-cube surfaces

Note: `MAX_CLUSTER_PTS = 2000` (not 1500 as in the original side-view pipeline). The overhead camera gives near-side cubes up to ~1600 points; 1500 would silently reject them.

### Step 7 — Centroid Output

```python
centroid = cluster_points.mean(axis=0)   # world-frame (x, y, z) in metres
```

PCA is also fitted per cluster. The mean (`pca.mean_`) equals the arithmetic centroid and is what feeds the RL state vector. PCA axes are used only for Polyscope visualisation (orientation arrows).

### Output — State Vector

```python
state = centroids.flatten().round(4)
# shape: (k*3,)  e.g. (12,) for 4 cubes
# values: world-frame x, y, z per cube in metres
```

### Why DBSCAN Over K-Means

| Property | K-Means | DBSCAN |
|---|---|---|
| Requires knowing k | Yes | No |
| Handles noise points | No — forces all into clusters | Yes — labels noise as −1 |
| Sensitive to unequal density | Yes — merges sparse clusters | No |
| Cluster shape assumption | Spherical (centroid-based) | Arbitrary (density-based) |

K-Means was the original algorithm. It failed on random scenes because:
1. It required knowing the cube count `k` upfront
2. The overhead camera gives near-side cubes ~50% more points than far-side cubes; `StandardScaler` then shifted the feature mean, causing K-Means to merge the two sparser cubes into one cluster

Removing `StandardScaler` fixed the density problem, but the need to specify `k` remained a fundamental limitation for variable-count scenes.

---

## 8. Layer 6 — Dataset Generation & Verification

### Files
- `ros2_ws/src/holoassist_sim/scripts/dataset_capture.py`
- `clustering/verify_detection.py`

### dataset_capture.py

Automates generation of N labelled scenes (default 60: 50 train + 10 val). Per scene:

1. Set `cube_count` parameter on scene_controller (random 2–4)
2. Call `/scene/randomize_cubes`
3. Wait 1.0 s for physics to settle
4. Query settled cube poses from Gazebo:
   ```bash
   ign topic -e -n 1 -t /world/{world}/dynamic_pose/info
   ```
   Parses protobuf text output with regex to extract x, y, z, qx, qy, qz, qw per cube
5. Subscribe to `/camera/points`, wait for next frame
6. Save PLY
7. Write `labels.json` with ground truth:

```json
{
  "scene_id": "scene_0001",
  "split": "train",
  "cube_count": 3,
  "table_top_z": 0.5,
  "cubes": [
    {
      "name": "cube_rand_00",
      "position": {"x": -0.135, "y": 0.045, "z": 0.520},
      "orientation": {"x": 0, "y": 0, "z": 0.81, "w": 0.59},
      "size_m": 0.04
    }
  ]
}
```

### verify_detection.py

Runs the full DBSCAN pipeline on every `scene_NNNN.ply` in the dataset and evaluates against ground truth.

**Matching algorithm — Hungarian (linear sum assignment):**

```python
cost_matrix = cdist(detected_centroids, gt_centroids)   # (n_det, k_gt)
row_idx, col_idx = linear_sum_assignment(cost_matrix)
```

Handles variable detected count vs ground truth k:
- **Missed cubes** (detected < GT): unmatched GT cubes receive `miss_penalty = 0.50 m` so misses are clearly visible in the mean error
- **False positives** (detected > GT): extra clusters are counted separately but do not inflate mean centroid error

**Reported metrics:**

| Metric | Description |
|---|---|
| Mean centroid error | Average Euclidean distance per GT cube including miss penalty |
| Std dev | Scene-to-scene consistency |
| Exact count accuracy | % scenes where detected k == GT k |
| Cube recall | matched cubes / total GT cubes |
| Per-k breakdown | Accuracy split by GT cube count (k=2, 3, 4) |

Full per-scene results are written to `~/holoassist_dataset/accuracy_report.json`.

---

## 9. Layer 7 — RL Observation Interface

### File
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/ur3e_mujoco_env.py`

### Observation Space (13D)

| Index | Content | Range | Source |
|---|---|---|---|
| 0–2 | EE position (normalised) | −1..+1 | MuJoCo FK |
| 3–5 | Cube position (normalised) | −1..+1 | **Perception centroid** |
| 6–8 | Bin position (normalised) | −1..+1 | Fixed/known |
| 9 | Grasped flag | 0 or 1 | EE within 3 cm + gripper closed |
| 10 | Gripper state | 0 or 1 | Actuator state |
| 11 | EE height (normalised) | 0..1 | ee_z / WORKSPACE_HEIGHT_M |
| 12 | Timestep (normalised) | 0..1 | step / MAX_EPISODE_STEPS |

Indices 3–5 are `centroids[0]` from `detect_cubes.py` normalised against workspace bounds. During MuJoCo training, ground truth position from `mjData.xpos` fills this slot. At deployment on real hardware, the live D435i centroid replaces it — the normalisation is identical in both cases.

---

## 10. Configuration Reference

`ros2_ws/src/holoassist_sim/config/sim_params.yaml` — single source of truth for all perception parameters.

```yaml
camera:
  pose: [0.0, 0.0, 1.2, 0.0, 1.5708, 0.0]  # [x, y, z, roll, pitch, yaw]
  update_rate: 30.0
  width: 848
  height: 480
  horizontal_fov_deg: 87.0
  near_clip: 0.105
  far_clip: 10.0

capture:
  output_dir: ~/holoassist_pointclouds/
  description: topdown
  one_shot: true
  voxel_size: 0.0

table:
  size: [0.5, 0.5, 0.5]
  pose: [0.0, 0.0, 0.25, 0, 0, 0]

cubes:
  - {name: cube_red,    size: [0.04,0.04,0.04], pose: [0.10, 0.10, 0.52, 0,0,0]}
  - {name: cube_green,  size: [0.04,0.04,0.04], pose: [-0.10, 0.10, 0.52, 0,0,0]}
  - {name: cube_blue,   size: [0.04,0.04,0.04], pose: [-0.10,-0.10, 0.52, 0,0,0]}
  - {name: cube_yellow, size: [0.04,0.04,0.04], pose: [0.10,-0.10, 0.52, 0,0,0]}
```

**DBSCAN parameters** (in `clustering/detect_cubes.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `eps` | 0.015 m | Neighbourhood radius |
| `min_samples` | 20 | Core point threshold |
| `MIN_CLUSTER_PTS` | 50 | Reject clusters smaller than this |
| `MAX_CLUSTER_PTS` | 2000 | Reject clusters larger than this |
| `z_min` | `table_top + 0.015` | Bottom of cube layer crop (3σ above table) |
| `z_max` | `table_top + cube_height + 0.010` | Top of cube layer crop |

---

## 11. How to Run

All commands from repo root `/home/guy/git/HoloAssist-AI`. Never mix the clustering venv and ROS in the same terminal.

### First-time setup

```bash
sudo apt install python3.10-venv
python3 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install --upgrade pip && pip install -r clustering/requirements.txt
deactivate

source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install && cd ..
```

### Quick single-capture test (no randomisation)

```bash
# Terminal 1
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py

# Terminal 2
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml

# Terminal 3
source clustering/.venv/bin/activate
python3 clustering/detect_cubes.py --no-viz
```

### Full pipeline with randomisation

```bash
# Terminal 1 — sim (keep running throughout)
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py

# Terminal 2 — scene controller (keep running throughout)
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run holoassist_sim scene_controller.py --ros-args \
    -p params_file:=$(pwd)/ros2_ws/src/holoassist_sim/config/sim_params.yaml

# Terminal 3 — randomise then capture
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 service call /scene/randomize_cubes std_srvs/srv/Trigger
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml

# Terminal 4 — detect
source clustering/.venv/bin/activate
python3 clustering/detect_cubes.py --no-viz
```

### Dataset generation and evaluation

```bash
# Terminal 3 (with Terminals 1 and 2 running)
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run holoassist_sim dataset_capture.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml \
    --scenes 60 --output ~/holoassist_dataset

# Terminal 4
source clustering/.venv/bin/activate
python3 clustering/verify_detection.py \
    --dataset ~/holoassist_dataset \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

---

## 12. Validated Performance

60 scenes, 2–4 cubes per scene, 40 mm fixed size, overhead camera (0, 0, 1.2, pitch=π/2):

| Split | Scenes | GT cubes | Mean error | Std | Exact count | Recall |
|---|---|---|---|---|---|---|
| Train | 50 | 148 | **1.87 cm** | 0.02 cm | 100% (50/50) | 100% (148/148) |
| Val | 10 | 35 | **1.87 cm** | 0.03 cm | 100% (10/10) | 100% (35/35) |

Breakdown by cube count:

| k | Train scenes | Mean error |
|---|---|---|
| 2 | 17 | 1.87 cm |
| 3 | 18 | 1.87 cm |
| 4 | 15 | 1.87 cm |

Error is within 4× the depth sensor noise floor (σ = 0.005 m). The 0.02–0.03 cm standard deviation shows detection is consistent across the full range of randomised positions and cube counts.

---

## 13. Design Decisions & Failure History

### Camera angle progression

**Configuration 1 — Side view** (original, from `main` branch):
```
pose: [-0.6, 0.0, 0.85, 0.0, 0.5281, 0.0]   # 30° downward tilt
```
Failed because `cube_red` (+0.10, +0.10) is directly behind `cube_green` (−0.10, +0.10) from the camera's viewpoint — they share the same Y coordinate and different X. K-Means merged them into one cluster at Y≈0.

**Configuration 2 — Overhead oblique** (59°):
```
pose: [-0.30, 0.0, 1.00, 0.0, 1.0304, 0.0]
```
Fixed the inter-cube occlusion but introduced two new systematic failures:
1. **Ghost stripe**: near-grazing incidence (~75° from table normal) at X=−0.15 to −0.19 produced thousands of spurious depth points at Z=0.515–0.531, overwhelming DBSCAN with a single giant cluster spanning the full table width.
2. **Far-side depth bias**: cubes at positive X measured at incorrect world Z values (7–15 cm below true position), placing them outside the Z-crop window entirely.

These failures were invisible on the default fixed-position scenes but appeared with high frequency on randomised positions, causing 32 cm mean error and 0% recall in the 60-scene evaluation.

**Configuration 3 — True overhead** (current):
```
pose: [0.0, 0.0, 1.2, 0.0, 1.5708, 0.0]   # straight down
```
- Zero oblique angle → no ghost stripe, no depth bias
- All cubes at ~0.66 m from lens → uniform point density and depth accuracy
- 100% recall, 1.87 cm mean error across all 60 evaluation scenes

### Algorithm progression

**K-Means** (original):
- Required specifying `k` (cube count) before running
- `StandardScaler` applied to XYZ shifted the feature mean when cluster densities were unequal (near-side cubes had ~50% more points than far-side with the oblique camera), causing the two sparser cubes to merge
- Removing `StandardScaler` fixed the merge but the fixed-`k` requirement remained a problem for variable-count scenes

**DBSCAN** (current):
- Finds k automatically — no prior knowledge of cube count needed
- Marks noise as −1 rather than forcing it into clusters
- With overhead camera and enforced 10 cm minimum separation, `eps=0.015 m` cleanly separates all cubes while rejecting noise

---

## 14. Short Summary Extract

A simulated Intel D435i RGBD camera is mounted at (0, 0, 1.2) metres looking straight down (pitch = π/2) over a tabletop in Gazebo Fortress. It publishes a PointCloud2 on `/camera/points` at 30 Hz with Gaussian depth noise σ = 0.005 m. The overhead position ensures uniform depth accuracy across all cube positions and eliminates oblique-angle depth artefacts that caused failures with earlier angled configurations.

A ROS 2 node (`save_pointcloud.py`) captures frames to binary PLY files. A scene controller (`scene_controller.py`) randomises cube positions live in the running sim via `ign service` calls, enforcing ≥ 10 cm separation between cubes. A dataset capture node (`dataset_capture.py`) automates 60 labelled scenes with ground-truth positions queried directly from Gazebo.

In an isolated Python 3.10 venv, `detect_cubes.py` processes each PLY in five stages: (1) ZYX Euler rotation to world frame; (2) Z-crop to the cube layer (table_top + 1.5 cm to cube_top + 1 cm); (3) statistical outlier removal; (4) DBSCAN (eps = 1.5 cm, min_samples = 20) which finds clusters automatically without a fixed k; (5) size filter [50–2000 pts] to reject noise. The mean centroid of each surviving cluster is the cube's world-frame position, fed directly as the cube-position slots in the RL agent's 13-dimensional observation vector.

`verify_detection.py` evaluates accuracy using Hungarian matching between detected and ground-truth centroids. Validated across 60 scenes with 2–4 randomly placed 40 mm cubes: **1.87 cm mean centroid error, 100% exact count accuracy, 100% cube recall** on both train and val splits.
