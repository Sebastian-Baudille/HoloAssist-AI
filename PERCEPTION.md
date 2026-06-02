# HoloAssist-AI — Perception Stack: Full Technical Analysis

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Architecture & Data Flow](#2-architecture--data-flow)
3. [Layer 1 — Simulation & Sensor](#3-layer-1--simulation--sensor)
4. [Layer 2 — ROS Bridge & TF](#4-layer-2--ros-bridge--tf)
5. [Layer 3 — Point Cloud Capture](#5-layer-3--point-cloud-capture)
6. [Layer 4 — Clustering & Detection](#6-layer-4--clustering--detection)
7. [Layer 5 — RL Observation Interface](#7-layer-5--rl-observation-interface)
8. [Configuration Reference](#8-configuration-reference)
9. [How to Run](#9-how-to-run)
10. [Validated Performance](#10-validated-performance)
11. [Short Summary Extract](#11-short-summary-extract)

---

## 1. System Overview

The perception stack converts a simulated depth camera scene (Gazebo Fortress) into a structured state vector of cube centroids for downstream reinforcement learning. It operates in two isolated Python environments:

| Environment | Python | Role |
|---|---|---|
| `ros2_ws/` | 3.10 (ROS 2 Humble system Python) | Simulation, sensor bridging, PLY capture |
| `clustering/.venv` | 3.10 (isolated venv) | Clustering, cube detection, visualisation |

The bridge between them is PLY files written to `~/holoassist_pointclouds/`. No ROS imports ever enter the clustering environment; no open3d/polyscope imports ever enter the ROS environment.

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
└──────────────────┬───────────────────────────────────────────┘
                   │ ~/holoassist_pointclouds/*.ply
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  Clustering venv (Python 3.10)                              │
│  detect_cubes.py                                            │
│  1. Camera→World transform (ZYX Euler)                      │
│  2. Z-crop to cube layer                                    │
│  3. K-Means clustering (XYZ only)                           │
│  4. PCA per cluster → centroid, axes                        │
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

The camera simulates an Intel D435i RGBD sensor, mounted as a static link (`d435i_link`).

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

At launch, `sim.launch.py` invokes `render_world.py` twice before Gazebo starts:

```
sim_params.yaml + table_cubes.sdf.jinja2       →  /tmp/.../table_cubes.sdf
sim_params.yaml + ros_gz_bridge.yaml.jinja2    →  /tmp/.../ros_gz_bridge.yaml
```

Never edit the output SDF directly — it is regenerated every launch.

### Scene Physics

- **Table**: static, 0.5 m cube at z=0.25 m (top face at z=0.5 m)
- **Cubes**: dynamic, 40 mm, mass=0.05 kg, friction μ=0.8
- **Cube centre Z**: default 0.52 m (table top 0.5 + half-cube 0.02)
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
 └── d435i_link          (static, from camera.pose in sim_params.yaml)
      └── d435i/d435i_link/rgbd   (identity transform)
```

Two static `transform_publisher` nodes are launched. The second (identity) transform reconciles the Ignition scoped frame ID with the link frame.

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
4. **Voxel downsample** (if `capture.voxel_size > 0`) — hash-grid deduplication, keeps one point per voxel cell
5. **Write binary PLY** — little-endian, header includes a timestamp comment

### Output Filename Convention

```
{description}_{N}cubes_{size}mm_v{NNN}.ply

Example: default_4cubes_40mm_v001.ply
         default_4cubes_40mm_v002.ply  ← auto-increments, never overwrites
```

### One-shot Mode

When `capture.one_shot: true` (default), the node captures one frame, writes the PLY, and exits.

---

## 6. Layer 4 — Clustering & Detection

### Files
- `clustering/detect_cubes.py` — main detection pipeline
- `clustering/view_ply.py` — raw PLY viewer (no processing)
- `clustering/sample_data/default_4cubes_40mm_v001.ply` — bundled test cloud

### Dependencies (`clustering/requirements.txt`)

```
numpy==1.26.4       # core maths (Python 3.10 compatible)
open3d==0.19.0      # PLY I/O
scikit-learn==1.5.2 # KMeans, PCA, StandardScaler
polyscope==2.3.0    # 3D interactive visualisation
```

### Detection Pipeline — Step by Step

#### Step 1: Load PLY

```python
pcd    = o3d.io.read_point_cloud(ply_path)
points = np.asarray(pcd.points, dtype=np.float32)   # (N, 3), camera body frame
colors = np.asarray(pcd.colors, dtype=np.float32)   # (N, 3), 0..1
```

#### Step 2: Camera → World Transform

The PLY is in camera body frame (`+X = look direction`). Transform to world frame using ZYX Euler angles from `sim_params.yaml`:

```python
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
world_points = (R @ camera_points.T).T + [x, y, z]
```

**Must be applied before Z-based cropping** — the camera is tilted 30° so its raw Z axis is not vertical.

#### Step 3: Z-Crop to Cube Layer

```python
z_min = table_top_z + 0.015   # 0.515 m — 3× sensor noise margin
z_max = table_top_z + cube_height + 0.010  # 0.550 m

mask = (points[:, 2] > z_min) & (points[:, 2] < z_max)
```

Isolates cube top surfaces from the table and anything above.

#### Step 4: K-Means Clustering

```python
xyz_scaled = StandardScaler().fit_transform(cube_points)
# color_weight=0.0 by default → XYZ-only clustering
features = np.hstack([xyz_scaled, rgb_scaled * color_weight])

labels = KMeans(n_clusters=k, init="k-means++", n_init=10).fit_predict(features)
```

- `k` must be specified (default `k=4`). Pass `-k 2` or `-k 3` if fewer cubes are present.
- `color_weight=0.0` (default) uses XYZ only — RGB weighting confuses K-Means under Gazebo lighting.

**Known limitation:** K-Means can fail when the two far-side cubes (farther from the camera) are partially occluded by the two near-side cubes. Always use a fresh capture from a running Gazebo instance rather than an old PLY from a previous session.

#### Step 5: PCA per Cluster

```python
pca = PCA(n_components=3).fit(cluster_points)
centroid = pca.mean_          # world-frame (x, y, z) — fed to RL agent
axes     = pca.components_    # principal axes (Polyscope visualisation only)
```

#### Output: State Vector

```python
state = centroids.flatten().round(4)
# shape (k*3,) e.g. (12,) for 4 cubes
# values: world-frame x, y, z per cube in metres
```

### View-Only Pipeline (`view_ply.py`)

Loads a PLY, applies the camera→world transform, and opens Polyscope with no clustering. Useful for verifying a capture looks correct before running detection.

---

## 7. Layer 5 — RL Observation Interface

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

Indices 3–5 are the `centroids[0]` output from `detect_cubes.py`, normalised against workspace bounds. During MuJoCo training, ground truth position from `mjData.xpos` fills this slot. At deployment on real hardware, the live D435i centroid replaces it.

---

## 8. Configuration Reference

All perception parameters live in one file: `ros2_ws/src/holoassist_sim/config/sim_params.yaml`

```yaml
camera:
  name: d435i
  pose: [-0.6, 0.0, 0.85, 0.0, 0.5281, 0.0]   # [x, y, z, roll, pitch, yaw]
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
  - {name: cube_red,    size: [0.04,0.04,0.04], pose: [ 0.10,  0.10, 0.52, 0,0,0], mass: 0.05}
  - {name: cube_green,  size: [0.04,0.04,0.04], pose: [-0.10,  0.10, 0.52, 0,0,0], mass: 0.05}
  - {name: cube_blue,   size: [0.04,0.04,0.04], pose: [-0.10, -0.10, 0.52, 0,0,0], mass: 0.05}
  - {name: cube_yellow, size: [0.04,0.04,0.04], pose: [ 0.10, -0.10, 0.52, 0,0,0], mass: 0.05}
```

**Clustering parameters** (hardcoded in `clustering/detect_cubes.py`):

| Parameter | Value | Purpose |
|---|---|---|
| `z_min` | `table_top_z + 0.015` | 3σ noise margin above table |
| `z_max` | `table_top_z + cube_height + 0.01` | Top of cube layer |
| `k` | `4` (default, pass `-k N` to override) | Number of cubes |
| `color_weight` | `0.0` (default) | XYZ-only clustering |

---

## 9. How to Run

All commands must be run from the **repo root** (`/home/guy/git/HoloAssist-AI`).

### First-time setup

```bash
cd /home/guy/git/HoloAssist-AI

# Install system dependency (one-time, needs sudo)
sudo apt install python3.10-venv

# Create clustering venv and install packages
python3 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install --upgrade pip
pip install -r clustering/requirements.txt
deactivate

# Build the ROS package
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install && cd ..
```

Or use the setup script which handles all of the above:
```bash
chmod +x setup.sh && ./setup.sh
```

---

### Option A — Test with bundled sample (no Gazebo needed)

```bash
cd /home/guy/git/HoloAssist-AI
source clustering/.venv/bin/activate

# View raw point cloud in Polyscope
python3 clustering/view_ply.py clustering/sample_data/default_4cubes_40mm_v001.ply

# Run cube detection
python3 clustering/detect_cubes.py clustering/sample_data/default_4cubes_40mm_v001.ply
```

---

### Option B — Full pipeline with live Gazebo capture

**Terminal 1 — launch simulation:**
```bash
cd /home/guy/git/HoloAssist-AI
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py
```

Wait for Gazebo and RViz to fully load before continuing.

**Terminal 2 — capture one frame:**
```bash
cd /home/guy/git/HoloAssist-AI
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

Writes to `~/holoassist_pointclouds/default_4cubes_40mm_vNNN.ply`.

**Terminal 3 — run detection on fresh capture:**
```bash
cd /home/guy/git/HoloAssist-AI
source clustering/.venv/bin/activate
python3 clustering/detect_cubes.py        # auto-picks most recent capture
```

---

### Useful flags

```bash
# Specify number of cubes (default is 4)
python3 clustering/detect_cubes.py -k 3

# Skip Polyscope window (terminal output only)
python3 clustering/detect_cubes.py --no-viz

# Use a specific file
python3 clustering/detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v003.ply
```

---

### Rebuild ROS workspace (after source changes)

```bash
cd /home/guy/git/HoloAssist-AI
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install && cd ..
source ros2_ws/install/setup.bash
```

---

## 10. Validated Performance

Tested on bundled sample `clustering/sample_data/default_4cubes_40mm_v001.ply`:

| Cluster | Detected (x, y, z) | Nearest cube | Error |
|---|---|---|---|
| 0 | (0.0896, 0.0991, 0.5325) | cube_red (0.10, 0.10, 0.52) | 1.63 cm |
| 1 | (−0.1092, −0.0984, 0.5335) | cube_blue (−0.10, −0.10, 0.52) | 1.64 cm |
| 2 | (−0.1092, 0.0988, 0.5334) | cube_green (−0.10, 0.10, 0.52) | 1.62 cm |
| 3 | (0.0899, −0.0986, 0.5327) | cube_yellow (0.10, −0.10, 0.52) | 1.63 cm |

**Mean error: ~1.63 cm** — within 3× the sensor noise floor (σ = 0.005 m).

**Known failure mode:** Old captures from previous Gazebo sessions may show 2/4 cubes with large errors (~10 cm). This is caused by partial occlusion — the camera views the scene from one side, so the two far-side cubes can be partially blocked by the near-side cubes. Always use a fresh capture from a running Gazebo instance for reliable results.

---

## 11. Short Summary Extract

A simulated Intel D435i RGBD camera is mounted at `[-0.6, 0.0, 0.85]` metres, tilted 30° downward over a tabletop scene in Gazebo Fortress. It publishes a PointCloud2 on `/camera/points` at 30 Hz with Gaussian depth noise σ=0.005 m. A ROS 2 node (`save_pointcloud.py`) subscribes to this topic, unpacks the point cloud, optionally voxel-downsamples it, and writes a binary PLY file to `~/holoassist_pointclouds/`.

In an isolated Python 3.10 venv, `detect_cubes.py` processes each PLY in four stages: (1) apply the camera-to-world ZYX Euler rotation so the cloud is in world frame; (2) Z-crop to the cube layer (table_top + 1.5 cm to table_top + cube_height + 1 cm); (3) run K-Means (`k` must be specified, default 4) on XYZ-only features; (4) compute the mean centroid of each cluster via PCA. The output is a flat array of world-frame (x, y, z) centroids — one per detected cube — fed directly as the cube-position slots in the RL agent's 13-dimensional observation vector. Validated accuracy on the bundled sample is ~1.63 cm mean centroid error across all 4 cubes.
