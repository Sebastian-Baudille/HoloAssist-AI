# HoloAssist-AI

Part of the **41118 Artificial Intelligence in Robotics** course — an AI extension to the RS2 HoloAssist project. The goal is to develop and validate point cloud clustering algorithms that can identify objects on a table from an RGB-D camera, producing cube centroid positions for a downstream RL agent.

---

## Overview

```
[Gazebo sim]  →  /camera/points (ROS 2)  →  save_pointcloud.py  →  .ply files
                                                                          ↓
                                                               detect_cubes.py
                                                               (DBSCAN pipeline)
                                                                          ↓
                                                               centroid array → RL agent
```

Two isolated Python environments communicate through PLY files on disk:

| Environment | Python | Role |
|---|---|---|
| `ros2_ws/` | 3.10 (ROS 2 Humble) | Gazebo sim, camera bridge, PLY capture, scene control |
| `clustering/.venv` | 3.10 (isolated venv) | DBSCAN detection, visualisation, accuracy evaluation |

---

## Setup

### Prerequisites

- Ubuntu 22.04
- ROS 2 Humble — [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- `sudo apt install python3.10-venv python3-pip`

### Install

```bash
git clone <repo-url> && cd HoloAssist-AI
chmod +x setup.sh && ./setup.sh
```

Or manually:

```bash
# ROS workspace
source /opt/ros/humble/setup.bash
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install && cd ..

# Clustering venv
python3 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install -r clustering/requirements.txt
```

---

## Running the full pipeline

All commands run from the repo root. Never mix `source clustering/.venv/bin/activate` and ROS commands in the same terminal.

### Terminal 1 — Gazebo simulation

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py
```

### Terminal 2 — Scene controller

Manages cube randomisation in the running sim without restarting.

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run holoassist_sim scene_controller.py --ros-args \
    -p params_file:=$(pwd)/ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

Wait for: `scene_controller ready.`

### Terminal 3 — Randomise, capture, repeat

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash

# Randomise cube positions in the running sim
ros2 service call /scene/randomize_cubes std_srvs/srv/Trigger

# Capture one point cloud frame → PLY
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

### Terminal 4 — Detect cubes

```bash
source clustering/.venv/bin/activate
python3 clustering/detect_cubes.py         # auto-picks latest capture
python3 clustering/detect_cubes.py --no-viz  # suppress Polyscope window
```

---

## DBSCAN detection pipeline

`clustering/detect_cubes.py` converts a raw PLY into world-frame cube centroids in five stages:

### 1. Camera → world transform

The PLY is in camera body frame (`+X = look direction`). A ZYX Euler rotation from `sim_params.yaml` converts it to world frame:

```python
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
world_points = (R @ camera_points.T).T + [x, y, z]
```

This must happen before any Z-based cropping since the camera is not axis-aligned.

### 2. Z-crop to cube layer

```python
z_min = table_top_z + 0.015   # 3× sensor noise margin (σ = 0.005 m)
z_max = table_top_z + cube_height + 0.010
```

Slices out only the points on top of the cubes, discarding the table surface and everything above.

### 3. Statistical outlier removal

```python
pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
```

Removes points whose mean distance to 20 nearest neighbours exceeds `mean + 2σ`. Kills flying pixels and depth-discontinuity artefacts at cube edges before clustering.

### 4. DBSCAN clustering

```python
labels = DBSCAN(eps=0.015, min_samples=20).fit_predict(clean_points)
# -1 = noise,  0..k-1 = cluster IDs
```

- **`eps=0.015 m`** — neighbourhood radius. Cubes are enforced to be ≥ 8 cm apart (2× cube size), so DBSCAN never bridges two separate cubes.
- **`min_samples=20`** — minimum points to form a cluster core. Rejects sparse noise patches.
- **No fixed k** — DBSCAN finds however many clusters are present automatically.

### 5. Size filter + centroid

```python
valid = [c for c in clusters if 50 <= point_count(c) <= 2000]
centroid = cluster_points.mean(axis=0)   # world-frame (x, y, z)
```

Rejects clusters that are too small (noise) or too large (merged/non-cube objects). The mean of each surviving cluster is the cube's world position, fed directly to the RL agent.

### Why DBSCAN over K-Means

| | K-Means | DBSCAN |
|---|---|---|
| Requires knowing k | Yes | No |
| Handles noise | No — forces every point into a cluster | Yes — marks noise as −1 |
| Sensitivity to unequal densities | High | Low |
| Cluster shape assumption | Spherical | None |

K-Means was tried first and failed on random scenes because: it requires knowing the cube count upfront, and with variable-density clusters (the overhead camera gives near-side cubes more points than far-side) it was merging two cubes into one cluster.

---

## Camera setup

The camera simulates an Intel D435i RGBD sensor mounted directly above the table:

```yaml
camera:
  pose: [0.0, 0.0, 1.2, 0.0, 1.5708, 0.0]  # x y z roll pitch yaw
  # pitch = π/2 → straight down
```

**Why overhead?** Earlier configurations used a 59° oblique angle which caused two systematic failures:
1. Near-grazing incidence on the table surface (~75° from normal) producing thousands of ghost points just above the table in the Z-crop window.
2. Systematic depth bias for cubes on the far side of the table, placing their point clouds at the wrong world Z.

A true overhead camera eliminates both: all cubes are at the same distance from the lens (~0.66 m), depth measurements are perpendicular to the table, and the depth noise maps uniformly to world-Z noise (σ = 0.005 m).

---

## Scene randomisation

### Live (sim running)

The scene controller randomises cube positions in a running Gazebo sim via a ROS service — no restart required:

```bash
ros2 service call /scene/randomize_cubes std_srvs/srv/Trigger
ros2 service call /scene/reset std_srvs/srv/Trigger
```

Constraints enforced: cubes are placed within ±18 cm of table centre, all at 40 mm size, minimum 10 cm centre-to-centre separation (2.5× cube size). This guarantees DBSCAN can always separate adjacent clusters (4 cm surface gap >> 1.5 cm DBSCAN `eps`).

### Offline (update YAML for next launch)

```bash
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py           # 4 cubes
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py -k 3      # 3 cubes
python3 ros2_ws/src/holoassist_sim/scripts/randomize_scene.py --preview  # dry run
```

Writes new cube positions into `sim_params.yaml`. Relaunch the sim to apply.

---

## Dataset generation and accuracy evaluation

### Generate a labelled dataset

With Terminal 1 (sim) and Terminal 2 (scene_controller) running:

```bash
source /opt/ros/humble/setup.bash && source ros2_ws/install/setup.bash
ros2 run holoassist_sim dataset_capture.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml \
    --scenes 60 --output ~/holoassist_dataset
```

For each scene this:
1. Randomises the scene (2–4 cubes)
2. Waits 1 s for physics to settle
3. Queries actual settled cube poses from Gazebo
4. Captures one PLY frame
5. Writes `scene_NNNN.ply` + `scene_NNNN.labels.json` with ground truth positions

### Evaluate detection accuracy

```bash
source clustering/.venv/bin/activate
python3 clustering/verify_detection.py \
    --dataset ~/holoassist_dataset \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

Runs the full DBSCAN pipeline on every PLY, matches detections to ground truth using the **Hungarian algorithm** (handles variable detected count vs ground truth k), and reports:

- Mean / std / max centroid error per split
- Exact count accuracy (detected k == ground truth k)
- Cube recall (matched cubes / total ground truth cubes)
- Per-k breakdown (k=2, k=3, k=4)
- Full per-scene results in `accuracy_report.json`

### Validated results

60 scenes, 2–4 cubes per scene, 40 mm fixed size, overhead camera:

| Split | Scenes | Mean error | Std | Exact count | Recall |
|---|---|---|---|---|---|
| Train | 50 | **1.87 cm** | 0.02 cm | 100% | 100% |
| Val | 10 | **1.87 cm** | 0.03 cm | 100% | 100% |

Error is within 4× the sensor noise floor (σ = 0.005 m). Consistent across k=2, k=3, and k=4 scenes.

---

## Configuration reference

All perception parameters live in one file: `ros2_ws/src/holoassist_sim/config/sim_params.yaml`

| What to change | YAML key |
|---|---|
| Camera position / angle | `camera.pose` (body frame, +X = look direction) |
| Camera resolution / FOV | `camera.width`, `camera.height`, `camera.horizontal_fov_deg` |
| Table size / position | `table.size`, `table.pose` |
| Default cube layout | `cubes[*].pose`, `.color`, `.size` |
| Capture output directory | `capture.output_dir` |
| Scene label (filename prefix) | `capture.description` |

DBSCAN parameters are in `clustering/detect_cubes.py`:

| Parameter | Value | Purpose |
|---|---|---|
| `eps` | 0.015 m | Neighbourhood radius |
| `min_samples` | 20 | Core point threshold |
| `MIN_CLUSTER_PTS` | 50 | Minimum valid cluster size |
| `MAX_CLUSTER_PTS` | 2000 | Maximum valid cluster size |
| `z_min` | `table_top + 0.015` | Bottom of cube layer crop |
| `z_max` | `table_top + cube_height + 0.010` | Top of cube layer crop |
