# HoloAssist-AI

Part of the **41118 Artificial Intelligence in Robotics** course — an AI extension to the RS2 HoloAssist project. The goal is to develop and validate point cloud clustering algorithms (K-means, PCA, DBSCAN) that can identify objects on a table from an RGB-D camera.

This repo contains:
- A Gazebo simulation of the physical test setup (table + cubes + RealSense D435i)
- Tools to capture point cloud datasets from the sim
- A reference pipeline for the clustering algorithms
- Utilities to validate results in [Polyscope](https://polyscope.run/)

---

## Getting started

### Prerequisites

You need these installed before anything else:

- **Ubuntu 22.04** (Jammy)
- **ROS 2 Humble** — [installation guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- **Python 3.12** — `sudo apt install python3.12 python3.12-venv`

### Setup

Clone the repo and run the setup script. It installs all ROS package dependencies, builds the workspace, and creates an isolated Python 3.12 environment for the clustering work.

```bash
git clone <repo-url> && cd HoloAssist-AI
chmod +x setup.sh && ./setup.sh
```

If you only need one side:

```bash
./setup.sh --ros-only   # ROS workspace only
./setup.sh --py-only    # Python 3.12 clustering venv only
```

---

## How it works

The project has two separate environments that communicate through files — this is intentional because ROS 2 Humble pins Python 3.10 and our clustering tools need 3.12.

```
[Gazebo sim]  →  /camera/points (ROS topic)  →  [save_pointcloud.py]  →  .ply files on disk
                                                                               ↓
                                                              [your clustering scripts]
                                                              scikit-learn · polyscope · open3d
```

**Sim side (Python 3.10):** Runs Gazebo with a table, four coloured cubes, and a simulated D435i camera. Publishes point clouds over ROS 2.

**Clustering side (Python 3.12):** Reads the saved PLY files. No ROS required. Run your scripts here with scikit-learn, open3d, and Polyscope.

---

## Running the simulation

```bash
# Terminal 1 — start the sim (Gazebo + RViz)
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py

# Terminal 2 — capture a point cloud snapshot
source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params $(ros2 pkg prefix holoassist_sim)/share/holoassist_sim/config/sim_params.yaml
```

This saves a file like `~/holoassist_pointclouds/default_4cubes_40mm_v001.ply`. The version number increments automatically so you never overwrite previous captures.

### Labelling your datasets

Before each capture run, set a short description in [`sim_params.yaml`](ros2_ws/src/holoassist_sim/config/sim_params.yaml):

```yaml
capture:
  description: "spread"   # → spread_4cubes_40mm_v001.ply
```

This makes it easy to manage multiple configurations without manually renaming files.

---

## Modifying the scene

All scene parameters live in one place:
[`ros2_ws/src/holoassist_sim/config/sim_params.yaml`](ros2_ws/src/holoassist_sim/config/sim_params.yaml)

You can change the table size, add or remove cubes, move the camera, adjust resolution, and configure capture behaviour — all from that file. The simulation world is regenerated from it automatically at every launch, so there's nothing else to edit.

Key parameters:

| What you want to change | YAML key |
|---|---|
| Table size / position | `table.size`, `table.pose` |
| Cube positions, colours, sizes | `cubes[*].pose`, `.color`, `.size` |
| Camera position and angle | `camera.pose` (body frame, +X = look direction) |
| Camera resolution | `camera.width`, `camera.height`, `camera.horizontal_fov_deg` |
| Where captures are saved | `capture.output_dir` |
| Dataset label | `capture.description` |

---

## Using captured clouds in clustering scripts

```bash
source clustering/.venv/bin/activate
python clustering/your_script.py
```

Quick example to load a cloud and run K-means:

```python
import numpy as np
import polyscope as ps
from sklearn.cluster import KMeans
from plyfile import PlyData

ply = PlyData.read("~/holoassist_pointclouds/default_4cubes_40mm_v001.ply")
v   = ply["vertex"]
xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1) / 255.0

labels = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(xyz)

ps.init()
pc = ps.register_point_cloud("scene", xyz)
pc.add_color_quantity("rgb", rgb, enabled=True)
pc.add_scalar_quantity("cluster", labels.astype(float), enabled=True)
ps.show()
```

For the full preprocessing → clustering → PCA pipeline, see [`rgbd_clustering_pipeline.md`](rgbd_clustering_pipeline.md).

---

## Manual build (without setup.sh)

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src --rosdistro humble -r -y
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash

python3.12 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install -r clustering/requirements.txt
```
