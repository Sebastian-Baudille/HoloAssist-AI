# HoloAssist-AI

Part of the **41118 Artificial Intelligence in Robotics** course — an AI extension to the
RS2 HoloAssist project. End-to-end perception + control: a UR3e arm sees coloured cubes on
a tabletop with an RGB-D camera, clusters their 3D positions, and grasps them using a
PPO policy.

Built and validated in simulation (Gazebo Fortress + ROS 2 Humble), with a clear path to a
physical RealSense D435i and UR3e arm.

---

## Documentation map

| Document | Purpose |
|---|---|
| [ARCHITECTURE.md](ARCHITECTURE.md) | System design, data flow, stage statuses, file map |
| [LAUNCH.md](LAUNCH.md) | All run commands: sim, capture, clustering, RL training, deployment |
| [PROGRESS.md](PROGRESS.md) | Current section + training metrics snapshot |
| [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md) | Forward plan: port RL stack to Isaac Lab on Windows |
| [docs/clustering-pipeline.md](docs/clustering-pipeline.md) | Reference manual: RGB-D preprocessing, clustering, PCA, tuning |
| [clustering/README.md](clustering/README.md) | Clustering stack roadmap + 60-scene validation results |
| [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) | PPO action-scale rationale and parallel-training config |

---

## Quick reference

### Environment boundaries

| Environment | Python | Used for | Constraint |
|---|---|---|---|
| ROS sim + capture + RL training | 3.10 (system, Humble) | Gazebo, RViz, point cloud capture, UR3e control, PPO training | Do not import `open3d` / `polyscope` here |
| Clustering / Polyscope | 3.12 (`clustering/.venv`) | DBSCAN, PCA, visualisation, dataset verification | Do not import `rclpy` here |

Bridge between the two sides: PLY files at `~/holoassist_pointclouds/` and labelled
datasets at `~/holoassist_dataset/`.

### Critical files

```text
ros2_ws/src/holoassist_sim/
  config/sim_params.yaml          single source of truth: scene, camera, capture
  config/ros_gz_bridge.yaml.jinja2
  worlds/table_cubes.sdf.jinja2   rendered at launch; never edit generated SDF
  launch/sim.launch.py            renders templates, starts Gazebo/bridge/TF/RViz
  scripts/render_world.py         CLI: PARAMS TEMPLATE OUTPUT
  scripts/save_pointcloud.py      ROS node to PLY; reads params YAML
  scripts/dataset_capture.py      unattended scene randomisation + labelled captures

clustering/
  requirements.txt                Python 3.12 perception/visualisation dependencies
  view_ply.py                     load PLY, world-frame transform, Polyscope viewer
  detect_cubes.py                 load, crop, DBSCAN, centroids, PCA, Polyscope
  verify_detection.py             benchmark detected centroids vs labelled ground truth

ur3e_rl_ws/
  src/ur3e_rl_env                 Gymnasium env, PPO trainers, evaluator
  src/ur3e_policy_controller      runtime node for trained PPO models
  src/cube_perception             live D435i DBSCAN perception
```

### `sim_params.yaml` keys

```text
table.{size,pose,color}
cubes[].{name,size,pose,color,mass}
camera.{name,pose,topic,update_rate,width,height,horizontal_fov_deg,near_clip,far_clip}
camera.imu.{enabled,update_rate,topic}
capture.{output_dir,description,one_shot,voxel_size}
```

### Runtime facts

- ROS topics: `/camera/{image,depth_image,points,camera_info}`, `/imu`, `/clock`,
  `/tf_static`.
- TF chain: `map -> d435i_link -> d435i/d435i_link/rgbd`.
- Camera +X is the look direction; `camera.pose` pitch `0.528` rad is about 30 degrees
  downward.
- PLY naming: `<capture.description>_<N>cubes_<size>mm_v<NNN>.ply`; version numbers
  auto-increment.
- `render_world.py` is called by `sim.launch.py` through `sys.executable`; templates use
  Jinja2 and compute Gazebo inertia inline.
- Ignition Fortress (`ign-gazebo6`) + `ros_gz_bridge`; simulated depth noise is Gaussian
  with sigma `0.005 m`.
- Cube centre Z default is `0.52 m` (`0.5 m` table top + `0.02 m` half-cube).
- Units are metres and radians throughout.

### Clustering contract

Point clouds arrive in the camera body frame, so detection must transform points to the
world frame before any height-based crop.

```text
camera_to_world -> crop Z -> outlier removal -> DBSCAN -> size filter -> centroids
```

Key defaults in `clustering/detect_cubes.py`:

- `z_min = table_top_z + 0.015` to exclude the table surface with a 3-sigma noise margin.
- `z_max = table_top_z + cube_height + 0.01`.
- `eps = 0.015 m`, `min_samples = 20`.
- Centroids are world-frame `(x, y, z)` means per valid DBSCAN cluster and feed the PPO
  observation/state contract.
- PCA is used for Polyscope orientation visualisation only.

Validated accuracy: about 1.6 cm centroid error vs ground truth, with 100% cube recall on
the 60-scene benchmark.

---

## Prerequisites

The Gazebo + ROS 2 side runs on **Ubuntu 22.04** only. On Windows 11, use **WSL2 with
Ubuntu 22.04** — Windows 11's WSLg provides Gazebo and RViz GUIs out of the box. The
forward Isaac Sim work runs natively on Windows; see [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md).

You need before running `setup.sh`:

- Ubuntu 22.04 (Jammy) — native or WSL2
- ROS 2 Humble — [Ubuntu install guide](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- Python 3.12 — `sudo apt install python3.12 python3.12-venv`

---

## Setup

```bash
git clone <repo-url> && cd HoloAssist-AI
chmod +x setup.sh && ./setup.sh
```

`setup.sh` installs all ROS package dependencies via `rosdep`, builds the workspace, and
creates the Python 3.12 clustering venv at `clustering/.venv`. Partial flags:

```bash
./setup.sh --ros-only   # ROS workspace only
./setup.sh --py-only    # Python 3.12 clustering venv only
```

Manual build (without `setup.sh`):

```bash
source /opt/ros/humble/setup.bash
rosdep install --from-paths ros2_ws/src --ignore-src --rosdistro humble -r -y
cd ros2_ws && colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash

python3.12 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install -r clustering/requirements.txt
```

---

## First-run smoke test

Capture one point cloud and view its clustering result.

```bash
# Terminal 1 — start Gazebo + RViz
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py

# Terminal 2 — capture a frame to PLY
source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml

# Terminal 3 — run DBSCAN clustering on the latest capture
source clustering/.venv/bin/activate
python clustering/detect_cubes.py
```

A Polyscope window opens with the cropped point cloud and detected cube centroids. Expected
accuracy: ~1.6 cm centroid error vs ground truth.

The full command reference (sim, dataset capture, RL training in 1- or 4-worker modes,
policy deployment, real-D435i perception) is in [LAUNCH.md](LAUNCH.md).

---

## Modifying the perception scene

All scene parameters live in
[ros2_ws/src/holoassist_sim/config/sim_params.yaml](ros2_ws/src/holoassist_sim/config/sim_params.yaml).
Change table size, cube positions/colours/sizes, camera pose, or capture behaviour — the
Gazebo world is regenerated from this file at every launch. Never edit the generated SDF.
