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
| [CLAUDE.md](CLAUDE.md) | Reference card for Claude Code sessions |

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
