# HoloAssist-AI — Project Overview

**Course:** 41118 Artificial Intelligence in Robotics
**Repository:** HoloAssist-AI

---

## 1. What this project is

HoloAssist-AI is the AI extension to the RS2 HoloAssist project. It builds an end-to-end perception + control pipeline that lets a robot arm see coloured cubes on a tabletop with an RGB-D camera, infer their 3D positions, and grasp them using a reinforcement-learning policy.

The pipeline is built and validated entirely in simulation (Gazebo Fortress / Ignition 6 + ROS 2 Humble), with a clear path to swapping in a physical Intel RealSense D435i and a UR3e arm without changing the perception or policy code.

**Core question:** Can classical clustering (K-Means + PCA) on a noisy simulated depth cloud produce centroids accurate enough to be the state vector for a PPO/DQN grasping policy on a UR3e arm?

---

## 2. System architecture

```
[Gazebo Simulation]
  Table + 4 cubes + D435i camera
         │
         │  ROS 2 topic: /camera/points
         ▼
[Point Cloud Capture]
  save_pointcloud.py  →  binary PLY (XYZ + RGB) on disk
         │
         │  ~/holoassist_pointclouds/*.ply
         ▼
[Preprocessing]
  camera-to-world transform  →  height crop above table top
         │
         ▼
[K-Means Clustering]
  K-Means on XYZ  →  cluster labels
  Mean per cluster → 3D centroid  (PCA used only for visualisation)
         │
         │  Centroid array, shape (k, 3) in metres, world frame
         ▼
[DQN / PPO State Vector]
  state = centroids.flatten() + arm joint state
         │
         ▼
[RL Agent — UR3e in Gazebo]
  PPO (Stable-Baselines3) — joint-delta control
  Reward: −distance to target, +grasp success, −collision
         │
         ▼
[Evaluation]
  Centroid error vs ground truth (sim_params.yaml)
  Grasp success rate, episode reward (TensorBoard)
```

**Validated accuracy of the perception stage:** ~1.6 cm centroid error vs ground truth, within the simulated D435i depth noise (σ = 0.005 m).

---

## 3. Repository layout

```
HoloAssist-AI/
├── ros2_ws/src/
│   ├── holoassist_sim/             ← Stage 1 sim: table + cubes + D435i
│   │   ├── config/sim_params.yaml      single source of truth for the scene
│   │   ├── worlds/table_cubes.sdf.jinja2
│   │   ├── launch/sim.launch.py
│   │   └── scripts/{render_world.py, save_pointcloud.py}
│   ├── moveit_robot_control[/_msgs] ← MoveIt control + custom messages
│   ├── onrobot_description          ← OnRobot RG2 gripper URDF
│   ├── onrobot_driver               ← gripper driver
│   └── ur_onrobot                   ← UR3e + RG2 MoveIt config
│
├── clustering/                     ← Python 3.11 venv — perception stage
│   ├── detect_cubes.py              full pipeline: load → crop → K-Means → centroids
│   ├── view_ply.py                  raw PLY → world-frame Polyscope viewer
│   ├── sample_data/                 reference PLY captures
│   └── requirements.txt
│
├── ur3e_rl_ws/                     ← Stage 3: UR3e PPO reach-to-cube workspace
│   ├── src/
│   │   ├── ur3e_gazebo_sim          Gazebo scene: trolley + UR3e + RG2 + cubes + bins
│   │   ├── ur3e_rl_env              Gymnasium env, PPO training, demo recorder, BC pretrain
│   │   ├── ur3e_safety_layer        joint-delta clamp, TCP height check, collision flag
│   │   └── ur3e_policy_controller   runtime node that loads a trained PPO model
│   ├── scripts/launch_moveit_workers.sh
│   ├── rl_models/                   PPO checkpoints + Gazebo worker logs
│   ├── tb_logs/                     TensorBoard run history
│   └── TRAINING_NOTES.md            current training state and parameter rationale
│
├── README.md                       setup + first-run guide
├── LAUNCH.md                       quick-reference launch commands
├── architecture.md                 detailed component breakdown + ownership
├── rgbd_clustering_pipeline.md     RGB-D clustering reference manual
├── CLAUDE.md                       Claude Code reference card
├── setup.sh                        one-command bootstrap (ROS + Python venv)
└── PROJECT_OVERVIEW.md             ← this document
```

---

## 4. The two Python environments (and why)

ROS 2 Humble pins Python 3.10. The clustering / RL stack needs newer libraries (numpy 2.x, open3d 0.19, scikit-learn 1.8, Stable-Baselines3). The two sides are intentionally isolated and communicate through PLY files on disk.

| Environment | Python | Used for | Activate |
|---|---|---|---|
| ROS sim + capture | 3.10 (system, Humble) | Gazebo, RViz, point cloud capture, UR3e control | `source ros2_ws/install/setup.bash` |
| Clustering + RL | 3.11 (venv) | K-Means, PCA, Polyscope, PPO training | `source clustering/.venv/bin/activate` |

**Bridge:** PLY files written to `~/holoassist_pointclouds/`.
**Never mix:** no `rclpy` imports in the clustering venv; no `open3d`/`polyscope` imports inside the ROS-sourced shell.

---

## 5. Stage-by-stage status

### Stage 1 — Point cloud data collection — **complete**

- Gazebo Fortress scene built from `sim_params.yaml` (single source of truth: table size, cube poses/colours/sizes, camera intrinsics, capture parameters).
- World SDF and bridge YAML are rendered from Jinja2 templates at launch — never edit the generated SDF.
- `save_pointcloud.py` writes binary PLY (XYZ float32 + RGB uint8) with auto-incremented version numbers, naming format:
  `<description>_<Ncubes>cubes_<size>mm_v<NNN>.ply`
- Simulated D435i: 848×480, 87° H FOV, Gaussian depth noise σ = 0.005 m.

### Stage 2 — Clustering — **complete (validated)**

- Point cloud arrives in the **camera body frame** (+X = look direction). Must be transformed to the world frame before any height-based crop.
- Pipeline: `camera_to_world` → crop Z to `[table_top + 0.015 m, table_top + cube_height + 0.01 m]` → K-Means on XYZ only → mean per cluster.
- `color_weight = 0.0` — RGB weighting was tested and rejected: it confuses K-Means under simulated Gazebo lighting where shaded faces shift apparent colour.
- Centroid = `cluster_pts.mean(axis=0)` in the world frame, fed directly into the RL state vector.
- PCA is still computed per cluster but used only for Polyscope visualisation of orientation axes.
- Reference manual: `rgbd_clustering_pipeline.md` (full RANSAC + scaling + DBSCAN alternative + tuning cheat sheet).

### Stage 3 — Reinforcement learning (UR3e reach-to-cube) — **in progress**

- Task: PPO policy that drives the UR3e end effector to within 4 cm of a target cube using joint-delta control.
- Algorithm: Stable-Baselines3 PPO.
- Action space: `Box(-0.24, 0.24, shape=(6,))` rad/step — recently raised from ±0.03 to give the random initial policy a realistic chance of stumbling into the success radius.
- Observation: 29-D vector (6 joint pos, 6 joint vel, 3 EE pos, 3 cube pos, 3 goal pos, 3 cube−EE, 3 goal−cube, 1 grasped flag, 1 collision flag).
- Safety layer: joint-delta clamp + TCP height ≥ 0.02 m + collision flag (optional MoveIt-backed checker).
- Parallel training: 4 headless Gazebo workers on separate `ROS_DOMAIN_ID` + Gazebo master ports. Drops training from ~6 hours → ~30 minutes on a 20-thread laptop.
- Human-demonstration pretraining: `record_demo` collects Unity XR teleop episodes; `pretrain_from_demos` behavior-clones them into a PPO policy before RL fine-tuning.
- Current state: scene, env, safety, and training scripts are wired; ongoing tuning of speed limits and MoveIt integration (see `ur3e_rl_ws/TRAINING_NOTES.md`).

### Stage 4 — Evaluation — **partially available**

- Clustering side: ground-truth cube poses are exact in `sim_params.yaml`, so centroid error is directly measurable.
- RL side: TensorBoard logs in `ur3e_rl_ws/tb_logs/`, `evaluate_policy` runs 20-episode batches and prints success rate.
- Visualisation: Polyscope (interactive 3D clusters + centroids + PCA axes), RViz (live sim), matplotlib (training curves).

### Stage 5 — Real hardware — **future**

- Real D435i via `realsense-ros` publishes to the same `/camera/points` topic — clustering needs no changes if coordinate frames match.
- UR3e + OnRobot RG2 drivers (`onrobot_driver`, `ur_onrobot`) already in `ros2_ws/src/` for the physical setup.
- Trained PPO weights load and run via `ur3e_policy_controller/rl_policy_node`.
- Sim-to-real gap is the main remaining risk.

---

## 6. Key files at a glance

| File | What it does |
|---|---|
| [ros2_ws/src/holoassist_sim/config/sim_params.yaml](ros2_ws/src/holoassist_sim/config/sim_params.yaml) | The only file you edit to change the scene |
| [ros2_ws/src/holoassist_sim/launch/sim.launch.py](ros2_ws/src/holoassist_sim/launch/sim.launch.py) | Renders templates, starts Gazebo + bridge + TF + RViz |
| [ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py](ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py) | ROS node — captures a frame and writes a versioned PLY |
| [clustering/detect_cubes.py](clustering/detect_cubes.py) | End-to-end perception: PLY → centroids → Polyscope |
| [clustering/view_ply.py](clustering/view_ply.py) | Raw PLY → world-frame viewer |
| [ur3e_rl_ws/src/ur3e_rl_env/](ur3e_rl_ws/src/ur3e_rl_env/) | Gymnasium env, PPO trainer, evaluator, demo recorder |
| [ur3e_rl_ws/src/ur3e_safety_layer/](ur3e_rl_ws/src/ur3e_safety_layer/) | Joint-delta clamp, TCP height check, collision flag |
| [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) | Current parameter rationale, parallel-training config |

---

## 7. Quick-start commands

**One-time setup:**
```bash
chmod +x setup.sh && ./setup.sh
```

**Capture a point cloud (two terminals):**
```bash
# T1 — sim
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py

# T2 — capture (writes ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply)
source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

**Run perception on the latest capture:**
```bash
source clustering/.venv/bin/activate
python clustering/detect_cubes.py
```

**Train PPO (parallel, 4 workers, no MoveIt):**
```bash
source ros2_ws/install/setup.bash
cd ur3e_rl_ws && source install/setup.bash
export ROS_DOMAIN_ID=30
UR3E_RL_NUM_ENVS=4 ros2 run ur3e_rl_env train_ppo_parallel
```

Full launch reference: [LAUNCH.md](LAUNCH.md)

---

## 8. Team and ownership

| Member | Primary responsibility |
|---|---|
| Person 1 | Project lead / integration — connecting clustering output to RL input, end-to-end pipeline |
| Person 2 | Depth camera data pipeline — point cloud filtering, preprocessing, dataset collection |
| Person 3 | K-Means implementation — clustering on 3D point cloud, tuning, evaluation metrics |
| Person 4 | RL implementation — MDP design, reward shaping, PPO/DQN training |
| Person 5 | Evaluation & validation — metrics, benchmarking, simulation environment setup |

**Key collaboration points:**
- Persons 3 & 4 own the interface where centroids become the RL state vector.
- Person 2 supports both with the data pipeline and dataset quality.
- Person 1 owns end-to-end integration.

---

## 9. Outstanding integration items

- [ ] Lock the centroid array contract (shape, dtype, coordinate frame, units) between perception and RL.
- [ ] Decide arm-state representation (joint angles vs end-effector pose) for the RL observation.
- [ ] Decide where state normalisation lives in the pipeline.
- [ ] Wire ground-truth cube-pose reader from `sim_params.yaml` for the centroid-error metric.
- [ ] Define the dataset format for captured PLY episodes used in BC pretraining and replay.
- [ ] Replace placeholder collision flag with a real MoveIt-backed checker for hardware deployment.

---

## 10. Reference documents in this repo

| Document | Purpose |
|---|---|
| [README.md](README.md) | Setup and first-run guide |
| [LAUNCH.md](LAUNCH.md) | Quick-reference launch commands for sim + clustering + RL |
| [architecture.md](architecture.md) | Detailed component breakdown with per-stage tool tables |
| [rgbd_clustering_pipeline.md](rgbd_clustering_pipeline.md) | Standalone reference manual for the RGB-D clustering pipeline |
| [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) | Current PPO training state, parameter rationale, parallel-worker config |
| [CLAUDE.md](CLAUDE.md) | Claude Code reference card — environments, key files, conventions |
