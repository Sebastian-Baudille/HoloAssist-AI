# HoloAssist-AI — Architecture

**Course:** 41118 Artificial Intelligence in Robotics
**Repository:** HoloAssist-AI

This is the system-design reference. For the project pitch and setup, see
[README.md](README.md). For commands, see [LAUNCH.md](LAUNCH.md). For current work in flight,
see [PROGRESS.md](PROGRESS.md). For the forward Isaac Sim migration plan, see
[ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md).

---

## 1. What this project is

HoloAssist-AI is the AI extension to the RS2 HoloAssist project. It builds an end-to-end
perception + control pipeline that lets a robot arm see coloured cubes on a tabletop with an
RGB-D camera, infer their 3D positions, and grasp them using a reinforcement-learning policy.

The pipeline is built and validated entirely in simulation (Gazebo Fortress / Ignition 6 +
ROS 2 Humble), with a clear path to swapping in a physical Intel RealSense D435i and a UR3e
arm without changing the perception or policy code.

**Core question:** Can classical clustering (DBSCAN + PCA) on a noisy simulated depth cloud
produce centroids accurate enough to be the state vector for a PPO grasping policy on a
UR3e arm?

**Validated to date:** ~1.6 cm centroid error vs ground truth on 60 scenes (within sensor
noise σ = 0.005 m), 100 % cube recall. See [clustering/README.md](clustering/README.md).

---

## 2. System data flow

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
  camera-to-world transform  →  height crop above table top  →  outlier removal
         │
         ▼
[DBSCAN Clustering]
  DBSCAN on XYZ  →  cluster labels
  Mean per cluster → 3D centroid  (PCA used only for visualisation)
         │
         │  Centroid array, shape (k, 3) in metres, world frame
         ▼
[PPO Observation Vector — 13-D]
  Normalised EE / cube / bin positions + grasped + gripper + height + timestep
         │
         ▼
[RL Agent — UR3e in Gazebo]
  Stable-Baselines3 PPO — joint-target control, ±0.24 rad/step
  Reward: reach → grasp → transport → place, with penalties
         │
         ▼
[Evaluation]
  Centroid error vs ground truth (sim_params.yaml)
  Grasp success rate, episode reward (TensorBoard)
```

---

## 3. Repository layout

```
HoloAssist-AI/
├── ros2_ws/src/
│   ├── holoassist_sim/             Stage 1 sim: table + cubes + D435i + scene controller
│   │   ├── config/sim_params.yaml      single source of truth for the scene
│   │   ├── worlds/table_cubes.sdf.jinja2
│   │   ├── launch/sim.launch.py
│   │   └── scripts/{render_world.py, save_pointcloud.py, scene_controller.py, dataset_capture.py}
│   ├── moveit_robot_control[_msgs] MoveIt control + custom messages + pick-place sequencer
│   ├── onrobot_description          OnRobot RG2 gripper URDF
│   ├── onrobot_driver               gripper driver
│   └── ur_onrobot                   UR3e + RG2 combined control / description / MoveIt config
│
├── clustering/                     Python 3.12 venv — perception stage
│   ├── detect_cubes.py              DBSCAN pipeline: load → crop → cluster → centroids
│   ├── verify_detection.py          accuracy benchmark vs labelled dataset
│   ├── view_ply.py                  raw PLY → world-frame Polyscope viewer
│   ├── sample_data/                 reference PLY captures
│   └── requirements.txt
│
├── ur3e_rl_ws/                     Stage 3: UR3e PPO workspace (Gazebo-backed)
│   ├── src/
│   │   ├── ur3e_gazebo_sim          Gazebo scene: trolley + UR3e + RG2 + cubes + bins
│   │   ├── ur3e_rl_env              Gymnasium env, PPO training, demo recorder, BC pretrain
│   │   ├── ur3e_safety_layer        joint clamp, TCP height check, collision flag
│   │   ├── ur3e_policy_controller   runtime node that loads a trained PPO model
│   │   └── cube_perception          live D435i perception (DBSCAN + tracker + RViz markers)
│   ├── rl_models/                   PPO checkpoints + Gazebo worker logs
│   ├── tb_logs/                     TensorBoard run history
│   └── TRAINING_NOTES.md            current training rationale, action-scale + parallel internals
│
├── README.md                       pitch + quickstart
├── ARCHITECTURE.md                 ← this document
├── LAUNCH.md                       command reference (sim, capture, clustering, RL, deploy)
├── PROGRESS.md                     current section + status snapshot
├── ISAAC_SIM_PLAN.md               forward plan: port RL stack to Isaac Lab on Windows
├── CLAUDE.md                       Claude Code reference card
├── setup.sh                        one-command bootstrap (ROS + Python venv)
└── docs/
    └── clustering-pipeline.md      reference manual: preprocessing, clustering, PCA, tuning
```

---

## 4. Two Python environments

ROS 2 Humble pins Python 3.10. The clustering / RL stack needs newer libraries (numpy 2.x,
open3d 0.19, scikit-learn 1.8, Stable-Baselines3). The two sides are intentionally isolated
and communicate through PLY files on disk.

| Environment | Python | Used for | Activate |
|---|---|---|---|
| ROS sim + capture + RL training | 3.10 (system, Humble) | Gazebo, RViz, point cloud capture, UR3e control, PPO training | `source ros2_ws/install/setup.bash` |
| Clustering + visualisation | 3.12 (venv) | DBSCAN, PCA, Polyscope, dataset verification | `source clustering/.venv/bin/activate` |

**Bridge:** PLY files written to `~/holoassist_pointclouds/` and labelled datasets to
`~/holoassist_dataset/`.
**Never mix:** no `rclpy` imports in the clustering venv; no `open3d`/`polyscope` imports
inside the ROS-sourced shell.

The PPO training stack (Stable-Baselines3 + Gymnasium) runs in the ROS-sourced Python 3.10
because it talks to live ROS topics. The forward Isaac Sim port uses its own bundled
Python 3.10 on Windows — see [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md).

---

## 5. Stage-by-stage status

### Stage 1 — Point cloud data collection — **complete**

- Gazebo Fortress scene built from `sim_params.yaml` (single source of truth: table size,
  cube poses/colours/sizes, camera intrinsics, capture parameters).
- World SDF and bridge YAML are rendered from Jinja2 templates at launch — never edit the
  generated SDF.
- `save_pointcloud.py` writes binary PLY (XYZ float32 + RGB uint8) with auto-incremented
  version numbers: `<description>_<Ncubes>cubes_<size>mm_v<NNN>.ply`.
- Simulated D435i: 848 × 480, 87° H FOV, Gaussian depth noise σ = 0.005 m.
- `scene_controller.py` + `dataset_capture.py` generate labelled datasets (PLY + JSON
  ground truth) for clustering validation.

### Stage 2 — Clustering — **complete (validated)**

- Point cloud arrives in the **camera body frame** (+X = look direction). Must be
  transformed to the world frame before any height-based crop.
- Pipeline: `camera_to_world` → crop Z to `[table_top + 0.015 m, table_top + cube_height + 0.01 m]`
  → statistical outlier removal → DBSCAN (`eps = 0.015 m`, `min_samples = 20`) → size filter
  → mean per cluster.
- DBSCAN replaced K-Means: no fixed `k` required, naturally rejects noise, handles variable
  cube count. Trade-off: needs ≥ 4 cm centre-to-centre cube spacing (2× cube size).
- Centroid = `cluster_pts.mean(axis=0)` in the world frame, fed directly into the RL state
  vector.
- PCA is still computed per cluster but used only for Polyscope visualisation of orientation
  axes.
- Validated 1.63 cm mean error / 100 % recall on 60 labelled scenes (see
  [clustering/README.md](clustering/README.md)).
- Reference manual: [docs/clustering-pipeline.md](docs/clustering-pipeline.md) (full
  preprocessing + scaling + DBSCAN/K-Means alternatives + tuning cheat sheet).

### Stage 3 — Reinforcement learning (UR3e pick-and-place) — **in progress**

- Task: PPO policy that drives the UR3e end effector to within 4 cm of a target cube using
  joint-target control, grasps it, transports it to a bin, and places it.
- Algorithm: Stable-Baselines3 PPO.
- Action space: `Box(-1, 1, shape=(7,))` — six joint targets (scaled by ±0.24 rad/step =
  ~75 % of the UR3e's π rad/s hardware limit) + one gripper open/close command.
- Observation: **13-D normalised** vector:
  - `[0:3]` EE XYZ, `[3:6]` target-cube XYZ, `[6:9]` bin XYZ,
  - `[9]` grasped flag, `[10]` gripper state, `[11]` EE height, `[12]` timestep.
- Reward phases: reach → grasp → transport → place, with collision / time / smoothness
  penalties. See `ur3e_rl_env/reward.py`.
- Safety layer: per-step slew clamp (`max_delta_rad = 0.24`) + absolute joint-limit clamp +
  TCP height ≥ 0.02 m + collision flag (optional MoveIt-backed checker, per-domain).
- Parallel training: 4 headless Gazebo workers on separate `ROS_DOMAIN_ID` (30–33) and
  Gazebo master ports (11400–11403). Drops training from ~6 hours → ~30 minutes on a 20-
  thread laptop.
- Demonstration warm-start: `record_demo` collects Unity XR teleop episodes;
  `pretrain_from_demos` behaviour-clones them into a PPO policy before RL fine-tuning.
- Live perception package `cube_perception` provides DBSCAN-based cube detection from a
  real D435i with temporal tracking + occlusion hold; used at deployment, not training.
- Current state: scene, env, safety, training scripts, perception node all wired; ongoing
  long training runs. See [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) for
  current parameter rationale.

### Stage 4 — Evaluation — **partially available**

- Clustering side: ground-truth cube poses are exact in `sim_params.yaml` and in the
  labelled dataset; `clustering/verify_detection.py` computes centroid error.
- RL side: TensorBoard logs in `ur3e_rl_ws/tb_logs/`, `evaluate_policy` runs 20-episode
  batches and prints success rate.
- Visualisation: Polyscope (interactive 3D clusters + centroids + PCA axes), RViz (live
  sim + cube markers), TensorBoard (training curves).

### Stage 5 — Real hardware — **future**

- Real D435i via `realsense2_camera` publishes to the same `/camera/depth/color/points`
  topic — clustering needs no changes if coordinate frames match.
- UR3e + OnRobot RG2 drivers (`onrobot_driver`, `ur_onrobot`) already in `ros2_ws/src/`
  for the physical setup.
- Trained PPO weights load and run via `ur3e_policy_controller/rl_policy_node`.
- Live perception via `cube_perception` package (pose + confidence + occlusion hold).
- Sim-to-real gap is the main remaining risk; mitigated by the forward Isaac Sim migration
  ([ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md)) which adds GPU-parallel training + domain
  randomisation.

---

## 6. Key files at a glance

| File | Role |
|---|---|
| [ros2_ws/src/holoassist_sim/config/sim_params.yaml](ros2_ws/src/holoassist_sim/config/sim_params.yaml) | The only file you edit to change the perception scene |
| [ros2_ws/src/holoassist_sim/launch/sim.launch.py](ros2_ws/src/holoassist_sim/launch/sim.launch.py) | Renders templates, starts Gazebo + bridge + TF + RViz |
| [ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py](ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py) | ROS node — capture a frame and write a versioned PLY |
| [ros2_ws/src/holoassist_sim/scripts/dataset_capture.py](ros2_ws/src/holoassist_sim/scripts/dataset_capture.py) | Automated labelled-dataset capture (60-scene batches) |
| [clustering/detect_cubes.py](clustering/detect_cubes.py) | End-to-end perception: PLY → centroids → Polyscope |
| [clustering/verify_detection.py](clustering/verify_detection.py) | DBSCAN accuracy benchmark vs ground truth |
| [ur3e_rl_ws/src/ur3e_rl_env/](ur3e_rl_ws/src/ur3e_rl_env/) | Gymnasium env, PPO trainer, evaluator, demo recorder |
| [ur3e_rl_ws/src/ur3e_safety_layer/](ur3e_rl_ws/src/ur3e_safety_layer/) | Joint clamp + slew limit + TCP height check |
| [ur3e_rl_ws/src/cube_perception/](ur3e_rl_ws/src/cube_perception/) | Live D435i DBSCAN + tracker + RViz markers |
| [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) | Current PPO parameter rationale + parallel-worker config |

---

## 7. Reference documents in this repo

| Document | Purpose |
|---|---|
| [README.md](README.md) | Setup and first-run guide |
| [LAUNCH.md](LAUNCH.md) | Command reference for sim, clustering, RL training, deployment |
| [PROGRESS.md](PROGRESS.md) | Current section, status, latest training metrics |
| [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md) | Forward plan: migrate RL stack to Isaac Lab on Windows |
| [docs/clustering-pipeline.md](docs/clustering-pipeline.md) | Reference manual: RGB-D preprocessing, clustering, PCA |
| [clustering/README.md](clustering/README.md) | Clustering stack roadmap and validated accuracy results |
| [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) | Current PPO training state and parameter rationale |
| [CLAUDE.md](CLAUDE.md) | Claude Code reference card — environments, key files, conventions |
