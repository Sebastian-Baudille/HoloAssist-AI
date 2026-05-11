# Project Architecture

**Course:** 41118 Artificial Intelligence in Robotics
**Goal:** Estimate the 3D positions of cubes on a tabletop using an RGB-D camera, then train a DQN agent to grasp them with a robot arm.

---

## System Overview

```
[Gazebo Simulation]
  Table + 4 cubes + D435i camera
         │
         │  ROS 2 topic: /camera/points
         ▼
[Point Cloud Capture]  ──────────────────────────────────── Person 2
  save_pointcloud.py
  XYZ + RGB → .ply files on disk
         │
         │  ~/holoassist_pointclouds/*.ply
         ▼
[Preprocessing]  ────────────────────────────────────────── Person 2
  RANSAC table removal
  Height crop (cube layer only)
  Voxel downsample / outlier removal
         │
         ▼
[K-Means Clustering]  ──────────────────────────────────── Person 3
  Feature: scaled XYZ + weighted RGB
  K-Means → cluster labels
  PCA per cluster → centroid + orientation
         │
         │  Cube centroids [ (x₁,y₁,z₁), ..., (xₙ,yₙ,zₙ) ]
         ▼
[DQN State Vector]  ◄──── Integration point ────────────── Persons 1 + 3 + 4
  State:  cube centroids + arm joint state
  Action: Δ joint angles / gripper command
  Reward: grasp success / distance to target
         │
         ▼
[DQN Agent]  ───────────────────────────────────────────── Person 4
  MDP design
  Neural network
  Training loop (sim-based)
         │
         ▼
[Evaluation & Validation]  ─────────────────────────────── Person 5
  Clustering metrics (silhouette, centroid error)
  DQN metrics (success rate, episode reward)
  Polyscope / Gazebo visualisation
```

---

## Team Responsibilities
 
| Member     | Primary Responsibility                                      |
|------------|-------------------------------------------------------------|
| Person 1   | Project lead / integration — connecting K-Means output to DQN input, overall pipeline |
| Person 2   | Depth camera data pipeline — point cloud filtering, preprocessing, dataset collection |
| Person 3   | K-Means implementation — clustering on 3D point cloud, tuning K, evaluation metrics   |
| Person 4   | DQN implementation — MDP design, reward shaping, neural network training              |
| Person 5   | Evaluation & validation — metrics, benchmarking, simulation environment setup         |
 

**Key collaboration points:**
- Persons 3 & 4 own the interface where K-Means centroids become the DQN state vector.
- Person 2 supports both with the data pipeline and dataset quality.
- Person 1 owns the end-to-end integration and resolves cross-component issues.

---

## Component Breakdown

### 1. Simulation Environment
**Owner:** Person 5 (setup) + Person 2 (data collection)

| Item | Detail |
|------|--------|
| Simulator | Gazebo Fortress (Ignition 6) via ROS 2 Humble |
| Scene | Table (0.5 m cube) + 4 coloured cubes (4 cm) |
| Camera | Intel RealSense D435i model — 848×480, 87° H FOV |
| Config | `ros2_ws/src/holoassist_sim/config/sim_params.yaml` (single source of truth) |
| Launch | `ros2 launch holoassist_sim sim.launch.py` |

All scene parameters (cube positions, colours, camera pose) are set in `sim_params.yaml`. Changing that file and relaunching regenerates the world automatically.

---

### 2. Data Pipeline
**Owner:** Person 2

| Step | Tool | Key file |
|------|------|----------|
| Capture | `save_pointcloud.py` (ROS node) | `ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py` |
| Format | Binary PLY (XYZ float32 + RGB uint8) | `~/holoassist_pointclouds/*.ply` |
| Load | `open3d.io.read_point_cloud` | `clustering/detect_cubes.py` |
| Plane removal | RANSAC via `open3d.segment_plane` | `clustering/detect_cubes.py` |
| Height crop | Signed distance above plane | `clustering/detect_cubes.py` |
| Downsample | Voxel grid (configurable in `sim_params.yaml`) | `save_pointcloud.py` |

**PLY filename format:** `<description>_<N>cubes_<size>mm_v<NNN>.ply`
Example: `default_4cubes_40mm_v001.ply`

---

### 3. K-Means Clustering
**Owner:** Person 3

Input: preprocessed point cloud (cube layer only), XYZ + RGB per point
Output: cluster label per point + one `(x, y, z)` centroid per cube

| Step | Approach |
|------|----------|
| Features | `StandardScaler` on XYZ and RGB independently, then concatenate with RGB weight |
| Clustering | `sklearn.cluster.KMeans` — k = number of cubes (known from sim), k-means++ init |
| Centroid extraction | `sklearn.decomposition.PCA` per cluster — `pca.mean_` is the 3D centroid |
| Orientation | PCA principal axes → bounding box + pose matrix for downstream use |
| Tuning | Silhouette score, elbow curve, variance ratio per cluster (~[0.33, 0.33, 0.33] for a cube) |

Current entry point: `clustering/detect_cubes.py`

**Centroid output format (fed to DQN):**
```python
centroids: np.ndarray  # shape (k, 3) — one row per detected cube, in metres
```

---

### 4. DQN Agent
**Owner:** Person 4

| Item | Detail |
|------|--------|
| Framework | TBD (PyTorch recommended — compatible with Python 3.12 venv) |
| State vector | Cube centroids from K-Means + robot arm joint state |
| Action space | Δ joint angles or end-effector Cartesian delta + gripper binary |
| Reward | +1 grasp success, −distance penalty, −collision penalty |
| Training env | Gazebo simulation (closed loop) |

**State vector interface (agreed with Person 3):**
```python
# K-Means centroids flattened into the state
cube_centroids: np.ndarray  # (k, 3) from detect_cubes.py
arm_state: np.ndarray       # joint angles / end-effector pose from ROS
state = np.concatenate([cube_centroids.flatten(), arm_state])
```

---

### 5. Evaluation & Validation
**Owner:** Person 5

#### Clustering metrics
| Metric | Measures |
|--------|----------|
| Silhouette score | Cluster cohesion and separation |
| Centroid error | Euclidean distance between estimated and ground-truth cube centre |
| Variance ratio | Whether each cluster is cube-shaped (~[0.33, 0.33, 0.33]) |

Ground-truth positions are known exactly from `sim_params.yaml` (cube poses), so centroid error is directly measurable in simulation.

#### DQN metrics
| Metric | Measures |
|--------|----------|
| Grasp success rate | % of episodes ending in successful grasp |
| Mean episode reward | Training progress |
| Steps to grasp | Efficiency |

#### Visualisation tools
- **Polyscope** — interactive 3D point cloud + cluster + centroid inspection (`clustering/detect_cubes.py`)
- **RViz** — live camera feed and point cloud during simulation
- **Matplotlib** — elbow curves, silhouette scores, training reward curves

---

## Environments and Dependencies

Two isolated Python environments — this is required because ROS 2 Humble pins Python 3.10.

| Environment | Python | Used for | Activate |
|-------------|--------|----------|----------|
| ROS sim + capture | 3.10 (system) | Gazebo, RViz, point cloud capture | `source ros2_ws/install/setup.bash` |
| Clustering + DQN | 3.12 (venv) | K-Means, PCA, DQN, Polyscope | `source clustering/.venv/bin/activate` |

**Data handoff:** PLY files at `~/holoassist_pointclouds/` — no shared process needed.

Install everything:
```bash
chmod +x setup.sh && ./setup.sh
```

Key packages (Python 3.12 venv):

| Package | Used for |
|---------|----------|
| `open3d` | PLY I/O, RANSAC plane segmentation |
| `scikit-learn` | KMeans, PCA, DBSCAN, StandardScaler, silhouette score |
| `polyscope` | 3D interactive visualisation |
| `numpy` | Array operations throughout |
| `matplotlib` | 2D metric plots |

---

## Repository Layout

```
HoloAssist-AI/
├── ros2_ws/src/holoassist_sim/
│   ├── config/
│   │   ├── sim_params.yaml          ← edit this to change the scene
│   │   └── ros_gz_bridge.yaml.jinja2
│   ├── worlds/
│   │   └── table_cubes.sdf.jinja2   ← rendered at launch, never edit directly
│   ├── launch/
│   │   └── sim.launch.py
│   └── scripts/
│       ├── render_world.py
│       └── save_pointcloud.py
├── clustering/
│   ├── detect_cubes.py              ← Person 3 entry point
│   └── requirements.txt
├── setup.sh                         ← one-command bootstrap
├── architecture.md                  ← this file
├── README.md                        ← human-readable project guide
└── CLAUDE.md                        ← Claude Code reference card
```

---

## Simulation Tools by Project Stage

Everything communicates through ROS 2 topics. This means swapping simulated components for real ones (camera, arm) requires no changes to the clustering or DQN code — only the drivers change.

---

### Stage 1 — Point cloud data collection
**Status: complete**

| Tool | Role |
|------|------|
| Gazebo Fortress (Ignition 6) | Physics simulation — renders the scene, simulates the D435i sensor |
| ROS 2 Humble | Middleware — moves data between sim and capture script |
| ros_gz_bridge | Translates Gazebo sensor topics to ROS 2 topics |
| RViz | Live visualisation of the point cloud and camera feed |
| `save_pointcloud.py` | ROS node that captures a frame and writes it to PLY |

---

### Stage 2 — Clustering development and validation
**Status: complete (initial implementation)**

| Tool | Role |
|------|------|
| open3d | PLY loading, RANSAC plane removal |
| scikit-learn | KMeans, PCA, StandardScaler, silhouette score |
| Polyscope | Interactive 3D inspection of clusters, centroids, PCA axes |
| matplotlib | Elbow curves, silhouette scores, centroid error plots |
| numpy | Array operations throughout |

Ground truth for validation comes directly from `sim_params.yaml` — cube poses are exactly known, so centroid error is directly measurable without any additional tooling.

---

### Stage 3 — DQN training
**Status: not yet built — largest remaining simulation effort**

| Tool | Role | Status |
|------|------|--------|
| Gazebo Fortress | Extended to include a robot arm model | Needs arm added to world |
| ros2_control | Sends joint commands from DQN to the simulated arm | Needs configuration |
| MoveIt 2 | Inverse kinematics — converts Cartesian targets to joint angles | Optional but recommended |
| PyTorch | Neural network definition, forward pass, backpropagation | Not started |
| Stable Baselines 3 | RL training loop, replay buffer, DQN algorithm | Not started |
| Gazebo contact plugin | Detects grasp success for reward signal | Needs configuration |

The arm model is a significant open decision — a simpler arm (fewer joints, well-supported ROS 2 packages) reduces integration complexity considerably for a course project.

**The DQN control loop:**
```
Observe state (cube centroids + arm joint angles)
        │
        ▼
DQN policy network → select action
        │
        ▼
Action → ROS 2 topic → ros2_control → Gazebo joint actuators → arm moves
        │
        ▼
New state + reward (grasp success / distance to target)
        │
        ▼
Store in replay buffer → update network weights → repeat
```

---

### Stage 4 — Evaluation and benchmarking
**Status: partially available (clustering side ready)**

| Tool | Role |
|------|------|
| Polyscope | Visualising clustering quality on captured datasets |
| matplotlib | Training reward curves, grasp success rate over episodes |
| `sim_params.yaml` | Ground truth source for clustering centroid error |
| Gazebo | Running held-out evaluation episodes — same sim, no code changes |

---

### Stage 5 — Real hardware deployment
**Status: future**

| Tool | Role |
|------|------|
| Physical Intel D435i | Replaces simulated camera — same ROS 2 topic interface |
| `realsense-ros` | ROS 2 driver for the physical camera |
| Robot arm ROS 2 driver | Replaces Gazebo controller — same topic interface |
| Clustering pipeline | No changes needed if coordinate frames are matched |
| DQN policy | Loaded from trained weights — sim-to-real gap is the main risk at this stage |

---

## Integration Checklist

These are the cross-component interfaces that need explicit agreement between owners before implementation.

- [ ] **Centroid array format** — shape, dtype, coordinate frame (world/camera), units. *(Person 3 → Person 4)*
- [ ] **Arm state representation** — joint angles vs end-effector pose, what's included. *(Person 4)*
- [ ] **State normalisation** — who normalises the state vector before feeding DQN. *(Person 1)*
- [ ] **Evaluation ground truth** — process for reading cube poses from `sim_params.yaml` for centroid error metric. *(Person 5)*
- [ ] **Dataset format for training** — how captured PLY episodes are labelled and stored for DQN replay buffer. *(Persons 2 + 4)*
