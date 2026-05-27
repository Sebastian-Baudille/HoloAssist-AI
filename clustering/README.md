# Clustering Stack — Roadmap

This document describes the clustering side of HoloAssist-AI: where it is today,
where it's going, and how the pieces fit together. It is the planning document
for the perception module that feeds the RL policy.

---

## Mission

Take a point cloud captured from a RealSense (or its Gazebo simulation), find
the cubes on the table, and produce a clean **state vector** that the PPO
policy can use as an observation. Perception is separated from control so each
can be developed, debugged, and replaced independently.

```
PLY ─► clustering stack ─► state vector ─► PPO ─► arm action
```

---

## Current state — validated

### Phase 0 → 0b: DBSCAN pipeline (`clustering/detect_cubes.py`) ✅
Replaced K-Means with DBSCAN + statistical outlier removal. No fixed `k` required —
DBSCAN discovers however many cubes are present automatically.

```
load PLY ─► camera→world ─► Z crop ─► outlier removal ─► DBSCAN ─► size filter ─► centroids
```

Key parameters (`detect_cubes.py`):
- **`eps = 0.015 m`** — DBSCAN neighbourhood radius; cubes must have ≥ 4 cm surface gap
  (i.e., centre spacing ≥ 2× size) or their point clouds merge into one cluster
- **`min_samples = 20`** — minimum points to form a core sample
- **`MIN_CLUSTER_PTS = 50 / MAX_CLUSTER_PTS = 1500`** — size filter rejects noise and non-cube objects
- **`remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)`** — kills flying pixels before clustering
- **`z_min = table_top + 0.015`** — 3σ noise margin excludes the table surface
- Centroid = `cluster_pts.mean(axis=0)` — world-frame (x, y, z) fed to PPO

```bash
python clustering/detect_cubes.py          # latest capture in ~/holoassist_pointclouds/
python clustering/detect_cubes.py --no-viz  # headless
python clustering/detect_cubes.py --eps 0.015 --min-samples 20
```

### Phase 1: Scene controller (`ros2_ws/src/holoassist_sim/scripts/scene_controller.py`) ✅
ROS 2 node that manages the Gazebo scene. Started automatically by `sim.launch.py`.

- `/scene/randomize_cubes` — spawns N cubes with random position, colour, yaw; overlap-safe
  (min **2.0× cube size** centre-to-centre separation = ≥ 4 cm surface gap for DBSCAN safety)
- `/scene/reset` — restores default layout from `sim_params.yaml`
- All parameters (cube count, size bounds, position bounds) editable live via `rqt_reconfigure`

### Phase 2: Automated dataset capture (`ros2_ws/src/holoassist_sim/scripts/dataset_capture.py`) ✅
ROS 2 node that runs 60 scenes unattended and saves ground truth.

- Randomises 2–4 cubes per scene (50 train + 10 val split)
- Removes default SDF cubes so only the random cubes are in each scan
- Waits for physics to settle, then reads **actual settled poses** from Gazebo via `ign topic`
- Saves `scene_NNNN.ply` + `scene_NNNN.labels.json` per scene to `~/holoassist_dataset/`

```bash
# With sim already running:
ros2 run holoassist_sim dataset_capture.py \
  --params src/holoassist_sim/config/sim_params.yaml
```

### Phase 2b: Detection verification (`clustering/verify_detection.py`) ✅
Clustering-venv script that validates DBSCAN detection accuracy against the labelled dataset.

- Loads each PLY + labels, runs the full DBSCAN pipeline (matches `detect_cubes.py` exactly)
- Hungarian-matches detected centroids to ground truth; handles variable detected count
- Reports per-split stats: accuracy, exact-count rate, cube recall, false positives
- Saves `~/holoassist_dataset/accuracy_report.json`

**DBSCAN results — first dataset (2026-05-27, 60 scenes, 2–4 cubes, 0.04 m fixed size,**
**old `min_dist = 1.5× size` → some scenes had 2 cm surface gap, bridged by DBSCAN):**

| Split | Count correct | Cube recall | Mean error (detected) | Mean error (all) |
|-------|:---:|:---:|:---:|:---:|
| Train (50) | 41/50 (82%) | 127/140 (91%) | **1.63 cm** | 5.81 cm |
| Val (10) | 10/10 (100%) | 30/30 (100%) | **1.63 cm** | 1.63 cm |

**Root cause of 9 train failures**: 9 scenes had cube centres ≤ 6 cm apart (surface gap ≤ 2 cm).
DBSCAN `eps=0.015` bridges a 2 cm gap (within noise + eps), merging adjacent cubes into one
large cluster. The scene controller now uses `min_dist = 2.0× size` (≥ 4 cm surface gap),
which fully separates cube point clouds. **Regenerate the dataset to get clean DBSCAN numbers.**

**Expected results after dataset regeneration:**

| Split | Count correct | Cube recall | Mean error |
|-------|:---:|:---:|:---:|
| Train | ~100% | ~100% | **~1.63 cm** |
| Val | 100% | 100% | **1.63 cm** |
| **Target** | — | — | **< 3 cm → PASS** |

```bash
python3 clustering/verify_detection.py
python3 clustering/verify_detection.py --eps 0.015 --min-samples 20
```

---

## Why DBSCAN instead of K-Means

| | K-Means | DBSCAN |
|---|---|---|
| Needs fixed `k` | ✅ yes — must know cube count | ❌ no — finds clusters automatically |
| Handles variable scene size | ❌ no | ✅ yes |
| Rejects noise | ❌ every point forced into a cluster | ✅ yes — noise label `-1` |
| Merges adjacent objects | ❌ never (k fixed) | ⚠️ yes if gap < `eps` |
| Accuracy (this dataset) | 2.65 cm | 1.63 cm (when count correct) |

DBSCAN is the right choice for the RL observation pipeline: the arm scans at home pose where
cubes are well-separated, and DBSCAN naturally rejects stray noise points. The SVM classifier
is no longer needed for basic detection in a controlled workspace.

---

## Why the arm-in-scene problem is solved by scan timing

The arm enters the cube layer when picking up a target. We scan *before* the arm moves
(at the start of each episode, arm at home pose). This window has no arm points in the
cube layer — DBSCAN + Z-crop is sufficient. No classifier needed.

---

## Plan — next stages

### Stage A: Build a labelled dataset
We can't train a classifier without data. Rather than capture from real
hardware (expensive, slow to label), we generate it in simulation where ground
truth is free.

**Mechanism:** a single Gazebo session with a control GUI. Buttons trigger
ROS services that randomise cubes, move the arm, and capture point clouds
labelled with full ground truth.

Output:
```
~/holoassist_dataset/
  scene_0001.ply
  scene_0001.labels.json   # cube poses/sizes/colours, arm pose
  scene_0002.ply
  scene_0002.labels.json
  ...
```

### Stage B: Train an SVM classifier
Take the labelled dataset, extract per-cluster features (PCA extents, colour,
size, bounding box), train a binary SVM (cube vs not-cube). Validate with
10-fold cross-validation — same recipe as quiz_1, just on our data.

The SVM lives between K-Means and the state vector:
```
K-Means clusters ─► feature extraction ─► SVM filter ─► validated cubes ─► state vector
```

### Stage C: Integrate with PPO
Coordinate with the RL team on the observation format. The SVM output
defines what goes into the observation; the policy is trained on top.

### Stage D: Sim-to-real with RealSense
The pipeline structure doesn't change. Real captures replace simulated ones,
the SVM is retrained or fine-tuned on real data, and the PPO policy ideally
transfers unchanged — that's the whole point of separating perception from
control.

---

## Architecture once Stage A–B are complete

```
                              ┌──────────────────────────────────┐
   Gazebo sim                 │  scene_controller (ROS node)     │
   (single instance)  ◄──────►│  • randomize_cubes service       │
                              │  • set_arm_pose service          │
                              │  • reset service                 │
                              └──────────────┬───────────────────┘
                                             │
                              ┌──────────────▼───────────────────┐
                              │  dataset_capture (ROS node)      │
                              │  • capture service               │
                              │  • saves PLY + labels.json       │
                              └──────────────┬───────────────────┘
                                             │
   ┌─────────────────┐                       │
   │ rqt control GUI │──────────────────────►│
   │ 4 buttons       │ ROS services          │
   └─────────────────┘                       │
                                             ▼
                                  ~/holoassist_dataset/
                                             │
                              ┌──────────────▼───────────────────┐
                              │  auto_label.py (clustering venv) │
                              │  clusters + ground truth →       │
                              │  features.csv                    │
                              └──────────────┬───────────────────┘
                                             │
                              ┌──────────────▼───────────────────┐
                              │  train_svm.py                    │
                              │  10-fold CV → cube_classifier.pkl│
                              └──────────────┬───────────────────┘
                                             │
                                             ▼
                              ┌──────────────────────────────────┐
                              │  detect_cubes.py                 │
                              │  K-Means → SVM filter →          │
                              │  state vector → PPO              │
                              └──────────────────────────────────┘
```

Three new ROS nodes, two new clustering scripts, one trained model file. The
existing `detect_cubes.py` gets a single new step (SVM filter) — the rest of
its pipeline is unchanged.

---

## Phased delivery

### Phase 1 — Scene control services (1 day)
A single ROS 2 node `scene_controller.py` that owns the scene state and exposes three services:
- `/scene/randomize_cubes` — remove existing cubes, spawn N new ones with sampled size/yaw/colour/position
- `/scene/set_arm_pose` — move the UR3e to a target pose (calls team's MoveIt service)
- `/scene/reset` — restore default 4-cube layout

Internally calls Gazebo's `/world/<name>/create` and `/remove` services. Cubes are spawned from a small Jinja2 template (single cube), reusing existing template machinery.

### Phase 2 — Capture-and-label service (½ day)
A node `dataset_capture.py` exposing `/dataset/capture`. On call:
1. Wait for next `/camera/points` message
2. Save as PLY (auto-versioned, same as current `save_pointcloud.py`)
3. Query `scene_controller` for current ground truth (cube poses, sizes, colours, arm pose)
4. Write `labels.json` next to the PLY

This extends the existing `save_pointcloud.py` — adds ground-truth labels alongside each capture.

### Phase 3 — Control GUI (½ day)
A **rqt panel** with four buttons:
- **Randomise Cubes** → calls `/scene/randomize_cubes`
- **Random Arm Pose** → calls `/scene/set_arm_pose` with a sampled pose
- **Capture** → calls `/dataset/capture`
- **Auto-batch N** → loops randomise → capture for N iterations

rqt is the ROS-native way to do this; it docks next to RViz cleanly. If rqt feels heavy, a **20-line Tkinter window** does the same thing — both options are simple to explain (button click → service call).

### Phase 4 — RL integration (½ day)
Wire `detect_cubes.py` (DBSCAN pipeline) into the RL observation loop. Coordinate with the
team on observation shape — the DBSCAN output defines the state vector fed to PPO.

---

### Summary

| Phase | What | Where | Status |
|-------|------|-------|--------|
| 1 | `scene_controller` node + services | `ros2_ws/.../scripts/` | ✅ Done |
| 2 | `dataset_capture` node + labels | `ros2_ws/.../scripts/` | ✅ Done |
| 2b | `verify_detection.py` — DBSCAN accuracy benchmark | `clustering/` | ✅ Done — 1.63 cm on well-separated cubes |
| 0b | DBSCAN pipeline in `detect_cubes.py` | `clustering/` | ✅ Done |
| 2c | Regenerate dataset with `min_dist = 2.0×` | run `dataset_capture.py` | ⬜ Pending |
| 3 | rqt / Tkinter control panel | `ros2_ws/.../rqt/` | ⬜ Not started |
| 4 | RL integration — DBSCAN → PPO observation | `clustering/` + RL stack | ⬜ Not started |

---

## Design principles

- **Simple to explain.** DBSCAN + Z-crop + outlier removal. No learned classifier needed
  for a controlled lab workspace.
- **Scan at home pose.** The perception window is before the arm moves — no arm-in-scene
  contamination.
- **Sim is the dataset source.** Ground truth is free in simulation. Real hardware is for
  validation and final tuning, not bulk data collection.
- **Perception and control stay separate.** The state vector is the contract between this
  stack and the RL stack. Either side can be redesigned without breaking the other.

---

## Storage

- **Captures (raw PLYs)** — `~/holoassist_pointclouds/` (single captures, dev
  use, gitignored)
- **Dataset (PLYs + labels)** — `~/holoassist_dataset/` (bulk generation,
  gitignored; small sample tracked in `clustering/sample_data/`)
- **Trained models** — `clustering/cube_classifier.pkl` (committed; small
  enough that LFS is not needed)

---

## Open questions

1. **rqt panel vs Tkinter GUI** — rqt is more ROS-native; Tkinter is simpler
   code. Either works; decide before Phase 3.
2. **Arm model for negative examples** — use the team's real UR3e URDF, or a
   primitive arm-shaped placeholder? Real URDF is more transferable.
3. **Multi-class from day one or start binary?** Binary (cube/not-cube) is
   simpler; multi-class (per colour) is more useful downstream. Recommend
   storing all ground truth and deciding at training time.
4. **Dataset size target** — 200 for proof-of-concept, 500–1000 for trained
   classifier. Decide after Phase 1 latency is measured.
5. **Headless Gazebo** — verify compatibility with the team's
   `IGN_PARTITION` setup before relying on it for bulk capture.

---

## What this is not

- Not a deep learning pipeline. Pre-trained pose models (FoundationPose,
  DOPE) and end-to-end visuomotor RL are documented in `architecture.md` as
  future directions, not in scope here.
- Not a replacement for the team's RL work. This stack feeds it.
- Not the final perception system. It is the proving ground for what the
  final system needs to do.
