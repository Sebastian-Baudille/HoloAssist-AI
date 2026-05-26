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

The classical pipeline is working at ~1.6 cm centroid accuracy (within sensor
noise σ = 0.005 m).

```
load PLY ─► camera→world ─► Z crop ─► K-Means (XYZ only) ─► centroids
```

Key parameters and design decisions live in `detect_cubes.py` and are
summarised in the project `CLAUDE.md`. Worth highlighting:

- **`color_weight = 0`** — RGB clustering hurts under Gazebo lighting; XYZ
  alone separates cubes that are 0.16 m apart cleanly
- **`z_min = table_top + 0.015`** — 3σ noise margin excludes the table surface
- **Centroid via `cluster_pts.mean(axis=0)`** — same number as `pca.mean_` but
  explicit about what it is

Run it on the bundled sample:
```bash
python clustering/detect_cubes.py
```

---

## Why this isn't enough yet

The current pipeline assumes the cube layer contains *only* cubes. Three
things will break that assumption:

1. **The arm enters the cube layer** when reaching for a target. Z-cropping
   alone will treat arm points as cube points.
2. **Sensor noise on real hardware** produces blobs that pass the Z crop but
   aren't real objects.
3. **Multi-cube scenes with varying sizes / orientations / counts** — once we
   randomise the scene for RL training, fixed-`k` K-Means won't always match
   the true cube count.

We need a way to say: *"this cluster is a cube, that one is not."*

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

### Phase 4 — Auto-labelling (½ day)
A clustering-side script that consumes the captured dataset:
- Loads each PLY + `labels.json`
- Runs the existing K-Means pipeline
- Matches each cluster to a ground-truth object by nearest-neighbour
- Writes `features.csv`: per-cluster features (PCA extents, colour, size, bbox) + class label

This is where the dataset becomes ML-ready.

### Phase 5 — SVM training (1 day)
- Binary classifier first: cube vs not-cube
- 10-fold CV (quiz_1 style)
- Save model to `clustering/cube_classifier.pkl`
- Polyscope viz: colour clusters by SVM prediction to validate

### Phase 6 — Integration with RL (½ day)
Wire the SVM into `detect_cubes.py` as a filter step. Coordinate with the team on observation shape — feed only validated cube features to PPO.

---

### Summary

| Phase | What | Where | Effort |
|-------|------|-------|--------|
| 1 | `scene_controller` node + services | `ros2_ws/.../scripts/` | 1 day |
| 2 | `dataset_capture` node + labels | `ros2_ws/.../scripts/` | ½ day |
| 3 | rqt control panel | `ros2_ws/.../rqt/` | ½ day |
| 4 | `auto_label.py` | `clustering/dataset/` | ½ day |
| 5 | `train_svm.py` + validation | `clustering/dataset/` | 1 day |
| 6 | SVM filter in `detect_cubes.py` + RL integration | `clustering/` | ½ day |

**Total: ~4 days of focused work.**

---

## Design principles

- **Simple to explain.** Three nodes, four buttons, one classifier. Every
  step has a clear input, output, and reason for existing.
- **Reuse what works.** K-Means stays. PCA stays. The classical pipeline is
  not being replaced — it is being *complemented* with a learned filter.
- **Sim is the dataset source.** Ground truth is free in simulation. Real
  hardware is for validation and final tuning, not bulk data collection.
- **Perception and control stay separate.** The state vector is the
  contract between this stack and the RL stack. Either side can be
  redesigned without breaking the other.

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
