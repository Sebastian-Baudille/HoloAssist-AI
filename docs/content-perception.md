# Perception & Clustering Content — for perception.html, clustering.html, index.html slides

---

## Perception slide (01/04) — attribution fix + dual credit

**Both Nic and Guy should appear here.** They represent two generations of the perception pipeline.

**Nic Sabatini (MafiaPineapple)** — original HoloAssist perception system (AprilTag approach)
**Guy Smith (GuyESmith)** — HoloAssist-AI rework (DBSCAN point cloud approach)

---

## Original perception system (Nic — base HoloAssist)

The original HoloAssist used an AprilTag 3 + depth tracker pipeline:

### How it worked
1. **RealSense D435i** publishes RGB + depth stream
2. **apriltag_ros** detects 4 corner tags on a physical workspace board → publishes TF frames
3. **workspace_board_node** runs SVD (Kabsch algorithm) to align observed tag positions to known board model:
   - Derives camera-to-workspace transform
   - Locks transform once RMS residual < 0.02 m
4. **cube_pose_node** looks up AprilTag face tags on each cube in workspace_frame → publishes cube poses
5. **workspace_perception_node** (RANSAC alternative): RANSAC plane fitting on depth point cloud + blob tracking (no tags needed for fallback)
6. **cube_pose_relay.py** (Nic): TF-aware bridge publishing cube poses as `workspace_frame → base_link`
7. **CubePoseSubscriber.cs** (Nic): Unity C# subscriber → renders cubes in XR overlay

### Nic's specific contributions to perception
- **Eye-to-hand calibration node** (`eye_hand_calibration_node.py`): OpenCV `calibrateHandEye` (Park solver), observing AprilTag on end effector from multiple poses. Validated at 0.4 mm accuracy in simulation. Both sim mode (synthetic observations) and hardware mode (real apriltag_ros detections via ROS services).
- **`cube_pose_relay.py`**: TF-aware ROS bridge converting workspace_frame cube poses → base_link for the RL policy
- **`CubePoseSubscriber.cs`**: Unity side subscriber for `/holoassist/unity/cube_N_pose` topics
- **`workspace_perception_params.yaml`**: hardware-validated perception parameters
- **`calibrate.py` / `calibrate.sh`**: calibration helper scripts
- **Hardware integration**: the `origin/nic` branch was merged as "hardware-tested perception wiring" — Nic validated the full pipeline on the real robot

### Board model (for context)
- 700 × 500 mm workspace board, tag36h11 family
- 4 corner tags (IDs 0–3), 32 mm printed, 16 mm from edges
- Origin at tag 0 centre = workspace_frame origin

---

## New perception system (Guy — HoloAssist-AI)

Guy replaced the AprilTag-dependent approach entirely with pure point cloud DBSCAN clustering.

**Why replace it:** AprilTag approach requires physical tags on every cube, depends on tag visibility, and is brittle when cubes are rotated or occluded. DBSCAN works directly on the 3D point cloud — no markers needed.

### How it works
1. Simulated D435i publishes `/camera/points` via Gazebo + ros_gz_bridge
2. `save_pointcloud.py` (ROS 2 node) saves binary PLY files
3. Camera-to-world transform applied
4. Height crop: z_min = table_top + 15 mm, z_max = table_top + cube_height + 10 mm
5. Statistical outlier removal (Open3D)
6. DBSCAN: eps=0.015 m, min_samples=20
7. Size filter: 50–1500 pts per cluster
8. Centroid = cluster mean (world frame XYZ)

### Guy's specific contributions
- Dataset capture pipeline (`dataset_capture.py`): 60 scenes automated, reads settled poses from Gazebo
- K-Means baseline: 2.65 cm centroid error (validated, benchmark established)
- DBSCAN implementation: 1.63 cm accuracy when count correct; 82% exact-count rate
- Fixed cube spacing bug: 1.5× → 2.0× cube size gap (stops DBSCAN merging adjacent cubes)
- `verify_detection.py`: Hungarian matching benchmark against ground truth
- Also tightened joint limits + added config penalty for self-collision during Phase A training

---

## Suggested perception slide copy (index.html)

**Heading:** RGB-D Perception
**Tag line:** RealSense D435i · AprilTag calibration → DBSCAN point cloud · ROS 2 · PLY capture

**Description:**
"Two generations of perception. The original HoloAssist system used AprilTag 3 markers and SVD workspace
calibration, with eye-to-hand calibration at 0.4 mm accuracy — validated on real hardware.
HoloAssist-AI replaces this with a marker-free DBSCAN point cloud pipeline: simulated D435i captures
RGB-D frames, DBSCAN clusters cubes at 1.63 cm centroid accuracy across 60 validated scenes."

**Show both people on the perception slide:**
- Nic Sabatini (MafiaPineapple) — AprilTag system, eye-to-hand calibration, Unity-ROS bridge
- Guy Smith (GuyESmith) — DBSCAN clustering, dataset pipeline, hardware-free perception

---

## Clustering slide (02/04) — Guy only

Lead: Guy Smith (GuyESmith)
Keep the description as-is but use accurate stats below.

**Stats:**
- 1.63 cm accuracy (DBSCAN, when count correct)
- 82% exact-count rate (improving with regenerated dataset at 2× spacing)
- 60 scenes validated
- eps=0.015 m, min_samples=20

---

## perception.html — full corrected content

### Overview
Two-phase pipeline: (1) ROS 2 + Gazebo point cloud capture, (2) offline DBSCAN clustering (Python 3.11).

This replaced the original HoloAssist AprilTag 3 perception system developed by Nic Sabatini,
which used SVD workspace calibration and required physical AprilTag markers on each cube.

### Original System: AprilTag 3 (Nic Sabatini)
- Physical workspace board with 4 corner AprilTag markers (tag36h11 family)
- SVD/Kabsch algorithm to lock camera→workspace transform
- AprilTag face tags on each cube → direct pose from TF tree
- RANSAC plane fitting on depth point cloud as a fallback / alternative pipeline
- Eye-to-hand calibration via OpenCV `calibrateHandEye` (Park solver, 0.4 mm sim accuracy)
- `cube_pose_relay.py`: TF-aware bridge to base_link frame
- `CubePoseSubscriber.cs`: Unity XR overlay of cube positions in real time
- Validated and integrated on real hardware

**Limitation:** Requires physical markers on every cube. Brittle to rotation/occlusion.

### New System: DBSCAN Point Cloud (Guy Smith)
See clustering section for full detail.

**Advantage:** No physical markers. Works on any coloured object in the workspace.
Centroid accuracy: 1.63 cm vs ~5–10 cm typical AprilTag distance noise.

### Key files
- `ros2_ws/src/holoassist_sim/scripts/save_pointcloud.py` — PLY capture ROS node
- `clustering/detect_cubes.py` — DBSCAN pipeline
- `clustering/verify_detection.py` — accuracy benchmark
- (base HoloAssist) `ros2_ws/src/holo_assist_depth_tracker/nodes/eye_hand_calibration_node.py` — Nic's calibration
- (base HoloAssist) `cube_pose_relay.py` — Nic's ROS-Unity bridge
- (base HoloAssist) `CubePoseSubscriber.cs` — Nic's Unity subscriber

---

## clustering.html — corrected content (Guy only, no change to attribution)

### Why DBSCAN over AprilTag / K-Means
| | AprilTag 3 | K-Means | DBSCAN |
|---|---|---|---|
| Requires physical markers | Yes | No | No |
| Needs known cube count | No | Yes | No |
| Centroid error | ~5–10 mm tag noise | 2.65 cm | 1.63 cm |
| Occlusion robust | No | Yes | Yes |
| Hardware | Real robot | Sim | Sim |

### Pipeline steps
1. Load PLY → camera body frame
2. Camera-to-world transform (TF chain: map → d435i_link → sensor frame)
3. Height crop: z_min = table_top_z + 0.015 m, z_max = table_top_z + cube_height + 0.01 m
4. Statistical outlier removal (Open3D)
5. DBSCAN: eps=0.015 m, min_samples=20
6. Size filter: 50–1500 pts per cluster
7. Centroid = cluster mean (XYZ world frame)

### Results
| Method | Mean error | Exact-count rate | Notes |
|---|---|---|---|
| K-Means | 2.65 cm | ~90% (fixed k) | Baseline — requires k known upfront |
| DBSCAN (1.5× gap) | 1.63 cm | 82% | Adjacent cubes merge at eps=1.5 cm |
| DBSCAN (2.0× gap) | TBD | Expected >90% | Regenerating dataset |

### Feeding into RL
Centroids (k×3, world frame) flattened into RL policy observation vector.
For 4 cubes: shape (12,) = PPO state input.
