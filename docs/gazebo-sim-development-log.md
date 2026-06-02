# Gazebo Simulation Development Log — ROS 2 Ignition Fortress

**Project:** HoloAssist-AI (41118 AI in Robotics)  
**Author:** Ollie (ollie-lau) — Gazebo simulation; Guy (GuyESmith) — dataset capture  
**Period:** May–June 2026  
**Stack:** ROS 2 Humble + Ignition Fortress (ign-gazebo6) + ros_gz_bridge

---

## Overview

The Gazebo stack is the "ground truth" simulation environment for the project. Its responsibilities:

1. Render a physics-accurate scene with a UR3e + RG2 arm, a table, and colour-coded 4 cm cubes
2. Publish a synthetic RGBD point cloud over ROS 2 (mimicking an Intel D435i camera)
3. Provide a scene controller that randomises cube positions on demand
4. Capture labelled PLY files as training data for the clustering pipeline

The Gazebo sim serves as the data source — it is **not** used for RL training (that happens in MuJoCo / Isaac Sim). The output artefact is PLY files at `~/holoassist_pointclouds/` or `~/holoassist_dataset/`, bridging to the Python 3.11 clustering environment.

---

## Architecture

### Template-Driven World Generation

Rather than hardcoding a fixed SDF world file, the sim uses a **Jinja2 template pipeline**:

```
sim_params.yaml  (single source of truth)
    ↓
render_world.py  (Jinja2 renderer)
    ↓
table_cubes.sdf  (rendered into /tmp at launch time)
    ↓
gz sim -r table_cubes.sdf
```

This means changing the camera position, number of cubes, table size, or any physical property is a YAML edit — no SDF editing required. The same pattern applies to the ROS–Gazebo bridge config (`ros_gz_bridge.yaml.jinja2`).

`render_world.py` is called via `sys.executable` (not bare `python3`) so Jinja2 and PyYAML are guaranteed to be available in the same Python environment as the launch script.

### sim_params.yaml — Single Source of Truth

```
ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

All scene and camera parameters live here:

| Key | Purpose |
|---|---|
| `table.{size,pose,color}` | Table solid block geometry |
| `cubes[].{name,size,pose,color,mass}` | Free list — add/remove cubes without touching SDF |
| `camera.{name,pose,topic,...}` | D435i parameters (pose, FOV, resolution) |
| `camera.imu.{enabled,update_rate,topic}` | IMU sensor toggle |
| `capture.{output_dir,description,one_shot,voxel_size}` | PLY capture settings |

The default camera pose is:
```yaml
pose: [-0.6, 0.0, 0.85, 0.0, 0.5281, 0.0]  # x y z roll pitch yaw
```
This places the camera 0.6 m in front of the robot origin, 0.85 m up, tilted 30° downward (0.528 rad pitch) — a wide-angle view of the table surface.

### Launch Pipeline

`sim.launch.py` orchestrates everything:

1. Reads `sim_params.yaml`
2. Renders the SDF and bridge config into a temp directory
3. Spawns `gz sim -r <rendered.sdf>` (Ignition Fortress)
4. Spawns `ros_gz_bridge` with the rendered config
5. Publishes **two static TFs**: `map → d435i_link → d435i/d435i_link/rgbd`
6. Spawns RViz2 with the bundled rviz config
7. Spawns `scene_controller.py` for cube randomisation
8. Spawns `rqt_reconfigure` for live parameter editing

The two TFs are required because Ignition Fortress scopes the sensor frame as `<model>/<link>/<sensor_name>` — the published `frame_id` is `d435i/d435i_link/rgbd`, not a simple `camera_link`.

### TF Chain

```
map → d435i_link → d435i/d435i_link/rgbd
```

The body-frame pose (`camera.pose`) directly sets the `map → d435i_link` transform. The `d435i_link → d435i/d435i_link/rgbd` transform is identity (same physical location). This is a quirk of how Ignition Fortress names sensor frames — the double-nested path is mandated by the engine, not a design choice.

Camera body frame: **+X = look direction**, **Z = up**. This is the opposite convention from the standard ROS camera optical frame (+Z = forward). The optical frame transformation is handled by the bridge, not the TF chain.

### Camera Model

The simulated sensor mimics an Intel RealSense D435i:

```yaml
width: 848
height: 480
horizontal_fov_deg: 87.0
near_clip: 0.105   # ~10.5 cm min depth (matches real D435i)
far_clip: 10.0
update_rate: 30.0
```

Ignition Fortress's `rgbd_camera` uses a **single shared intrinsics block** for colour and depth — there are no separate RGB/depth intrinsics. Depth noise is Gaussian σ=0.005 m. This is baked into the sensor plugin and cannot be disabled.

Published topics: `/camera/image`, `/camera/depth_image`, `/camera/points`, `/camera/camera_info`, `/imu`

### Scene Controller

`scene_controller.py` is a ROS 2 node that manages cube spawning at runtime:

- Exposes `/scene/randomize_cubes` (Trigger service) — spawns N cubes at random positions
- Exposes `/scene/reset` — removes all random cubes
- Parameters (`cube_count`, `cube_size_min`, `cube_size_max`) editable live via `rqt_reconfigure`
- Overlap prevention: cubes are rejection-sampled with minimum separation = 2.0× cube size (was 1.5×, see below)

---

## Build

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash
```

`--symlink-install` means Python scripts and YAML files are symlinked from `src/` rather than copied, so edits take effect immediately without rebuilding.

## Run

```bash
# Start the full sim
ros2 launch holoassist_sim sim.launch.py

# Capture a single PLY frame
ros2 run holoassist_sim save_pointcloud.py --params ros2_ws/src/holoassist_sim/config/sim_params.yaml

# Capture a full labelled dataset (60 scenes)
ros2 run holoassist_sim dataset_capture --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

---

## PLY Capture Pipeline

### save_pointcloud.py

A ROS 2 node that subscribes to `/camera/points` (`PointCloud2`) and writes one PLY file per capture.

Key implementation details:
- Subscribed with `BEST_EFFORT` QoS (Ignition Fortress publishes sensor data with best-effort reliability by default; a reliable subscriber would drop all messages)
- RGB is packed in the ROS convention as a float32 bit-cast from `uint32 0x00RRGGBB` — the node unpacks it to `uint8 R, G, B`
- PLY is written as **binary little-endian** — not ASCII. MuJoCo and Open3D both expect binary STL/PLY; ASCII variants cause decoder failures downstream
- Optional voxel-grid downsampling (`voxel_size > 0`) using a hash-based uniqueness filter

**PLY naming convention:**
```
<capture.description>_<N>cubes_<size>mm_v<NNN>.ply
```
Example: `default_4cubes_40mm_v001.ply`. Version auto-increments so successive captures don't overwrite.

Output goes to `~/holoassist_pointclouds/` by default.

### dataset_capture.py — Automated Dataset Generation

A more advanced ROS 2 node that orchestrates 60 scenes (50 train + 10 val) automatically:

1. Sets `cube_count` parameter on `scene_controller` via direct service call
2. Calls `/scene/randomize_cubes` to spawn cubes
3. Waits 1 s for physics to settle
4. Reads **actual settled poses** from Gazebo via `ign topic -e -n 1 -t /world/<world>/dynamic_pose/info` (parsing the protobuf text output with regex)
5. Captures one PointCloud2 frame
6. Saves `scene_NNNN.ply` + `scene_NNNN.labels.json` with ground truth

The `labels.json` format:
```json
{
  "scene_id": "scene_0001",
  "split": "train",
  "cube_count": 3,
  "table_top_z": 0.5,
  "cubes": [
    {"name": "cube_rand_00", "position": {"x": 0.12, "y": -0.08, "z": 0.52},
     "orientation": {"x": 0, "y": 0, "z": 0, "w": 1}, "size_m": 0.04}
  ]
}
```

Cube size is back-calculated from settled Z position: `size = (cube_z - table_top_z) * 2`.

This approach — reading settled poses rather than commanded poses — correctly captures cubes that have drifted or stacked due to physics. It gave the clustering pipeline honest ground truth at the actual cube positions rather than nominal spawn positions.

---

## Problems Encountered

### Depth Noise and Z-Crop Margin

The camera has Gaussian depth noise σ=0.005 m. The initial clustering Z-crop cut exactly at `table_top_z`, which caused the table surface noise to bleed into the cube layer. The fix was a 3× noise margin:

```
z_min = table_top_z + 0.015  # 3 × σ
```

### BEST_EFFORT QoS Requirement

When the initial `save_pointcloud.py` used a reliable QoS subscriber for `/camera/points`, no messages were received. Ignition Fortress publishes sensor data with `BEST_EFFORT` reliability — a reliable subscriber advertises a different QoS profile and the bridge silently drops all matching attempts. Changing the subscriber to `BEST_EFFORT` fixed this immediately.

### Cube Separation — 1.5× → 2.0×

The initial scene controller used 1.5× cube size as the minimum separation between cube centres (≈ 2 cm surface gap for 4 cm cubes). When DBSCAN with `eps=0.015` was applied, cubes this close would often merge into a single cluster, causing DBSCAN to report 1 fewer cube than the ground truth. The separation was increased to 2.0× (≈ 4 cm surface gap) to ensure adjacent cubes stayed outside each other's neighbourhood radius.

### Frame Convention Confusion

The point cloud is published in **camera body frame** (+X forward, +Z up), not the standard ROS camera optical frame (+Z forward, -Y up). Early versions of the clustering pipeline tried to crop by world Z without first transforming from camera frame to world frame, producing completely wrong results. The correct pipeline step is:

```python
world_points = camera_to_world(camera_frame_points, camera_pose)
# THEN do Z-crop
```

This was documented in CLAUDE.md after validation.

### Static TF — Two Transforms Required

An early version only published one static TF (`map → camera_link`). The point cloud's `frame_id` was `d435i/d435i_link/rgbd` (the Ignition-scoped name), which had no TF chain to `map`. RViz showed no point cloud, and the transformation in Python had to be done manually. Adding the second TF (`camera_link → d435i/d435i_link/rgbd` as identity) fixed both RViz and established the correct chain for any tool that looks up TF.

---

## What We Would Change

### 1. Decouple Sensor Frame from Ignition Naming

The `frame_id` embedded in published PointCloud2 messages is `d435i/d435i_link/rgbd` — tied to Ignition's internal naming scheme. If the model name or camera name changes, the TF chain breaks silently. A wrapper node that re-publishes the cloud with a fixed `frame_id` like `camera_optical` would be more robust.

### 2. Dataset Size

60 scenes (50 train + 10 val) was the dataset used for evaluation. For a production clustering pipeline this is small. With `dataset_capture.py` the bottleneck is the 1 s settle time per scene, so scaling to 500+ scenes would take ~8 minutes — completely feasible.

### 3. Cube Variation

All cubes in the dataset have fixed 4 cm size and all cubes of a given colour are identical. Adding size variation (already supported by `scene_controller.py` via `cube_size_min` / `cube_size_max`) and random yaw during dataset capture would make the pipeline more robust to real-world variation.

### 4. Non-Jinja Rendering

The Jinja2 template pipeline was an elegant choice for a dynamic SDF but adds a dependency that must be available in the exact Python environment used by the launch file. If `render_world.py` is called with a different Python than the one that has Jinja2 installed, it fails silently. A simpler approach might be to generate the SDF programmatically in Python and write it directly, avoiding the template step.

---

## Discussion

### Why Ignition Fortress?

ROS 2 Humble's officially supported simulator is Ignition Fortress (ign-gazebo6), which replaced Gazebo Classic. The bridge (`ros_gz_bridge`) translates between Ignition's internal transport and ROS 2 topics. This added complexity (the two TF setup, BEST_EFFORT QoS quirks, scoped frame_id) compared to Gazebo Classic, but Fortress is the supported long-term choice for Humble.

### Why PLY as the Bridge Format?

The clustering pipeline runs in Python 3.11 (for Open3D and Polyscope compatibility) while ROS 2 Humble requires Python 3.10. Rather than trying to mix environments or call across processes, PLY files on disk serve as a clean, version-agnostic bridge. PLY is a binary format natively supported by Open3D, Polyscope, and NumPy, and carries both XYZ and RGB without any additional libraries. It also creates a natural checkpoint — captures can be reviewed and re-analysed without re-running the simulation.

### Why Not Use the RL Gazebo Stack?

Earlier in the project there was an attempt to use Gazebo Classic with a PPO training loop, with the robot controlled via keyboard teleoperation (`c5777c2`) and then via a MoveIt safety layer (`51bdd5d`). This was abandoned because Gazebo Classic is too slow for RL training (cannot run >1 environment in parallel without significant infrastructure) and the MoveIt overhead added latency that made tight control loops impractical. The RL training was moved to MuJoCo (John) and Isaac Sim (Seb), while Gazebo remained as the perception data source only.
