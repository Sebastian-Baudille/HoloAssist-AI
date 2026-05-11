# HoloAssist-AI

41118 AI in Robotics. ROS 2 Humble sim → PLY capture → Python 3.11 clustering/Polyscope.

## Environments

| | Python | Key constraint |
|---|---|---|
| ROS sim + capture | 3.10 (Humble) | No open3d/polyscope imports |
| Clustering / Polyscope | 3.11 (`clustering/.venv`) | No rclpy imports |
| Bridge: PLY files at `~/holoassist_pointclouds/` | | |

## Critical files

```
ros2_ws/src/holoassist_sim/
  config/sim_params.yaml          single source of truth — scene + camera + capture
  config/ros_gz_bridge.yaml.jinja2
  worlds/table_cubes.sdf.jinja2   rendered at launch; never edit output SDF
  launch/sim.launch.py            renders templates → starts gz/bridge/TF/rviz
  scripts/render_world.py         CLI: PARAMS TEMPLATE OUTPUT
  scripts/save_pointcloud.py      ROS node → PLY; reads params yaml
clustering/requirements.txt       numpy==2.4.3 open3d==0.19.0 scikit-learn==1.8.0 polyscope==2.6.1
clustering/view_ply.py            load PLY → world frame transform → Polyscope viewer
clustering/detect_cubes.py        full pipeline: load → crop → K-Means → centroids → Polyscope
LAUNCH.md                         quick-reference launch commands for sim + clustering
```

## sim_params.yaml keys

```
table.{size,pose,color}
cubes[].{name,size,pose,color,mass}          # free list
camera.{name,pose,topic,update_rate,width,height,horizontal_fov_deg,near_clip,far_clip}
camera.imu.{enabled,update_rate,topic}
capture.{output_dir,description,one_shot,voxel_size}
```

## Build

```bash
cd ros2_ws && source /opt/ros/humble/setup.bash
colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash
```

## Run

```bash
ros2 launch holoassist_sim sim.launch.py [params_file:=/abs/path.yaml]
ros2 run holoassist_sim save_pointcloud.py --params <yaml>
```

## ROS topics

`/camera/{image,depth_image,points,camera_info}` · `/imu` · `/clock` · `/tf_static`

## TF chain

`map → d435i_link → d435i/d435i_link/rgbd`  (two static_transform_publishers in launch)  
Camera +X = look direction. `camera.pose` pitch 0.528 rad ≈ 30° downward tilt.  
Scoped frame_id from Ignition Fortress: `<model>/<link>/<sensor_name>`

## PLY naming

`<capture.description>_<N>cubes_<size>mm_v<NNN>.ply`  
Version auto-increments. Functions: `_build_base_name(params)`, `_next_version(dir, base)`

## Template rendering

`render_world.py` is called by `sim.launch.py` via `sys.executable` (not bare `python3`).  
Script located via `get_package_prefix(PKG) / lib / PKG / render_world.py`.  
Templates use Jinja2; Gazebo inertia is computed inline (`m/12*(a²+b²)`).

## Sim facts

- Ignition Fortress (ign-gazebo6) + ros_gz_bridge
- rgbd_camera shares one FOV/resolution for colour+depth (no separate RGB intrinsics)
- Depth noise: Gaussian σ=0.005 m
- Cubes: dynamic, mass=0.05 kg, μ=0.8; table: static
- Cube centre Z default = 0.52 m (table top 0.5 + half-cube 0.02)
- Units: metres, radians throughout

## Clustering pipeline (validated)

Point cloud is published in **camera body frame** (+X = look direction). Must transform to
world frame before any height-based cropping.

```
camera_to_world → crop Z → K-Means (XYZ only) → mean per cluster → DQN state vector
```

Key parameters (detect_cubes.py):
- `z_min = table_top_z + 0.015`  — 3× sensor noise margin to exclude table surface
- `z_max = table_top_z + cube_height + 0.01`
- `color_weight = 0.0`  — XYZ-only clustering; RGB weighting confuses K-Means under Gazebo lighting
- Centroid = `cluster_pts.mean(axis=0)` — world-frame (x, y, z) fed directly to DQN
- PCA still used for orientation axes (Polyscope visualisation only)

Validated accuracy: ~1.6 cm centroid error vs ground truth (within sensor noise σ=0.005 m).
DQN state vector = centroids.flatten(), shape (k*3,) = (12,) for 4 cubes.
