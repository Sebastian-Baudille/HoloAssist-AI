# Launch Commands

Quick reference for running the HoloAssist-AI pipeline. Open terminals from the repo root.

---

## First-time setup

```bash
chmod +x setup.sh && ./setup.sh
```

If Python 3.12 is not available (Ubuntu 22.04 default), create the clustering venv manually:

```bash
python3.11 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install -r clustering/requirements.txt
```

---

## Simulation — collect a point cloud

Requires two terminals.

**Terminal 1 — launch Gazebo + RViz:**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py
```

Wait for Gazebo and RViz to fully load before continuing.

**Terminal 2 — capture one frame to PLY:**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run holoassist_sim save_pointcloud.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

---

## Scene control (scene_controller)

The `sim.launch.py` automatically starts the **scene_controller** node and
`rqt_reconfigure`. Use rqt to edit settings live, then call services to
apply.

**Edit cube count / size / position bounds** — in the rqt_reconfigure window,
expand `/scene_controller`. Adjust sliders for `cube_count`, `cube_size_min/max`,
`x_min/max`, `y_min/max`, `randomize_yaw`, `randomize_color`.

**Apply the current settings:**
```bash
ros2 service call /scene/randomize_cubes std_srvs/srv/Trigger
```

**Restore the default 4-cube layout:**
```bash
ros2 service call /scene/reset std_srvs/srv/Trigger
```

Output file saved to `~/holoassist_pointclouds/` with auto-incremented version:
```
default_4cubes_40mm_v001.ply
default_4cubes_40mm_v002.ply  ← next capture, never overwrites
```

To label a dataset variant, edit `capture.description` in `sim_params.yaml` before capturing:
```yaml
capture:
  description: "spread"   # → spread_4cubes_40mm_v001.ply
```

---

## Clustering — Python 3.11 venv

All clustering commands require the venv to be activated first:

```bash
source clustering/.venv/bin/activate
```

**View a raw point cloud in Polyscope (no processing):**
```bash
python clustering/view_ply.py
# auto-picks most recent capture in ~/holoassist_pointclouds/
# falls back to clustering/sample_data/default_4cubes_40mm_v001.ply if no captures exist
python clustering/view_ply.py clustering/sample_data/default_4cubes_40mm_v001.ply
```

**Run cube detection — RANSAC + K-Means + PCA + Polyscope:**
```bash
python clustering/detect_cubes.py
# auto-picks most recent capture, or specify a file:
python clustering/detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply

# override number of cubes or colour weight (default 0 = XYZ only):
python clustering/detect_cubes.py -k 4 --color-weight 0.0

# skip Polyscope (terminal output only):
python clustering/detect_cubes.py --no-viz
```

---

## Rebuild the ROS workspace

Run after any changes to ROS package source files:

```bash
source /opt/ros/humble/setup.bash
cd ros2_ws
colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash
```

---

## Notes

- The ROS sim (Python 3.10) and clustering scripts (Python 3.11 venv) are **separate environments** — never mix them.
- Captured PLY files at `~/holoassist_pointclouds/` are the handoff point between the two environments.
- Scene parameters (cube positions, camera pose, capture settings) live in one file: `ros2_ws/src/holoassist_sim/config/sim_params.yaml`.
- The Gazebo world SDF is generated at launch from `sim_params.yaml` — never edit the generated SDF directly.
