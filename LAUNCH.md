# Launch Commands

Command reference for the full HoloAssist-AI pipeline. Open terminals from the repo root.

For system design and stage status, see [ARCHITECTURE.md](ARCHITECTURE.md). For current
in-flight work and training results, see [PROGRESS.md](PROGRESS.md).

---

## Contents

- [First-time setup](#first-time-setup)
- [Perception sim — collect a point cloud](#perception-sim--collect-a-point-cloud)
- [Automated labelled dataset capture](#automated-labelled-dataset-capture)
- [Clustering — Python 3.12 venv](#clustering--python-312-venv)
- [Detection accuracy benchmark](#detection-accuracy-benchmark)
- [RL training](#rl-training)
- [RL evaluation and deployment](#rl-evaluation-and-deployment)
- [Real D435i — live perception runbook](#real-d435i--live-perception-runbook)
- [Rebuild a workspace](#rebuild-a-workspace)
- [Notes](#notes)

---

## First-time setup

```bash
chmod +x setup.sh && ./setup.sh
```

If Python 3.12 is not available, create the clustering venv manually:

```bash
python3.12 -m venv clustering/.venv
source clustering/.venv/bin/activate
pip install -r clustering/requirements.txt
```

---

## Perception sim — collect a point cloud

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

Output saved to `~/holoassist_pointclouds/` with auto-incremented version:
```
default_4cubes_40mm_v001.ply
default_4cubes_40mm_v002.ply   ← next capture, never overwrites
```

To label a dataset variant, edit `capture.description` in `sim_params.yaml`:
```yaml
capture:
  description: "spread"   # → spread_4cubes_40mm_v001.ply
```

---

## Automated labelled dataset capture

`dataset_capture.py` runs N scenes unattended and saves PLY + JSON ground truth, used by
`clustering/verify_detection.py`.

**Terminal 1 — launch sim:**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 launch holoassist_sim sim.launch.py
```

**Terminal 2 — run capture (waits for sim to be ready):**
```bash
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash
ros2 run holoassist_sim dataset_capture.py \
    --params ros2_ws/src/holoassist_sim/config/sim_params.yaml
```

To start with a clean dataset, clear the old one **before** Terminal 2:
```bash
rm ~/holoassist_dataset/scene_*.ply \
   ~/holoassist_dataset/scene_*.json \
   ~/holoassist_dataset/accuracy_report.json
```

---

## Clustering — Python 3.12 venv

Activate the venv first:
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

**Run cube detection — DBSCAN + PCA + Polyscope:**
```bash
python clustering/detect_cubes.py
# auto-picks most recent capture, or specify a file:
python clustering/detect_cubes.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply

# parameter overrides:
python clustering/detect_cubes.py --eps 0.015 --min-samples 20

# skip Polyscope (terminal output only):
python clustering/detect_cubes.py --no-viz
```

---

## Detection accuracy benchmark

After a `dataset_capture` run, validate DBSCAN accuracy vs labelled ground truth:

```bash
source clustering/.venv/bin/activate
python clustering/verify_detection.py
```

Outputs per-split stats and writes `~/holoassist_dataset/accuracy_report.json`. Current
baseline: 1.63 cm mean error, 100 % recall on 60 scenes — see
[clustering/README.md](clustering/README.md).

---

## RL training

The PPO training stack ([ur3e_rl_ws/](ur3e_rl_ws/)) lives in the ROS-sourced Python 3.10 —
it talks to live ROS topics published by a Gazebo instance.

### One-time Python deps
```bash
python3 -m pip install gymnasium stable-baselines3 tensorboard
```

### Single-environment training (1 terminal sim + 1 terminal train, ~6 h on a laptop)

**Terminal 1 — Gazebo + bridges:**
```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py gui:=false
```

**Terminal 2 — PPO trainer:**
```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
UR3E_RL_TOTAL_TIMESTEPS=10000 ros2 run ur3e_rl_env train_ppo
```

### Parallel training — 4 Gazebo workers (~30 min on a 20-thread laptop)

`train_ppo_parallel` spawns its own headless Gazebo instances on `ROS_DOMAIN_ID` 30–33 and
Gazebo master ports 11400–11403. Stop any other Gazebo launches before starting.

```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
pkill -f "ign|gz|train_ppo_parallel" || true

UR3E_RL_NUM_ENVS=4 \
UR3E_RL_TOTAL_TIMESTEPS=500000 \
UR3E_RL_PPO_N_STEPS=2048 \
UR3E_RL_PPO_BATCH_SIZE=256 \
ros2 run ur3e_rl_env train_ppo_parallel
```

### Parallel training with MoveIt collision checking

MoveIt's `/check_state_validity` is domain-scoped — one move_group instance per worker
domain. Open four extra terminals, one for each domain:

```bash
# Repeat for ROS_DOMAIN_ID in 30, 31, 32, 33
export ROS_DOMAIN_ID=30   # or 31 / 32 / 33
cd ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
    ur_type:=ur3e onrobot_type:=rg2 \
    use_sim_time:=false launch_rviz:=false launch_servo:=false
```

Wait for all four `move_group` to finish initialising, then start the trainer:
```bash
UR3E_RL_NUM_ENVS=4 \
UR3E_RL_USE_MOVEIT_COLLISION_CHECKER=1 \
UR3E_RL_MOVEIT_GROUP_NAME=ur_onrobot_manipulator \
UR3E_RL_MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE=1 \
ros2 run ur3e_rl_env train_ppo_parallel
```

### Behaviour-cloning warm-start from teleop demos

```bash
# Step 1 — record demos (UR driver + Unity teleop must be live)
ros2 run ur3e_rl_env record_demo

# Step 2 — behaviour-clone into a PPO model
ros2 run ur3e_rl_env pretrain_from_demos
# output: ./rl_models/ppo_ur3e_pretrained
```

Then continue with `train_ppo` / `train_ppo_parallel` starting from the pretrained model.

### Monitor with TensorBoard

```bash
tensorboard --logdir ur3e_rl_ws/tb_logs
# open http://localhost:6006
```

Watch `ep_rew_mean`, `ep_len_mean`, `rollout/success_rate`, and the PPO loss terms.

### Key environment variables

```bash
UR3E_RL_NUM_ENVS=4                  # parallel workers (default 4)
UR3E_RL_BASE_ROS_DOMAIN_ID=30       # worker 0 domain; others increment
UR3E_RL_TOTAL_TIMESTEPS=500000      # total PPO timesteps
UR3E_RL_CONTROL_DT=0.1              # seconds per action step
UR3E_RL_MAX_EPISODE_STEPS=200       # steps before forced reset
UR3E_RL_PPO_N_STEPS=2048            # rollout buffer size per env
UR3E_RL_PPO_BATCH_SIZE=256          # minibatch size for SGD
UR3E_RL_TORCH_THREADS=2             # keep low — bottleneck is sim, not torch
UR3E_RL_RANDOMIZE_CUBE=1            # randomise cube pose per episode
```

See [ur3e_rl_ws/TRAINING_NOTES.md](ur3e_rl_ws/TRAINING_NOTES.md) for action-scale rationale
and parallel-training internals.

---

## RL evaluation and deployment

### Evaluate a trained policy (20 episodes)

```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur3e_rl_env evaluate_policy
```

### Run a trained policy live (sim or real)

```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# default model: ./rl_models/ppo_ur3e_reach_object
ros2 run ur3e_policy_controller rl_policy_node

# custom model:
UR3E_RL_MODEL_PATH=./rl_models/ppo_best \
    ros2 run ur3e_policy_controller rl_policy_node
```

The policy node reads the current observation, predicts an action, clamps via
`SafetyChecker`, and publishes a joint target every `0.2 s`.

---

## Real D435i — live perception runbook

The `cube_perception` package replaces `save_pointcloud.py` + `detect_cubes.py` for live
hardware: subscribes to a streaming D435i, runs the same DBSCAN pipeline at ~10 Hz,
publishes per-cube poses with confidence and occlusion hold.

### One-time build

```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select cube_perception --symlink-install
source install/setup.bash
```

### Launch the full stack (TF + camera + perception)

```bash
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch cube_perception cube_perception.launch.py
```

If the D435i node is already running elsewhere:
```bash
ros2 launch cube_perception cube_perception.launch.py start_camera:=false
```

If the static TF is already being published:
```bash
ros2 launch cube_perception cube_perception.launch.py \
    start_camera:=false publish_static_tf:=false
```

### Verify output

```bash
# TF
ros2 run tf2_ros tf2_echo base_link camera_link

# Topics
ros2 topic list | grep -E '^/cube_|/cube_detections'
ros2 topic hz /cube_0/pose
ros2 topic echo /cube_0/pose --once
ros2 topic echo /cube_0/confidence --once
```

### RViz debug view

```bash
rviz2
```
Add displays:
- `PointCloud2` → `/camera/depth/color/points`
- `PointCloud2` → `/cube_detections/debug`
- `MarkerArray` → `/cube_detections/markers`
- `TF`

### Detection benchmark on live stream

```bash
ros2 run cube_perception perception_benchmark --n-samples 50
```

---

## Rebuild a workspace

After changing ROS package source files:

```bash
# Perception sim workspace
cd ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select holoassist_sim --symlink-install
source install/setup.bash

# RL workspace
cd ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

---

## Notes

- The ROS sim + RL training (Python 3.10) and the clustering scripts (Python 3.12 venv) are
  **separate environments** — never mix them. PPO training and Gazebo go through ROS in the
  3.10 env; DBSCAN and Polyscope go through the 3.12 venv.
- Captured PLY files at `~/holoassist_pointclouds/` and labelled scenes at
  `~/holoassist_dataset/` are the handoff between environments.
- Scene parameters (perception sim) live in `ros2_ws/src/holoassist_sim/config/sim_params.yaml`.
- The Gazebo perception world is generated at launch from `sim_params.yaml` — never edit
  the generated SDF.
- The RL Gazebo world is `ur3e_rl_ws/src/ur3e_gazebo_sim/worlds/pick_place_world.sdf` —
  edit directly, then rebuild `ur3e_gazebo_sim`.
- `train_ppo_parallel` does not support `--help`; invoking it starts training immediately.
- Before each parallel-training run, clear stale Gazebo processes:
  `pkill -f "ign|gz|train_ppo_parallel" || true`.
