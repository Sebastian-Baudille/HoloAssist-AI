# Simulation Content — for simulation.html and index.html

Two simulation environments in the project: Gazebo (Ignition Fortress) and Isaac Sim.

---

## Gazebo Ignition Fortress (Ollie Lau + John Chen)

**Who built it:**
- Ollie Lau (ollie-lau / LazyTurtle): bootstrapped the Gazebo Classic environment with UR3e and table.
  Added the first PPO training scripts and ur3e_safety_layer package (collision detection, joint limits).
- John Chen: ported to Ignition Fortress, added Moveit2 collision avoidance, cube randomization,
  parallel IGN_PARTITION isolation for multi-worker training, RG2 gripper integration, OnRobot ROS 2 packages.
- After handoff: Ollie owns ongoing Gazebo + Moveit2 work.
- Sebastian: updated RViz/Gazebo sim files for dataset collection.
- Guy: tightened joint limits and added config penalty during Phase A training.

**What it does:**
- Simulates the UR3e arm + OnRobot RG2 gripper on a table with 2–4 coloured cubes
- Provides `/camera/points` point cloud for perception pipeline
- Cube positions randomized each episode (continuous uniform X/Y range via env vars)
- Safety layer enforces joint limits (max_delta_rad = 0.24 rad/step) and collision checking
- MoveIt2 collision checker runs as a separate node, one per parallel training worker
- Workers isolated via IGN_PARTITION environment variable

**Key packages:**
- `ur3e_rl_ws/src/ur3e_gazebo_sim/` — world, launch, URDF
- `ros2_ws/src/holoassist_sim/` — camera, point cloud capture, templates
- `ros2_ws/src/onrobot_driver/` — OnRobot RG2 Modbus hardware interface
- `ros2_ws/src/onrobot_description/` — RG2 URDF and meshes
- `ros2_ws/src/ur_onrobot/` — combined UR+OnRobot description + MoveIt config

---

## MuJoCo (John Chen) — fast training environment

See content-training.md for full detail. Summary:

- MJCF scene: floor, table at (-0.031, 0, 0.29), 4 free-body cubes, 2 bin sites, 6 actuators
- UR3e URDF expanded with base_z=1.10, base_yaw=π (matches Gazebo launch)
- RG2 gripper: all-fixed joints (no physics actuation — virtual gripper_closed bool)
- Table top z=1.07 m (matches TABLE_TOP_Z constant)
- Obs/action spaces identical to Gazebo env for weight transfer compatibility

**Files:**
- `ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco/scene.xml`
- `ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco/robot.xml`
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/ur3e_mujoco_env.py`

---

## Isaac Sim 5.1 (Sebastian Baudille)

**Environment:**
- NVIDIA Isaac Sim 5.1 + Isaac Lab DirectRLEnv API
- RTX 4070 Ti SUPER (16 GB, Ada Lovelace) — single GPU
- 4096 environments in one process
- 50k+ steps/s throughput

**Robot asset (USD):**
Layered USD structure from URDF→USD conversion of `ur_onrobot_prepared.urdf`:
- `ur_onrobot_prepared_base.usd` — geometry and meshes
- `ur_onrobot_prepared_physics.usd` — articulation physics
- `ur_onrobot_prepared_robot.usd` — robot prim
- `ur_onrobot_prepared_sensor.usd` — sensor definitions
- `ur_onrobot_prepared.usd` — top-level composition

**Isaac Lab extension: `holoassist_tasks`**
Located at `isaac_rl/holoassist_tasks/`
- Task: `HoloAssist-Reach-Direct-v0`
- Modular strategy: separate observation, action, reward modules
- RSL-RL PPO config: `tasks/direct/reach/agents/rsl_rl_ppo_cfg.py`
- Built-in domain randomisation via Isaac Lab for sim-to-real transfer

**Simulation slide (04/04) — suggested description update:**
"Isaac Sim 5.1 provides GPU-vectorised simulation for final policy training. The UR3e + RG2 is modelled
as a layered USD asset converted from URDF. Isaac Lab's DirectRLEnv API runs 4096 environments in parallel
on a single RTX 4070 Ti SUPER — 50k+ steps/s. Sub-policies developed in MuJoCo are loaded via ONNX for
further fine-tuning and deployment."

---

## simulation.html — additional sections to add

### Robot Asset Pipeline
```
ur_onrobot.urdf
    → Isaac Sim URDF importer
    → ur_onrobot_prepared.urdf (physics preparation)
    → ur_onrobot_prepared_base.usd + physics.usd + robot.usd + sensor.usd
    → ur_onrobot_prepared.usd (composed asset)
```

### Environment Reset
Each episode: robot reset to home pose, cube positions randomized within workspace bounds,
domain randomisation applied (mass, friction, lighting — Isaac Lab built-in).

### Deployment
Trained RSL-RL checkpoint → loaded for play/evaluation.
MuJoCo sub-policies → ONNX export → loaded in Isaac Sim for chained execution.
