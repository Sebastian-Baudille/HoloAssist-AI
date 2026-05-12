# UR3e RL Training Notes
_Last updated: 2026-05-12_

---

## What We Changed Today

### 1. Robot speed — 75% of hardware max

**Problem:** Action bounds were `±0.03 rad/step` with `control_dt=0.1s` → max speed 0.3 rad/s (~10% of UR3e hardware limit). The random initial policy had almost no chance of stumbling into the 4 cm success radius.

**Fix:** Increased to `±0.24 rad/step` → 2.4 rad/s (~75% of UR3e's π rad/s limit).

Two files must always match:

| File | Parameter | Old | New |
|---|---|---|---|
| `ur3e_rl_env/envs/ur3e_pick_place_env.py` | action space `low`/`high` | `±0.03` | `±0.24` |
| `ur3e_safety_layer/safety_checker.py` | `max_delta_rad` default | `0.05` | `0.24` |

The `SafetyChecker` sits downstream of the action space clip — if you change one, change both or the safety clamp will silently eat the extra range.

---

### 2. MoveIt launch fixes

**Problem A — wrong workspace:** `ur_onrobot_moveit_config` is built in `ros2_ws` with `--symlink-install`. The `ur3e_rl_ws/install` copy only has a compiled `.pyc` — `ros2 launch` can't use it. **Always source `ros2_ws` for the MoveIt terminal.**

**Problem B — base yaw:** `base_yaw_rad` defaulted to `0.0` but the robot is mounted at 180° (π rad) on the trolley. Fixed the default in:
```
ros2_ws/src/ur_onrobot/ur_onrobot_moveit_config/launch/ur_onrobot_moveit.launch.py
```
`default_value="0.0"` → `default_value="3.14159"`. Stale `.pyc` files deleted. No rebuild needed (symlink install).

**Problem C — sim time:** `use_sim_time:=true` for MoveIt stalls the move_group until a `/clock` topic appears. For collision checking only, `use_sim_time:=false` is correct and avoids the ordering dependency.

---

### 3. Workflow — don't manually launch Gazebo when training

`train_ppo_parallel` spawns its own Gazebo process for each worker. If you also run `ur3e_pick_place_world.launch.py` manually, you get two conflicting Gazebo instances on the same `ROS_DOMAIN_ID`. The manual Gazebo launch is only for visual debugging, never alongside training.

---

## Single-Run vs Parallel Training

### Single-run (`ros2 run ur3e_rl_env train_ppo`)

- One Gazebo, one environment, one PPO learner.
- Uses whatever Gazebo is already running (does not spawn its own).
- Blocking: data collection and policy update happen in the same process sequentially.
- Useful for debugging — easy to see what's happening.
- Slow: with `control_dt=0.1s` and `max_episode_steps=200`, worst case is 20 s/episode of real time.

### Parallel-run (`ros2 run ur3e_rl_env train_ppo_parallel`)

- `N` independent workers via `SubprocVecEnv` (separate OS processes).
- **Each worker spawns its own Gazebo** headlessly on a unique `ROS_DOMAIN_ID` and `GAZEBO_MASTER_URI`.
- Worker isolation:

  | Worker | `ROS_DOMAIN_ID` | Gazebo port |
  |--------|----------------|-------------|
  | 0 | 30 | 11400 |
  | 1 | 31 | 11401 |
  | 2 | 32 | 11402 |
  | 3 | 33 | 11403 |

- PPO collects `N × n_steps` transitions per update instead of `n_steps` — effectively `N×` faster.
- With `NUM_ENVS=4` on an i7-13700H (20 logical CPUs), training drops from ~6 hours → ~30 minutes.
- **4 envs is the practical ceiling** for this CPU — each Gazebo uses 3–5 threads; going higher causes the instances to compete and slow each other down.

### Key env vars for parallel training

```bash
UR3E_RL_NUM_ENVS=4                    # number of parallel workers (default 4)
UR3E_RL_BASE_ROS_DOMAIN_ID=30         # worker 0 domain; others increment from here
UR3E_RL_TOTAL_TIMESTEPS=200000        # total PPO timesteps
UR3E_RL_CONTROL_DT=0.1               # seconds per action step
UR3E_RL_MAX_EPISODE_STEPS=200         # steps before forced reset
UR3E_RL_PPO_N_STEPS=512              # rollout buffer size per env
UR3E_RL_PPO_BATCH_SIZE=256           # minibatch size for SGD updates
UR3E_RL_TORCH_THREADS=2              # keep low — bottleneck is sim, not torch
```

---

## MoveIt Collision Checking With Parallel Training

MoveIt's `/check_state_validity` service is domain-scoped. Worker N only talks to MoveIt on its own `ROS_DOMAIN_ID = 30 + N`.

**If using MoveIt collision checking:** run one MoveIt terminal per worker domain.

```bash
# Repeat for domains 30, 31, 32, 33 in separate terminals
export ROS_DOMAIN_ID=<30|31|32|33>
cd ~/git/HoloAssist-AI/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
  ur_type:=ur3e onrobot_type:=rg2 \
  use_sim_time:=false launch_rviz:=false launch_servo:=false
```

**If skipping MoveIt:** the basic `SafetyChecker` (joint limits + TCP height ≥ 0.02 m) still runs unconditionally. Sufficient for the reach task. Omit `UR3E_RL_USE_MOVEIT_COLLISION_CHECKER=1` and the MoveIt terminals.

With `UR3E_RL_MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE=1`: if a worker can't reach MoveIt, it treats every state as a collision and all episodes terminate immediately — training breaks silently. Only set this if you have a MoveIt instance on every worker domain.

---

## Correct Training Commands (Today's State)

### Option A — No MoveIt (fast, 1 terminal)

```bash
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=30
pkill -f "ign|gz|train_ppo_parallel" || true
UR3E_RL_NUM_ENVS=4 ros2 run ur3e_rl_env train_ppo_parallel
```

### Option B — With MoveIt collision checking (5 terminals)

**Terminals 1–4** (one per domain, source `ros2_ws`):
```bash
export ROS_DOMAIN_ID=<30|31|32|33>
cd ~/git/HoloAssist-AI/ros2_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
ros2 launch ur_onrobot_moveit_config ur_onrobot_moveit.launch.py \
  ur_type:=ur3e onrobot_type:=rg2 \
  use_sim_time:=false launch_rviz:=false launch_servo:=false
```

**Terminal 5** (wait for all move_group nodes to finish initialising first):
```bash
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
export ROS_DOMAIN_ID=30
pkill -f "ign|gz|train_ppo_parallel" || true
UR3E_RL_NUM_ENVS=4 \
UR3E_RL_USE_MOVEIT_COLLISION_CHECKER=1 \
UR3E_RL_MOVEIT_GROUP_NAME=ur_onrobot_manipulator \
UR3E_RL_MOVEIT_FAIL_CLOSED_WHEN_UNAVAILABLE=1 \
ros2 run ur3e_rl_env train_ppo_parallel
```

---

## Monitoring Training

TensorBoard is the right way to watch parallel training — opens all 4 workers aggregated:

```bash
tensorboard --logdir ~/git/HoloAssist-AI/ur3e_rl_ws/tb_logs
# open http://localhost:6006
```

Gazebo GUIs can be opened per-worker by changing `gui:=false` → `gui:=true` in `train_ppo_parallel.py:_start_gazebo()`, but each GUI adds CPU overhead and will slow training.

---

## Speed Reference

| Parameter | Value | Notes |
|---|---|---|
| `control_dt` | 0.1 s | time per action step |
| Action bounds | ±0.24 rad/step | 75% of UR3e π rad/s limit |
| Max speed | 2.4 rad/s | = 0.24 / 0.1 |
| Max episode wall time | 20 s | 200 steps × 0.1 s |
| Success threshold | 4 cm | EE to cube distance |
| Failure: EE too low | z < 0.02 m | table crash prevention |
| Failure: collision | `collision_flag=True` | from MoveIt or always-False default |
