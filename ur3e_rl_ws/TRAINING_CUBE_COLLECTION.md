# Training — Cube Collection

This document is the implementation plan for training the UR3e arm to reach cube
positions using PPO. It covers what exists, what needs to change, the exact code
modifications, and the training procedure.

---

## Goal

Train the arm to **reliably move its TCP to within 4 cm of a randomly placed cube**
in under 200 steps, 80% of the time — from a fixed home pose, with the cube spawning
at a random position on the table each episode.

This is Phase A of a staged training plan:

| Phase | Task | Success target | Status |
|-------|------|:---:|:---:|
| **A** | **Reach cube (TCP within 4 cm)** | **80% in ≤ 200 steps** | ✅ 99% @ 696k steps — retraining with joint constraints |
| B | Grasp cube (close gripper at contact) | TBD | ⬜ Not started |
| C | Pick and place to target zone | TBD | ⬜ Not started |

---

## Why we are restarting training

300k steps were trained on the previous setup. Result: **0% success rate**.

Two root causes:

1. **Joint positions were missing from the observation.** The agent could see where
   its TCP was but not where its 6 joints were. Since many joint configurations produce
   the same TCP position, the agent could not learn consistent motion strategies.

2. **The reward function included grasping and placing signals the agent never
   encountered.** The `+5 grasp bonus` and `+10 place bonus` were never triggered,
   so the agent had no positive gradient to follow. Combined with a non-stationary
   reward landscape (the objective changes after grasping), the policy stalled.

Starting fresh with a cleaner, focused setup is faster than continuing from the
stalled checkpoint.

---

## Changes required

### 1. `constants.py` — add 14D observation size constant

Add `OBSERVATION_SIZE_14D = 19` alongside the existing `OBSERVATION_SIZE_13D = 13`.

### 2. `ros_interface.py` — add joint positions to observation

The 6 normalised joint positions are already being read from ROS into
`state["joint_positions"]` — they just aren't included in `build_observation()`.

**New 14D observation layout:**

| Index | Content | Normalisation |
|-------|---------|:---:|
| 0–2 | TCP position (x, y, z) | workspace bounds → [-1, 1] |
| 3–5 | Cube position (x, y, z) | workspace bounds → [-1, 1] |
| 6–11 | **Joint positions (6 joints)** | **joint limits → [-1, 1]** |
| 12 | EE height | clipped [0, 1] |
| 13 | Timestep progress | [0, 1] |

Removed from old 13D: `bin_x/y/z` (no bin task), `grasped`, `gripper_state`
(no grasping task).

Joint normalisation: each joint ∈ [-2π, +2π] → normalised as
`(joint - lower) / (upper - lower) * 2 - 1`.

### 3. `reward.py` — strip down to reach-only

Remove:
- Grasp bonus (`+5.0 if grasped`)
- Transport signal (`-0.5 × dist(cube → bin) if grasped`)
- Success bonus (`+10.0 if cube_in_bin`)

Keep:
- Dense reach signal: `-0.3 × dist(ee → cube)`
- Time penalty: `-0.001 × step`
- Action smoothness: `-0.01 × ||action||²`
- Collision penalty: `-0.5 if collision`

Add:
- **Terminal reach bonus: `+5.0` when `dist(ee → cube) ≤ 0.04 m`** — gives the
  agent a clear positive signal the moment it first succeeds, before the episode ends.

### 4. `ur3e_pick_place_env.py` — fix success condition and observation/action space

- Change `success` from `cube_in_bin` to `check_success(new_state)` (EE within 4 cm
  of cube). This is what `reward.py`'s `check_success()` already tests.
- Reduce action space from 7D to **6D** (drop the gripper dimension — not needed for
  reaching, and including it adds noise to the policy gradient).
- Update `observation_space` to match the new 14D vector.

---

## Exact code changes

### `constants.py`

```python
# Add alongside existing OBSERVATION_SIZE_13D:
OBSERVATION_SIZE_14D = 19
```

### `ros_interface.py`

Replace `build_observation()` with:

```python
from ur3e_rl_env.constants import (
    OBSERVATION_SIZE_14D,
    UR3E_JOINT_LOWER_LIMITS_RAD,
    UR3E_JOINT_UPPER_LIMITS_RAD,
    WORKSPACE_HEIGHT_M,
    WORKSPACE_X_MAX, WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX, WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX, WORKSPACE_Z_MIN,
)

OBSERVATION_SIZE = OBSERVATION_SIZE_14D

_JOINT_LOWER = np.array(UR3E_JOINT_LOWER_LIMITS_RAD, dtype=np.float32)
_JOINT_UPPER = np.array(UR3E_JOINT_UPPER_LIMITS_RAD, dtype=np.float32)

def _normalize_joints(joint_positions: np.ndarray) -> np.ndarray:
    span = np.maximum(_JOINT_UPPER - _JOINT_LOWER, 1e-6)
    normalized = 2.0 * ((joint_positions - _JOINT_LOWER) / span) - 1.0
    return np.clip(normalized, -1.0, 1.0).astype(np.float32)


def build_observation(
    state: Mapping[str, Any],
    step_count: int = 0,
    max_episode_steps: int = 200,
) -> np.ndarray:
    """Builds the normalised 14D PPO observation vector."""

    ee_position     = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    object_position = np.asarray(state["object_position"],       dtype=np.float32).reshape(3)
    joint_positions = np.asarray(state["joint_positions"],       dtype=np.float32).reshape(6)

    ee_height_norm  = float(np.clip(float(ee_position[2]) / WORKSPACE_HEIGHT_M, -1.0, 1.0))
    timestep_norm   = float(np.clip(float(step_count) / max(float(max_episode_steps), 1.0), 0.0, 1.0))

    observation = np.concatenate([
        _normalize_xyz(ee_position),        # [0:3]
        _normalize_xyz(object_position),    # [3:6]
        _normalize_joints(joint_positions), # [6:12]
        [ee_height_norm, timestep_norm],    # [12:14]
    ]).astype(np.float32)

    if observation.shape != (OBSERVATION_SIZE,):
        raise ValueError(f"Expected observation shape {(OBSERVATION_SIZE,)}, got {observation.shape}")
    return np.clip(observation, -1.0, 1.0).astype(np.float32)
```

Also update `get_observation()` to pass `step_count` — the call in `ros_interface.py`
currently omits it, so timestep is always 0. Pass it through from the env.

### `reward.py`

```python
SUCCESS_DISTANCE_M   = 0.04
MIN_EE_Z_M           = 0.02
COLLISION_PENALTY    = 0.5
TIME_PENALTY         = 0.001
ACTION_PENALTY_SCALE = 0.01
REACH_BONUS          = 5.0   # terminal bonus when TCP reaches the cube


def check_success(state: Mapping[str, object]) -> bool:
    return (
        _distance(state["end_effector_position"], state["object_position"])
        <= SUCCESS_DISTANCE_M
    )


def check_failure(state: Mapping[str, object]) -> bool:
    if bool(state.get("collision_flag", False)):
        return True
    ee_pos = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
    return float(ee_pos[2]) < MIN_EE_Z_M


def compute_reward(
    state: Mapping[str, object] | Sequence[float],
    action: Sequence[float] | None = None,
    step_count: int = 0,
    info: Mapping[str, object] | None = None,
) -> float:
    info_map   = dict(info or {})
    action_vec = np.asarray(action if action is not None else np.zeros(6), dtype=np.float32).reshape(-1)

    if isinstance(state, Mapping):
        ee_pos   = np.asarray(state["end_effector_position"], dtype=np.float32).reshape(3)
        cube_pos = np.asarray(state["object_position"],       dtype=np.float32).reshape(3)
        timestep = float(step_count)
        info_map.setdefault("collision", bool(state.get("collision_flag", False)))
    else:
        obs      = np.asarray(state, dtype=np.float32).reshape(-1)
        ee_pos   = obs[0:3]
        cube_pos = obs[3:6]
        timestep = float(step_count)

    dist_to_cube = float(np.linalg.norm(ee_pos - cube_pos))

    reward  = -0.3 * dist_to_cube                                                    # dense reach
    reward -= TIME_PENALTY         * timestep                                        # time penalty
    reward -= ACTION_PENALTY_SCALE * float(np.sum(np.square(action_vec[:6])))       # smoothness
    reward -= COLLISION_PENALTY    * float(bool(info_map.get("collision", False)))   # safety

    if bool(info_map.get("reached", False)):
        reward += REACH_BONUS                                                        # terminal bonus

    return float(reward)
```

### `ur3e_pick_place_env.py`

Change action space from 7D to 6D:
```python
self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
```

Update observation space:
```python
self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
# OBSERVATION_SIZE is now 19 (imported from ros_interface)
```

Fix success condition in `step()`:
```python
# Replace:
success = cube_in_bin

# With:
from ur3e_rl_env.reward import check_success
success = check_success(new_state)

info = {
    "collision":       bool(new_state.get("collision_flag", False)),
    "reached":         success,
    "distance_to_cube": float(np.linalg.norm(
        np.asarray(new_state["end_effector_position"]) -
        np.asarray(new_state["object_position"])
    )),
}
```

Remove gripper command handling from `step()` (no longer in action space):
```python
# Remove these lines:
gripper_command = float(action_array[6])
...
if gripper_command > 0.5:
    self.ros.close_gripper()
elif gripper_command < -0.5:
    self.ros.open_gripper()
```

---

## Training procedure

Open four terminals from the repo root. All assume:
```bash
cd /home/guy/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
```

---

### Terminal 1 — Start training (20 parallel envs, ~50 fps)

Fresh run (no pretrained model):
```bash
rm -f rl_models/ppo_ur3e_pretrained.zip

CUDA_VISIBLE_DEVICES="" \
UR3E_RL_TOTAL_TIMESTEPS=1000000 \
UR3E_RL_NUM_ENVS=20 \
UR3E_RL_PPO_N_STEPS=512 \
UR3E_RL_PPO_BATCH_SIZE=512 \
python3 -m ur3e_rl_env.train_ppo_parallel
```

Confirm it prints `No pretrained model found — training from scratch.`

**Resuming from a checkpoint** (e.g. after a pause):
```bash
# Find the latest checkpoint
ls -t rl_models/checkpoints_parallel/ | head -3

# Copy it to the pretrained slot (adjust filename to match)
cp rl_models/checkpoints_parallel/ppo_ur3e_reach_object_parallel_XXXXXX_steps.zip \
   rl_models/ppo_ur3e_pretrained.zip

# Launch with remaining steps (1M minus steps already done)
CUDA_VISIBLE_DEVICES="" \
UR3E_RL_TOTAL_TIMESTEPS=826000 \
UR3E_RL_NUM_ENVS=20 \
UR3E_RL_PPO_N_STEPS=512 \
UR3E_RL_PPO_BATCH_SIZE=512 \
python3 -m ur3e_rl_env.train_ppo_parallel
```

Confirm it prints `Loading pretrained model from rl_models/ppo_ur3e_pretrained`

**Stopping training:** `Ctrl+C` — the most recent checkpoint (within 10k steps) is safe to resume from.

---

### Terminal 2 — TensorBoard

```bash
tensorboard --logdir tb_logs/
```

Open **http://localhost:6006** in a browser.

Key metrics to watch:

| Metric | What it means | Target |
|--------|--------------|:---:|
| `rollout/success_rate` | Fraction of episodes that reached the cube | ≥ 0.80 |
| `rollout/ep_len_mean` | Mean episode length (shorter = succeeding faster) | < 100 |
| `rollout/ep_rew_mean` | Mean episode reward (should rise from ~-50) | > 0 |

Each training run appears as a new **PPO_N** entry. Uncheck old runs in the left panel
to see only the current one. TensorBoard auto-refreshes every 30 s — click the circular
arrow button to force an immediate refresh.

---

### Terminal 3 — RViz live monitor

Watch one training worker in real time (worker 0 by default, or pass a number 0–19):

```bash
QT_QPA_PLATFORM=xcb bash scripts/monitor_training.sh
# or to watch worker 3:
QT_QPA_PLATFORM=xcb bash scripts/monitor_training.sh 3
```

**What you'll see:**
- Robot arm moving toward a cube each episode
- **Green axes** — TCP (end effector) position
- **Red arrows** — cube positions (teleport to new random spot on each reset)
- **Yellow arrow** — goal the arm is trying to reach
- **Fine grid** — table surface at Z = 1.07 m

The arm resets to home pose between episodes. As training progresses, motions become
smoother and more direct.

**Note:** RViz connects to `ROS_DOMAIN_ID = 30 + worker_id`. Training workers use
domain IDs 30–49 (20 envs). RViz never interferes with training.

---

### Terminal 4 — Evaluate a checkpoint

Requires a separate Gazebo sim (not the training workers). Stop training first, or use
a free terminal after training completes:

```bash
python3 -m ur3e_rl_env.evaluate_policy \
  --model rl_models/checkpoints_parallel/ppo_ur3e_reach_object_parallel_XXXXXX_steps \
  --episodes 20
```

---

## Success criteria for Phase A

The policy graduates to Phase B when it achieves:

- **≥ 80% of episodes succeed** (TCP within 4 cm of cube) **within 200 steps**
- Tested over **20 evaluation episodes** with a fresh random cube position each episode
- Must hold over **2 consecutive evaluations** (rules out lucky runs)

---

## Phase B plan (grasping — do not implement yet)

Once Phase A is passing, extend as follows:

1. **Restore gripper to action space** (7D again)
2. **Add grasping reward**: `+5.0` when gripper is closed AND TCP within 3 cm of cube,
   sustained for ≥ 5 consecutive steps (prevents exploit of hovering with closed gripper)
3. **Add joint velocities to observation** (6 more values → 25D total) — gives temporal
   context for deceleration and fine control near the cube
4. **Load Phase A checkpoint** as starting point — the arm already knows how to reach,
   now it just needs to learn when to close the gripper

---

## Files modified

| File | Change |
|------|--------|
| `ur3e_rl_env/constants.py` | Add `OBSERVATION_SIZE_14D = 19` |
| `ur3e_rl_env/ros_interface.py` | Add joint positions to `build_observation()`, drop bin/grasp entries |
| `ur3e_rl_env/reward.py` | Strip to reach-only reward + terminal `REACH_BONUS` |
| `ur3e_rl_env/envs/ur3e_pick_place_env.py` | 6D action space, 14D obs space, success = `check_success()` |
