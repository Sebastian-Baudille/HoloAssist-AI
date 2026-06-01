# HoloAssist-AI — Progress

_Last updated: 2026-05-28_

Current in-flight work. For the system design, see [ARCHITECTURE.md](ARCHITECTURE.md).
For commands, see [LAUNCH.md](LAUNCH.md).

---

## Stage checklist

- [x] Section 0 — bug fixes (reward arg mismatch, shared action-scale constants, policy scaling)
- [x] Section 1 — Gazebo bridge topics (dynamic pose bridge for `/cube_0..3/pose`)
- [x] Section 2 — 13-D normalised observation space
- [x] Section 3 — target-position joint actions (+0.3 s trajectory interpolation)
- [x] Section 4 — gripper integration (7-D action space, grasped signal)
- [x] Section 5 — pick / place shaped reward + `cube_in_bin` info path
- [x] Section 6 — point cloud cube detector node (Open3D pipeline)
- [ ] Section 7 — long-run PPO training
- [ ] Section 8 — sim-to-real transfer

---

## Current section

**Section 7 — Training run.** Single-env and parallel-startup paths re-validated after
the safety slew-limit fix. Next step: long uninterrupted parallel run to collect 50k,
200k, 500k checkpoints with reward / success metrics in this doc.

---

## Last known working state

- Sections 0–6 complete and built:
  `colcon build --packages-select ur3e_rl_env ur3e_policy_controller ur3e_safety_layer` passes.
- Section 7 stabilisation in place:
  - `SafetyChecker.make_safe_target(...)` applies absolute joint-limit clamp **plus**
    per-step slew limit (`max_delta_rad = JOINT_DELTA_ACTION_SCALE_RAD = 0.24`) relative
    to current joints.
  - Env and runtime policy now pass current joint positions into safety targeting.
  - `smoke_test_joint_command` passes (joint 0 moves by +0.1 rad and joint states reflect motion).
  - Single-env PPO sanity run stable for 200+ steps post-fix; no NaN, no trajectory
    tolerance-violation logs.
  - Parallel PPO startup validated (4 workers launched, PPO initialised) then manually stopped.
- URGENT cube-pose bridge fix landed in `gazebo_pose_bridge.py`:
  - Explicit live mapping from `/world/ur3e_pick_place_world/dynamic_pose/info` TF child
    frames `cube_1..cube_4` to published `/cube_0..3/pose`.
  - Reset callback retained as fallback only.
  - Verified on `ROS_DOMAIN_ID=30`: `/cube_0/pose` reflects live cube_1 movement.

---

## Training results

| Run | Steps | `ep_rew_mean` | Success | Notes |
|---|---|---|---|---|
| Single-env (pre-slew-limit) | 200 | -106.1 | 0/10 | sanity run |
| Single-env (pre-slew-limit) | 400 | -100.2 | 0/10 | sanity run |
| Single-env (post-slew-limit) | 200 | -78.7 | 0/10 | reward trend improved with slew clamp |
| Parallel (4 envs) | — | — | — | initialisation confirmed, run manually stopped before checkpoint |
| Parallel (4 envs) | 50k | — | — | **pending** |
| Parallel (4 envs) | 200k | — | — | **pending** |
| Parallel (4 envs) | 500k | — | — | **pending** |

Logs at `ur3e_rl_ws/tb_logs/`. Open with `tensorboard --logdir ur3e_rl_ws/tb_logs`.

---

## `cube_perception` package

Live D435i DBSCAN perception with temporal tracking + occlusion hold. Package created
in a separate session.

- [x] Package created and built (`colcon build --packages-select cube_perception` passes)
- [x] TF transform verified (`base_link → camera_link` via easy_handeye2 calibration)
- [x] Node launches without error
- [x] Detector pipeline wired: workspace crop → RANSAC plane removal → DBSCAN →
      centroid + confidence
- [x] Temporal tracker with occlusion hold + confidence decay
- [x] Publishes `/cube_0..3/pose`, `/cube_0..3/confidence`, `/cube_detections/debug`,
      `/cube_detections/markers`
- [ ] 10 Hz publish-rate confirmation with live D435i stream
- [ ] Single-cube detection accuracy and occlusion tests
- [ ] RViz visual confirmation (green = confident, yellow = partial, red = low)
- [ ] All 4 cubes detected simultaneously
- [ ] Benchmark run — std deviation result

Run commands: see "Real D435i — live perception runbook" in [LAUNCH.md](LAUNCH.md).

---

## Next steps

1. **Section 7 — single-env full run.** Let the post-fix single-env training complete
   10k steps unattended:
   ```bash
   UR3E_RL_TOTAL_TIMESTEPS=10000 ros2 run ur3e_rl_env train_ppo
   ```
2. **Section 7 — full parallel run.** Long parallel run with checkpoint capture at 50k,
   200k, 500k. Record `ep_rew_mean` and success rate in the table above:
   ```bash
   UR3E_RL_TOTAL_TIMESTEPS=500000 \
   UR3E_RL_NUM_ENVS=4 \
   UR3E_RL_PPO_N_STEPS=2048 \
   UR3E_RL_PPO_BATCH_SIZE=256 \
   ros2 run ur3e_rl_env train_ppo_parallel
   ```
3. **Section 8 — sim-to-real checks.** Real hardware topics, RealSense stream, bounds
   calibration, policy deploy. Use [LAUNCH.md](LAUNCH.md) → "Real D435i" + "RL evaluation
   and deployment".
4. **Forward — Isaac Sim port.** See [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md). Independent
   track that should unlock 100×–1000× training throughput once landed.

---

## Known issues / gotchas

- `train_ppo_parallel` does not support `--help` — invoking it starts training immediately.
  Do not use it as a dry-run.
- Stale Gazebo processes break parallel launches. Clear before each run:
  `pkill -f "ign|gz|train_ppo_parallel" || true`.
- `open3d` was installed system-wide for the perception node via
  `python3 -m pip install open3d --break-system-packages`. This introduces a SciPy/NumPy
  compatibility warning at runtime, but the detector node starts.
- `https://github.com/DavidPL1/onrobot_ros2.git` was inaccessible during initial setup.
  The suggested fallback `shadow-robot/sr_ur_arm` is ROS 1 / catkin and was removed after
  verification — not compatible with this ROS 2 workspace.
- Training speed on a developer laptop is slow (~3.2 steps/s during single-env sanity run).
  Use parallel training for any serious run; the Isaac Sim port is the long-term fix.
- `cube_perception` calibration path is `~/.ros2/easy_handeye2/calibrations/holoassist_calibration.calib`.
  Missing file → launch falls back to identity transform (camera at world origin).
