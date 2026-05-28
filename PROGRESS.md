# HoloAssist-AI Progress

## Completed
- [x] Bug fixes (Section 0)
- [x] Gazebo bridge topics (Section 1)
- [x] 13D observation space (Section 2)
- [x] Target-position actions (Section 3)
- [x] Gripper integration (Section 4)
- [x] Pick-place reward (Section 5)
- [x] Point cloud pipeline (Section 6)
- [ ] Training run (Section 7)
- [ ] Sim-to-real transfer (Section 8)

## Current section
Section 7 — Training run

## Last known working state
- Sections 0–6 are complete and built (`colcon build --packages-select ur3e_rl_env ur3e_policy_controller ur3e_safety_layer` passed).
- Section 7 in progress:
- Added per-step target slew limiting in `SafetyChecker.make_safe_target(...)` using shared `JOINT_DELTA_ACTION_SCALE_RAD=0.24` while keeping absolute target-position actions.
- Updated env/policy call sites to pass current joint positions into safety clamping before sending trajectories.
- `smoke_test_joint_command` passes (joint 0 moves by +0.1 rad and joint states reflect motion).
- Single-env PPO sanity run reaches 200+ steps post-fix with no NaN/crash and no trajectory tolerance-violation logs.
- Parallel PPO startup (`train_ppo_parallel`) was validated: 4 workers launch, PPO initializes, then run was manually stopped.
- URGENT cube-pose bridge fix applied in `gazebo_pose_bridge.py`:
- Added explicit live mapping from `/world/ur3e_pick_place_world/dynamic_pose/info` TF child frames `cube_1..cube_4` to published `/cube_0..3/pose`.
- Reset callback retained as fallback only (`/gazebo_pose_bridge/cube_reset_pose` updates `cube_0` fallback pose).
- Rebuilt: `colcon build --packages-select ur3e_gazebo_sim --symlink-install`.
- Verification (ROS_DOMAIN_ID=30): `/cube_0/pose` position values changed across messages when cube_1 was moved with Gazebo `set_pose` (observed values included baseline `0.009,-0.4049,1.109999...` and moved `0.2,-0.35,1.109999...`).

## Known issues
- `PROGRESS.md` did not exist previously; created in this session.
- `https://github.com/DavidPL1/onrobot_ros2.git` could not be cloned here (GitHub auth prompt / inaccessible). Suggested fallback `shadow-robot/sr_ur_arm` is ROS1/catkin-based and not compatible with this ROS 2 workspace, so it was removed after validation.
- `pointcloud_cube_detector` uses `open3d`; installed with `python3 -m pip install open3d --break-system-packages`. This introduced a SciPy warning about NumPy version compatibility in this environment, but node startup works.
- `train_ppo_parallel` does not support `--help`; invoking it starts training immediately.
- Parallel Gazebo launches can fail if stale `ros2 launch ur3e_gazebo_sim ...` / `gz sim` processes are left running. Clean old processes before each new parallel run.

## Training results
- Single-env sanity run (pre-slew-limit): 400 steps reached without NaN/crash before manual stop.
- Snapshot: 200 steps reward ~ -106.1, 400 steps reward ~ -100.2 (success 0/10).
- Single-env sanity run (post-slew-limit): callback printed at 100 and 200 steps with stable execution.
- Snapshot: 200 steps reward ~ -78.7 (success 0/10).
- Parallel run: initialization confirmed (`Using cpu device`, `Logging to ./tb_logs/PPO_16`, `Training PPO with 4 Gazebo envs...`) then manually stopped before a long checkpoint.

## cube_perception package
- [x] Package created and built
- [x] TF transform verified (base_link → camera_link)
- [x] Node launches without error
- [ ] Topics publishing at ~10Hz
- [ ] Single cube detected correctly
- [ ] Occlusion test passed (pose held during hand occlusion)
- [ ] RViz markers showing (green=confident, yellow=partial, red=low)
- [ ] All 4 cubes detected simultaneously
- [ ] Benchmark run — std deviation results: ___mm
