# Training Content — for training.html and index.html RL slide

Current site only covers Isaac Sim training. Two parallel training backends exist.

---

## Two training backends

### 1. MuJoCo (John Chen) — fast iteration backend
Built in Python 3.11, Stable-Baselines3 PPO, no ROS dependency.
Purpose: iterate on reward shaping and staged policy design fast before committing to Isaac Sim.
Speed: ~1500 steps/s on 4 CPU workers vs hundreds in Gazebo — allows rapid experiments.

**Architecture: Staged sub-policies (Reach → Extend → Grasp → Transport)**
Instead of a single monolithic policy, John designed three specialised sub-policies:

| Stage | Env | Obs | Action | Done condition |
|---|---|---|---|---|
| Pan | `UR3ePanEnv` | `[cos(pan_err), sin(pan_err)]` (2D) | Δpan × 0.04 rad (1D) | `|pan_err| < 10°` |
| Extend | `UR3eExtendEnv` | `[xy_dist/0.6, z_err/0.5, lift/π, elbow/π, wrist1/π]` (5D) | `[Δlift, Δelbow, Δwrist1]` × 0.05 rad (3D) | EE XY-dist ≤ 8 cm |
| Grasp | `UR3eGraspEnv` | similar 5D | similar 3D | EE within 2 cm of cube |

Coordinator (`MuJoCoCoordinator`) chains stages via state machine: PAN → EXTEND → GRASP → TRANSPORT → RELEASE.

**Jacobian IK module**
Damped-least-squares Jacobian IK converts 3D Cartesian deltas (dx, dy, dz) to absolute joint angle targets.
Used to seed the policy with valid configurations near the goal — prevents the arm from learning to avoid
self-collision by thrashing, and dramatically speeds convergence.
Includes orientation correction to keep gripper pointing straight down for top-down grasps.
5 tests: cache building, output shape/dtype, joint limit clamping, directional movement, zero-action stability.

**Key design decisions**
- Home pose changed to all-zeros: TCP starts near cube height instead of 70 cm above table
- Orientation constraint removed from IK: arm uses full joint range
- ctrl set during reset: position actuators hold home pose during settle phase
- Renderer reuse: single `mujoco.Renderer` instance (not reallocated per `render()` call)
- Multiprocessing: `spawn` not `fork` to avoid MuJoCo memory issues

**Files**
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/kinematics.py` — IK module
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/reach_env.py` — Stage 1
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/extend_env.py` — Stage 2
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/grasp_env.py` — Stage 3
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/pan_env.py` — Pan sub-policy
- `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_pan_locked_reach.py` — combined train script
- `ur3e_rl_ws/src/ur3e_rl_env/export_onnx.py` — export trained SB3 models to ONNX for Isaac Sim

**ONNX export for Isaac Sim**
Sub-policies are exported to ONNX (opset 17, float32, deterministic) for hand-off to Sebastian's Isaac Sim deployment.
Inputs/outputs all in [-1, 1]. Actions are delta joint angles, not absolute positions.

---

### 2. Isaac Sim (Sebastian Baudille) — GPU training backend
Isaac Lab DirectRLEnv, RSL-RL PPO, 4096 GPU-parallel environments.
See simulation.html for environment setup detail.

**Observation space (12D ground truth)**
Module: `tasks/direct/reach/observations/ground_truth_12d.py`
- 6 joint positions (normalised)
- 3D end-effector position
- 3D target cube position

**Action space**
Module: `tasks/direct/reach/actions/joint_delta.py`
- 6 joint delta commands, scaled per-joint to respect limits

**Reward**
Multi-phase: dense distance to cube (reach), sparse grasp bonus, dense distance to bin (transport), sparse place bonus.
Collision, time, and joint limit penalties throughout.

**Training results (500 iterations)**
- Reward: -100 → -20 (monotonic improvement)
- Episode length: stabilises at ~110/200 steps
- Reach success: ~60–70%
- Status: partially converged — needs ~1000–1500 more iterations + entropy tuning

**Commands (Windows, from IsaacLab directory)**
```
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\train.py --task HoloAssist-Reach-Direct-v0 --num_envs 4096 --max_iterations 1500
isaaclab.bat -p scripts\reinforcement_learning\rsl_rl\play.py --task HoloAssist-Reach-Direct-v0 --num_envs 32
tensorboard --logdir IsaacLab/logs/rsl_rl/holoassist_reach_direct/
```

---

### 3. Gazebo Phase A training (Guy Smith) — early training on reach task
Before MuJoCo and Isaac Sim backends, Guy ran training directly in Gazebo.

**Phase A design (converged)**
- Obs: 14D — EE XYZ (3) + cube XYZ (3) + 6 joint positions + EE height (1) + timestep (1)
- Action: 6D joint deltas (gripper removed — reduces policy noise for reach-only task)
- Reward: -0.3×dist_to_cube + time penalty + action smoothness + collision penalty + +5.0 terminal reach bonus
- Success: TCP within 4 cm of cube
- Result: 99% success rate at 696k steps

**Joint limit tightening (to fix self-collision)**
- shoulder_lift: upper bound changed from +2π to -0.2 rad (keeps upper arm elevated)
- elbow lower: changed to -2.5 rad (prevents extreme reverse fold)
- wrist_1 upper: changed to 0 rad (prevents wrist flipping past neutral)
- Config penalty: soft ramp from 0 at 0.3 rad below limit to -2.0 at limit
- Note: checkpoint incompatible with new limits — required full retrain

---

## Index.html RL slide update

Suggested updated description for the PPO Grasping slide (03/04):

"PPO policies trained across two parallel backends: MuJoCo (CPU, SB3, ~1500 steps/s) for fast reward
shaping and staged sub-policy design; Isaac Lab (GPU, RSL-RL, 4096 envs, 50k+ steps/s) for final training.
MuJoCo sub-policies are exported to ONNX and loaded into Isaac Sim. The staged architecture —
Pan → Extend → Grasp — lets each sub-policy specialise on a narrow task rather than learning the full
sequence from scratch."

Suggested stats:
- 4096 parallel envs (Isaac Sim)
- 99% reach success (Gazebo Phase A / Guy)
- 3 sub-policies (MuJoCo / John)
