# Isaac Sim RL Development Log — UR3e Grasping Tasks

**Project:** HoloAssist-AI (41118 AI in Robotics)  
**Author:** Seb (Sebastian-Baudille)  
**Period:** May–June 2026  
**Platform:** Isaac Sim 5.1 / Isaac Lab 2.x, RSL-RL PPO, Windows (local GPU)  
**Logs:** `C:\Users\sebas\Github\IsaacLab\logs\rsl_rl\`

---

## Overview

The Isaac Sim stack is Seb's component of the HoloAssist-AI project. It trains RL policies for a UR3e + RG2 gripper using **Isaac Lab** (Omniverse-based robotics simulation framework) with **RSL-RL PPO**. Isaac Sim is GPU-accelerated and can run thousands of parallel environments simultaneously — the primary advantage over MuJoCo for large-scale RL.

The stack progressed through two major task phases:
1. **Reach task** — drive the end-effector to a randomised target position above the table
2. **Grab task** — reach to a cube, close the gripper, and lift it > 5 cm off the table

Both tasks use the same UR3e + RG2 URDF and a shared DirectRLEnv architecture based on Isaac Lab's framework.

---

## Repository Structure

```
isaac_rl/
  assets/
    urdf/
      ur_onrobot.urdf           UR3e + RG2 URDF (base)
      ur_onrobot_prepared.urdf  Processed version (fixed joints, etc.)
    meshes/
      onrobot_description/rg2/{collision,visual}/*.stl
      ur_description/ur3e/{collision,visual}/*.{stl,dae}
  holoassist_tasks/
    source/holoassist_tasks/holoassist_tasks/
      common/
        gripper_coupling.py     Gripper linkage kinematics + mimic control
        kinematics.py           UR3e FK/IK (scipy-based)
      tasks/direct/
        reach/                  Reach task (env, cfg, obs, rewards, actions)
        grab/                   Grab task (env, cfg, obs, rewards, actions)
    scripts/
      rsl_rl/
        train.py                RSL-RL training entrypoint
        play.py                 Policy playback
      list_envs.py, random_agent.py, zero_agent.py   Diagnostics
  TRAINING_RUNS.md              Catalogue of all training runs
  LAUNCH.md                     Launch commands
```

---

## Architecture — Strategy Pattern

Both the reach and grab tasks follow a **strategy-module architecture**: the env class (`reach_env.py`, `grab_env.py`) is a thin shell that handles lifecycle (init, scene setup, reset, dones), and all observation, reward, and action logic lives in sibling subpackages:

```
tasks/direct/reach/
  observations/ground_truth_12d.py   ← observation strategy
  rewards/dense_reach_v1.py          ← reward strategy (active)
  actions/joint_delta.py             ← action strategy
  reach_env.py                       ← env shell (imports strategies at top)
```

To A/B test a different reward design: change one import line, no env subclassing needed. This made it very fast to iterate through reward versions without copy-pasting the env boilerplate.

Multiple reward versions are kept simultaneously (`dense_reach_v0.py` through `dense_reach_v3_ik.py`). For the grab task, all 5 reward versions (`dense_grab.py` through `dense_grab_v5.py`) are registered in a dispatch dict so they can be selected by config string rather than requiring separate env registrations.

---

## URDF Pipeline

Isaac Sim loads the robot from a URDF. The UR3e + RG2 URDF required preprocessing before Isaac Sim could load it:

1. **`inspect_urdf.py`** — validates joint types, link masses, mesh paths
2. **`prepare_urdf.py`** — processes the raw URDF into `ur_onrobot_prepared.urdf`:
   - Converts fixed joints to fused links where needed
   - Sets up the RG2 gripper's mimic joints in a form Isaac Sim can handle

### Gripper: PhysxMimicJointAPI Problem

The RG2 gripper is a 4-bar parallel linkage. In URDF the follower joints are defined as `<mimic>` joints tracking the master. Isaac Sim 5.1's **PhysxMimicJointAPI is metadata-only** — it does not enforce mimic behaviour in the physics solver. This was proven empirically during development.

**Solution:** Enforce gripper kinematics via **stiff position drives** rather than physics mimic. All 6 linkage joints (master + 5 followers) are given commanded positions computed with the correct gear signs (`common/gripper_coupling.py`), and the drive stiffness is set high enough (`linkage_drive_stiffness`, typically 1000–5000) that they track the command rigidly under contact.

```python
# In grab_env.__init__:
stiff = torch.full((num_envs, 6), cfg.linkage_drive_stiffness, device=device)
robot.write_joint_stiffness_to_sim(stiff, joint_ids=linkage_ids_tensor)
```

The `finger_width` joint (a virtual joint measuring jaw opening) is used as the observable gripper state — its value is what reward terms and termination conditions check to determine if the gripper is actually closed.

### EE Proxy

Isaac Sim's "merge fixed joints" pass during USD conversion fuses frame-only links with no inertia into their parent. `gripper_tcp` (the true TCP frame) gets merged into `onrobot_base_link` and is not queryable as a body. The closest available body that approximates the grasp point is `left_inner_finger`.

For the grab task, gripper centre is computed as the **midpoint of both inner fingers**:
```python
gripper_center = (left_finger_pos + right_finger_pos) * 0.5
```

This is slightly asymmetric vs the geometric centre (few mm) but well within the success tolerances used.

---

## Task 1: Reach Task

### Goal

Drive the end-effector (`left_inner_finger` body) to a randomised XYZ target position above the table. No gripper control — arm only. Success when EE is within `success_tolerance_m` of the target.

### Observation Space (12D — `ground_truth_12d.py`)

```
[ee_x, ee_y, ee_z,         — end-effector world position (3)
 target_x, target_y, target_z,  — target world position (3)
 j0, j1, j2, j3, j4, j5]   — arm joint positions (6)
```

All values in SI units (metres, radians), not normalised. The 12D obs is ground truth — no perception noise.

### Action Space (6D — `joint_delta.py`)

Joint-delta control:
```
action[i] ∈ [-1, 1] → joint_i_delta = action[i] × cfg.action_scale_rad
```

Applied as position targets: `joint_pos_target += delta`. This is a clamped integrating controller — the arm moves in small increments each step and the physics engine resolves contacts.

### Reward Versions

| Version | Key Change | Result |
|---|---|---|
| `dense_reach_v0` | Raw distance reward | Early — converged slowly |
| `dense_reach_v1` | Action-rate penalty (smooth motion) | **Operational baseline** — working policy |
| `dense_reach_v2` | Jerk penalty (second-difference actions) | Robot folded into itself — over-penalised |
| `dense_reach_v3_ik` | IK-guided reward (tracking IK reference joints) | Also folded — IK pull too strong |

v1 is the keeper. The addition of an action-rate penalty (`‖a_t - a_{t-1}‖`) to the dense distance reward was enough to prevent the jerky, self-collision-prone motion of v0.

### IK Grid for v3

The reach env precomputes an IK grid at init time for reward version v3:
- Walks a 20×20 grid over the target spawn zone
- Runs `scipy` IK at each grid point via `kinematics.compute_ik_reference()`
- Caches solutions as device tensors: `_ik_grid_xy`, `_ik_grid_joints`
- At each reset, the nearest-neighbour IK solution is looked up and stored as `_ik_reference`

This costs ~2 seconds at startup but is amortised over the entire training run. The 1–2 cm nearest-neighbour rounding error is within the noise of the IK tracking reward term and doesn't affect final reach accuracy.

### Episode Termination

- **Success:** `dist(ee, target) ≤ success_tolerance_m`
- **Failure:** `ee_z < robot_base_height_m - min_ee_clearance_below_base_m` (table-crash guard)
- **Truncation:** episode length exceeded

### Results

`dense_reach_v1` is the operational baseline for the reach task, used as a starting point for grab task development.

---

## Task 2: Grab Task

### Goal

Reach a cube on the table, close the gripper around it, and lift it at least 5 cm above the table surface. Success = `cube_z > table_top + 5 cm`.

### Observation Space (16D — `ground_truth_16d.py`)

```
[ee_x, ee_y, ee_z,             — end-effector (gripper centre) world position (3)
 cube_x, cube_y, cube_z,       — cube world position (3)
 j0–j5,                        — arm joint positions (6)
 gripper_close_signal,         — action[6] from previous step (1)
 finger_width]                 — actual jaw width from physics (1)
                                 -- then 2 additional terms depending on version
```

### Action Space (7D — `joint_delta_gripper.py`)

```
action[0:6] ∈ [-1, 1]  → arm joint deltas × cfg.action_scale_rad
action[6]   ∈ [-1, 1]  → gripper signal (+1 = open, -1 = close)
```

The gripper signal maps to a commanded position for all 6 linkage joints simultaneously, with per-joint gear signs applied by `gripper_coupling.py`.

### Reward Design — The Hard Part

The grab task reward was iterated through **5 versions**. Each version exposed a different degenerate behaviour. This is the most detailed part of the Isaac Sim development log.

#### Version 0 — Working Baseline (`dense_grab.py`)

6 terms, all positive bonuses except term 1:

```
1. reach_distance       always-on: -reach_scale × dist(gripper, cube)
2. xy_alignment         proximity-gated: bonus when EE centered above cube (XY)
3. orient_alignment     proximity-gated: bonus when gripper pointing straight down
4. grasp_activation     bonus when EE close + gripper actually closing
5. lift_bonus           gated on grasped heuristic: scales with cube lift height
6. success_bonus        terminal: large bonus when cube_z > table_top + 5 cm
```

**Proximity gate** for terms 2 and 3: only active when `|EE_z - cube_z| < alignment_z_gate`. This prevents the policy from earning alignment rewards while the arm is nowhere near the cube.

**Grasped heuristic** for term 5:
```python
grasped = (cube_z > table_top + 0.005) AND (finger_width < grasped_width) AND (dist < grasped_dist)
```
This is a geometric check for "cube is slightly off the table, gripper is mostly closed, gripper is near the cube" — derived from quantities available without any sim-only ground truth.

**Result:** Works. Policy reaches 97% success rate (mean reward 196/200). Behaviour is a "side-sprawl approach" — the arm approaches the cube from the side rather than above. This causes visible self-collision (arm folds through itself). Success threshold was 5 cm, which was achievable with random gripper closures as long as the arm was near the cube.

**Limitation:** Self-collision is physically unrealistic.

#### Version 1 — Hover Trap (`dense_grab_v1.py`)

Added term: `approach_height` — bonus for the EE being at a "correct approach height" directly above the cube. Also made orient_alignment ungated (active at all times, not just when near the cube).

**Intent:** Encourage the arm to approach from above rather than the side, fixing the self-collision problem.

**Result:** The arm learned perfect overhead posture — correct wrist orientation, correct height, EE directly above cube — then **hovered indefinitely** without closing the gripper. Mean episode length: 200 (always timeout). Mean reward: 316 (misleadingly high from accumulated per-step alignment rewards).

**Diagnosis:** `rew_scale_orient_align = 1.5` (ungated) made hovering at the ideal height indefinitely more rewarding than the descent→grasp→lift trajectory. The policy found a local optimum where it earned ~1.5/step for maintaining orientation without ever attempting to grasp.

**Design lesson:** When orientation or alignment rewards are not gated on proximity, the policy will find a pose that maximises them continuously rather than moving toward grasping.

#### Version 2 — Lateral Misalignment (`dense_grab_v2.py`)

Added term: `time_penalty` (-1/step) to break the hover. Also rebalanced: `orient_scale 1.5→0.3`, `lift_scale 80→100`, `grasp_act_scale 1→5`, `success_bonus 200→300`.

**Intent:** Make hovering costly via the time penalty, and increase grasp-activation signal to pull toward closing.

**Result:** Two new exploits emerged:
1. **Hold-below-threshold:** policy closed gripper beside the cube (not around it), raised cube just below 10 cm (the increased success threshold), and milked continuous `lift_bonus` without triggering success termination
2. **Sloppy grasp:** arm approached with incorrect alignment, closed gripper anywhere near the cube to earn `grasp_activation` reward

Entropy climbed rather than decreasing — no coherent strategy converged.

**Diagnosis:** After rebalancing, `sum(per-step rewards) × episode_length > success_bonus`. This violated the key invariant: the terminal success bonus must be larger than the total per-step reward that can be accumulated by staying in the environment without succeeding. When this invariant breaks, the policy prefers to stay alive and milk per-step rewards over terminating with a success bonus.

**Design rule established:** `sum(per-step reward) × max_episode_length MUST be < success_bonus`.

#### Version 3 — Finger Drag (`dense_grab_v3.py`)

Strategic retreat to v0's 6-term design. Only change: added a small `elbow_up` posture nudge (max 30 per episode, not per step) to encourage the arm to keep its elbow elevated (preventing table strikes). Also added PhysX self-collision (disabled in v0), raised success threshold to 10 cm, and rebalanced to maintain the sum < success invariant.

**Result:** Approach posture is correct — arm reaches overhead, wrist-down, gripper centered. Fingers descend to the table surface with the cube between them. **Gripper does not close.** No successful lifts.

**Diagnosis:** "Finger drag" trap. Once the fingers touch the table, friction prevents the inward closing motion. The policy earns ~190/episode from `xy_align + orient_align + elbow_up` while stuck at the cube, but the grasp reward gradient is too weak to break out of this configuration. v0 escaped this by accident — its 5 cm success threshold meant random gripper closures sometimes lifted the cube high enough to fire success; v3's 10 cm threshold removed that lucky escape path.

**Design lesson:** A well-shaped reward can still produce stuck behaviour if the policy reaches a local maximum that physically prevents further progress. An explicit reward for **lifting the gripper above the table** during the closing transition was identified as the fix.

#### Version 4 — Safe Hover (`dense_grab_v4.py`)

Added `anti_drag` term: -1/step when `finger_z ≤ table_top + 0.20`. This was meant to push the fingers off the table surface during closing.

**Result:** The policy learned to hover above the table surface at exactly 0.21 m (just above the anti-drag threshold) without ever descending. Mean reward: ~13 — the worst result of all versions. The anti-drag pushed the policy away from the table but provided no gradient toward grasping.

#### Version 5 — Conservative Return (`dense_grab_v5.py`)

Status: Ready to train (not yet run at time of writing).

Design: Return to v0's exact reward scales (lift 50, success 200, orient 0.3 proximity-gated, grasp_act 1.0, 5 cm threshold — the configuration proven to find grasping). Add only:
- PhysX self-collision (fixes v0's arm-through-arm flaw)
- Tiny `elbow_up` posture nudge (max 30/episode)

No scale rebalancing, no new exploration constraints. Drops the 10 cm aspiration — 5 cm is the threshold empirically proven to be achievable with the v0 reward structure.

### Episode Termination (Grab)

- **Success:** `cube_z > table_top + success_lift_height` (5 cm for v0/v5, 10 cm for v1/v2/v3/v4)
- **Failure:** `ee_z < robot_base_height - min_clearance` (table-crash guard, same as reach)
- **Truncation:** episode length exceeded

---

## Training Runs Summary

| Run | Reward | Self-Collision | Threshold | Mean Reward | Behaviour |
|---|---|---|---|---|---|
| `grab-r0-run1` | dense_grab (v0) | OFF | 5 cm | 196/200 (97%) | **Working** — side-sprawl, self-collision visible |
| `grab-r1-run1` | dense_grab_v1 | ON | 10 cm | 316 (misleading) | Hover trap — never closes gripper |
| `grab-r2-run1` | dense_grab_v2 | ON | 10 cm | ~750 (misleading) | Exploit — hold below threshold, sloppy grab |
| `grab-r3-run1` | dense_grab_v3 | ON | 10 cm | ~190 | Finger drag — correct approach, no close |
| `grab-r4-run1` | dense_grab_v4 | ON | 10 cm | ~13 | Safe hover — worst result |
| `grab-r5-run1` | dense_grab_v5 | ON | 5 cm | TBD | Ready to train |

All runs: 4096 envs, 2500 max iterations, RSL-RL PPO. Run on Windows local GPU.

---

## Infrastructure and Tooling

### Training Commands

```powershell
# Full training (grab v5)
.\isaaclab.bat -p ...train.py --task Template-Holoassist-Grab-Direct-v5 --num_envs 4096 --max_iterations 2500 --headless

# Pretest (wiring check, 4 envs × 5 iters)
.\isaaclab.bat -p ...train.py --task Template-Holoassist-Grab-Direct-v5 --num_envs 4 --max_iterations 5 --headless --experiment_name grab-r5-pretest

# Playback
.\isaaclab.bat -p ...play.py --task Template-Holoassist-Grab-Direct-v0 --checkpoint "C:\...\model_3000.pt"
```

### Naming Convention

```
<task>-r<reward_version>-run<N>   full training runs
<task>-r<reward_version>-pretest  wiring-check runs
```

The run counter (`runN`) is set manually. If `--experiment_name` is omitted, the cfg default (`run1`) is used and successive runs land as sibling timestamps inside the same folder — potentially confusing in TensorBoard.

### TensorBoard

```powershell
cd C:\Users\sebas\Github\IsaacLab
.\isaaclab.bat -p -m tensorboard.main --logdir "C:\...\logs\rsl_rl" --port 6006
```

All runs visible at http://localhost:6006 — tick individual runs in the sidebar to compare reward curves.

---

## What We Would Change

### 1. Self-Collision From the Start

v0's working policy has visible arm self-collision (arm folds through itself). Enabling PhysX self-collision from the very first grab run and designing the reward to be compatible with it would have saved several failed reward iterations. The lesson: don't use "self-collision OFF" as a shortcut to get a working policy — it teaches physically impossible behaviours.

### 2. Invariant Check Before Training

The "sum of per-step rewards × episode_length < success_bonus" invariant was discovered empirically (from v2's failure). This invariant should be verified analytically before any training run starts. A simple calculation:

```
max_per_step_reward × max_steps vs success_bonus
```

If the left side ≥ right side, the policy will prefer to milk per-step rewards. v1 and v2 both violated this.

### 3. Curriculum for Closing

Versions v3 and v4 both show the policy can reach the correct position but cannot learn to close the gripper. A curriculum that rewards the gripper-close action specifically when the arm is correctly positioned (separate from the lift reward) might provide the gradient needed to break out of the finger-drag and safe-hover traps.

### 4. ONNX Export for Cross-Framework Deployment

The MuJoCo teammate (John) trained with Stable Baselines3; the Isaac Sim policies use RSL-RL checkpoints (`.pt` files). For the two pipelines to interoperate, ONNX export needs to be set up on both sides. RSL-RL supports ONNX export via `torch.onnx.export`. This should be validated early — not left for after training converges.

### 5. Log Storage

Training logs are stored locally on Seb's Windows machine at `C:\Users\sebas\Github\IsaacLab\logs\rsl_rl\`. These are not committed to the repo. A shared network location or cloud storage would preserve training history in case of machine failure.

---

## Discussion

### Why Isaac Sim?

Isaac Sim (Omniverse-based) enables GPU-parallelised physics at 4096+ simultaneous environments, which is the scale needed for PPO to converge in reasonable wall-clock time on hard manipulation tasks. MuJoCo (John's component) runs 16 envs efficiently on CPU; Isaac Sim at 4096 envs on a good GPU achieves similar sample throughput per hour at 256× the episode count per batch.

The practical cost is a more complex ecosystem: URDF preprocessing, Isaac Lab's DirectRLEnv framework, the PhysxMimicJointAPI issue, and Windows-specific tooling. The Isaac Lab framework abstracts most of this once the initial setup is working.

### Why RSL-RL?

RSL-RL (Robotic Systems Lab RL) is the PPO implementation bundled with Isaac Lab and tested against its parallel env API. Using SB3 or a custom PPO would require writing a custom vec-env wrapper. RSL-RL's `train.py` / `play.py` scripts handle logging, checkpointing, and TensorBoard output with minimal configuration.

### Comparison With MuJoCo Training

Both components train a UR3e + RG2 arm to grasp a cube, but with different decompositions:

| | MuJoCo (John) | Isaac Sim (Seb) |
|---|---|---|
| Approach | Sub-policy chain (pan + reach) | Monolithic grab (reach + grasp) |
| Algorithm | SB3 PPO | RSL-RL PPO |
| Parallel envs | 16 (CPU) | 4096 (GPU) |
| Gripper control | Open-loop script (watch phase) | Learned (action[6]) |
| Reward | Jaw-margin success condition | Gated multi-term grab reward |
| Status | Reach training (7.5M steps) | Working grab v0 (97%), v5 pending |

The Isaac Sim approach attempts a harder task (full grasp + lift in one policy) but has the advantage of GPU parallelism to compensate for the larger search space. Whether the 4096-env advantage is enough to overcome the harder credit-assignment problem (compared to MuJoCo's decomposed sub-policies) is an open question at project end.

### The Reward Shaping Challenge

The five reward iterations for the grab task illustrate the central challenge in dense reward manipulation RL: the policy will find and exploit any loophole in the reward function. Each version closed one loophole but opened another:

- v0: reward is achievable via self-collision (close proximity + no physics constraint)
- v1: reward is achievable via hovering indefinitely (ungated orientation bonus)
- v2: reward is achievable via holding cube below threshold + random closing
- v3: reward is achievable via finger-drag (approach correctly, never close)
- v4: reward is achievable via safe hover (stay above threshold, never descend)

v5's design principle is to return to the only configuration empirically proven to work (v0) and change exactly one thing (enable self-collision). This conservative iteration discipline — make one change at a time and verify each change doesn't introduce a new exploit — is slow but reliable.
