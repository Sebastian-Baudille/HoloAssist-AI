# Training runs — catalogue

Quick-reference for every meaningful Isaac Sim RL run in the project.
For setup/install history see [ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md).
For forward plan see [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md).
For launch commands see [LAUNCH.md](LAUNCH.md).

Logs live **outside** this repo at:
```
C:\Users\sebas\Github\IsaacLab\logs\rsl_rl\
```

---

## Naming scheme

```
<task>-r<reward_version>-run<N>       full training runs
<task>-r<reward_version>-pretest      wiring-check runs (4 envs x 5 iters)
```

- `<task>` — short task name (grab, reach, pickplace, …)
- `r<reward_version>` — reward design version (r0, r1, r2 …)
- `run<N>` — manual run counter, bumped per full training attempt
- `pretest` — bucket for verification runs (smoke tests) so they don't pollute the full-run TensorBoard view

Each timestamped subfolder inside is one training session
(e.g. `grab-r0-run1\2026-06-02_00-02-16\`).

### Run counter is manual

The cfg default sets `experiment_name = "grab-rN-run1"`. To use a different name (pretest, run2, run3, …) you must override at the CLI:

```powershell
# Pretest (wiring check)
.\isaaclab.bat -p ...train.py --task Template-... --num_envs 4 --max_iterations 5 --headless --experiment_name grab-r3-pretest

# First full training of a fresh reward — cfg default is used
.\isaaclab.bat -p ...train.py --task Template-... --num_envs 4096 --max_iterations 2500 --headless

# Re-running with same reward (different seed/iterations) — bump runN manually
.\isaaclab.bat -p ...train.py --task Template-... --num_envs 4096 --max_iterations 2500 --headless --experiment_name grab-r3-run2
```

If `--experiment_name` is omitted, the cfg default (`run1`) is used — the
new run lands as a sibling timestamp inside `run1`, which is misleading.

---

## Reward versions

### Reach task — `tasks/direct/reach/rewards/`

| Reward | File | Status | Notes |
|---|---|---|---|
| `dense_reach_v0` | dense_reach_v0.py | Early | First dense distance reward |
| `dense_reach_v1` | dense_reach_v1.py | **Operational baseline** | The working reach policy |
| `dense_reach_v2` | dense_reach_v2.py | Over-penalised | Robot folded into itself |
| `dense_reach_v3` | dense_reach_v3_ik.py | IK-guided | Also folded |

### Grab task — `tasks/direct/grab/rewards/`

| Reward | File | Status | Notes |
|---|---|---|---|
| `dense_grab` (v0) | dense_grab.py | **Working baseline** | 6 terms, side-sprawl approach, 5cm lift |
| `dense_grab_v1` | dense_grab_v1.py | Hover trap | 7 terms, overhead approach is correct but never descends |
| `dense_grab_v2` | dense_grab_v2.py | Failed | Rebalanced scales + time penalty intended to break v1 hover; instead enabled new exploits (lateral misalignment + hold-below-threshold) |
| `dense_grab_v3` | dense_grab_v3.py | Partial success | **Strategic retreat to v0's design.** Inherits v0's 6 terms unchanged (same proximity-gated alignment, same geometric grasped flag, same condition definitions). Adds 1 new term: tiny `elbow_up` posture nudge (max 30/episode). Cfg layer adds: self-collision ON (inherited from V1), success threshold 5cm→10cm, lift scale 50→10, success bonus 200→800. The lift+success rebalance maintains v0's "per-step accumulation < terminal success" invariant that v1 and v2 each violated. Approach posture works; gripper does not close — see run entry below for details. |
| `dense_grab_v4` | dense_grab_v4.py | Failed | **v3 + anti-drag (-1.0/step when finger frame ≤ table + 0.20) + grasp_act 1.0→2.0.** Anti-drag pushed policy AWAY from table but didn't pull it toward grasping — converged to "safe-hover" state at mean_reward ~13 (worst result of all attempts). |
| `dense_grab_v5` | dense_grab_v5.py | Ready to train | **Conservative return to v0.** Inherits v0's exact reward scales (lift 50, success 200, orient 0.3 proximity-gated, grasp_activation 1.0) + 5cm threshold — the only design empirically proven to find grasping. Adds only: PhysX self-collision (fixes v0's arm-through-arm flaw) + tiny elbow_up posture nudge (max 30/episode). NO scale rebalancing, NO new exploration constraints. Drops the 10cm aspiration. |

---

## Training runs

### Grab v0 (keeper) — `grab-r0-run1\2026-06-02_00-02-16\`

**Status**: Working — first end-to-end grab + lift policy.

| Field | Value |
|---|---|
| Reward | `dense_grab` (v0) — 6 terms, gated alignment |
| Self-collision | OFF (PhysX default) |
| Success threshold | cube lift > 5 cm |
| Final checkpoint | `model_3000.pt` |
| Mean reward | 196 / 200 (~97% success rate) |
| Mean episode length | 51 / 200 (fast termination on success) |
| Behaviour | Side-sprawl approach, grabs cube quickly, lifts ~5cm |
| Limitation | Visible self-collision (arm folds through itself) |
| TensorBoard tag | `grab-r0-run1\2026-06-02_00-02-16` |
| Play command | `play.py --task Template-Holoassist-Grab-Direct-v0` |

### Grab v1 (hover-trap reference) — `grab-r1-run1\2026-06-02_12-44-47\`

**Status**: Failed by design — preserved as reward-tuning reference.

| Field | Value |
|---|---|
| Reward | `dense_grab_v1` — 7 terms (added approach_height), ungated orient |
| Self-collision | ON (PhysX) |
| Success threshold | cube lift > 10 cm |
| Final checkpoint | `model_2999.pt` |
| Mean reward | 316 (misleading — see notes) |
| Mean episode length | 200 / 200 (no terminations, always timeout) |
| Behaviour | Arm achieves perfect overhead posture + wrist-down orientation, then hovers indefinitely above cube without closing gripper |
| Diagnosis | `rew_scale_orient_align = 1.5` (ungated) + `rew_scale_approach_height = 2.0` make hovering more rewarding than the descent-grasp-lift trajectory |
| TensorBoard tag | `grab-r1-run1\2026-06-02_12-44-47` |
| Play command | `play.py --task Template-Holoassist-Grab-Direct-v1` |

### Grab v2 (failed) — `grab-r2-run1\2026-06-02_14-58-26\`

**Status**: Failed — preserved as reward-tuning reference.

| Field | Value |
|---|---|
| Reward | `dense_grab_v2` — 8 terms (v1's 7 + time penalty) |
| Self-collision | ON (PhysX) |
| Success threshold | cube lift > 10 cm |
| Final checkpoint | `model_3000.pt` (stopped early after diagnosis) |
| Mean reward | ~750 (misleading — see notes) |
| Mean episode length | 200 / 200 (no terminations, always timeout) |
| Behaviour | Arm approaches cube with sloppy alignment, closes gripper beside it (not around it), occasionally bumps cube. Some envs hold cube at low height (< 10 cm) to milk per-step lift_bonus. Entropy climbed instead of decreasing — no clean strategy converged. |
| Diagnosis | Rebalanced v1 scales (orient 1.5→0.3, lift 80→100, grasp_act 1→5, success 200→300) made per-step reward accumulation worth more than terminal success. Two new exploits emerged: (a) hold cube just below 10cm threshold for continuous lift_bonus, (b) close gripper near (not around) cube for grasp_activation reward without alignment. |
| Lesson | New design rule: sum of per-step rewards × episode length MUST be < success bonus. |
| TensorBoard tag | `grab-r2-run1\2026-06-02_14-58-26` |
| Play command | `play.py --task Template-Holoassist-Grab-Direct-v2` |

### Grab v3 (partial success) — `grab-r3-run1\<timestamp>\`

**Status**: Partial — approach posture is correct, grasp does not happen.

| Field | Value |
|---|---|
| Reward | `dense_grab_v3` (v0's 6 terms + elbow_up nudge) |
| Self-collision | ON (PhysX) |
| Success threshold | cube lift > 10 cm |
| Mean reward | ~190 (no terminations) |
| Mean episode length | 200 / 200 (always timeout) |
| Behaviour | Arm reaches cube with proper overhead posture (elbow up, wrist down, gripper centered over cube). Fingers descend to the table surface and rest there with the cube between them. Gripper does not close. No lifts succeed. |
| Diagnosis | "Finger drag" trap. Fingers reach table level where friction blocks the inward closing motion. Policy is stuck in a local optimum earning ~190/episode from xy_align + orient_align + elbow_up while positioned at the cube — closing has no clear reward gradient because accidental closures cannot reach the 10 cm threshold to trigger success. v0 escaped this by accident (5 cm threshold meant random closures sometimes lifted high enough to fire success); v3's higher threshold removes that lucky path. |
| Lesson | A well-shaped reward can still produce stuck behaviour if the policy can earn enough reward in a position that physically prevents progress. Need an explicit signal that pushes the gripper above the table during the closing transition. |
| TensorBoard tag | `grab-r3-run1` |
| Play command | `play.py --task Template-Holoassist-Grab-Direct-v3` |

### Earlier grab attempts (historical) — `holoassist_grab_direct\`

Three early-exploration runs from 2026-06-01 evening (`23-46-36`,
`23-50-23`, `23-58-22`) — pre-keeper experimentation before the v0
reward was settled. Kept under the legacy folder name; not promoted.

### Reach runs (historical) — `holoassist_reach_direct\`

Multiple training runs from 2026-05-30 to 2026-05-31 spanning v0/v1/v2/v3
reward iterations. v1 reward (operational baseline) is the keeper policy;
exact timestamp-to-reward-version mapping not captured in this doc.

### Cartpole — `cartpole_direct\2026-05-28_19-28-11\`

Early Isaac Lab + RSL-RL validation. Not used for policy deployment.

---

## TensorBoard

Launch once, view all runs:

```powershell
cd C:\Users\sebas\Github\IsaacLab
.\isaaclab.bat -p -m tensorboard.main --logdir "C:\Users\sebas\Github\IsaacLab\logs\rsl_rl" --port 6006
```

Open http://localhost:6006. In the left sidebar, all experiment folders
appear; tick the runs you want to compare.

---

## Loading a specific model

`play.py` auto-discovers the latest checkpoint in the experiment folder
matching the task's PPO cfg `experiment_name`. To override:

```powershell
# Load a specific checkpoint
play.py --task Template-... --checkpoint "C:\...\grab-r0-run1\2026-06-02_00-02-16\model_3000.pt"

# Load latest from a named run within the same experiment folder
play.py --task Template-... --load_run "2026-06-02_00-02-16"
```
