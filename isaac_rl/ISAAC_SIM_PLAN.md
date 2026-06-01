# Isaac Sim RL Migration Plan

Forward-looking plan for porting the UR3e + RG2 RL pipeline from ROS 2 + Gazebo
([../ur3e_rl_ws/](../ur3e_rl_ws/)) to **NVIDIA Isaac Sim 5.1 + Isaac Lab** on
native Windows 11. The two stacks coexist *transitionally* — Isaac is the
long-term primary; ROS stays alive during the port for comparison runs.

For the **setup log + gotchas + version pins**, see
[ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md). This doc is the *forward plan*;
that one is the *backward record*.

---

## Why migrate

| | Current (ROS 2 + Gazebo) | Target (Isaac Sim + Isaac Lab) |
|---|---|---|
| Parallel envs | 4 (CPU-bound — separate Gazebo per worker) | 1024–4096 (GPU-vectorised, single process) |
| Throughput | ~hundreds steps/s | 50k–100k+ steps/s |
| 200 k-step run | ~30 min | < 1 min |
| ROS in training loop | yes (rclpy + bridges + topic plumbing) | no — pure GPU tensor ops |
| Sim-to-real tools | manual | built-in domain randomisation |
| Gripper modelling | fixed-open hack (URDF makes joints `type="fixed"`) | articulated parallel-jaw (7 DOF, real geometry) |
| Setup cost | done | ~1 day install + ~2 weeks port (in progress) |

The bottleneck today is throughput. Isaac Lab removes it AND lets us model the
gripper properly for the first time.

---

## Status (as of 2026-05-31)

| Phase | What | Status |
|---|---|---|
| 1 | Isaac Sim 5.1 + Isaac Lab install, cartpole verified | ✅ Done |
| 2 | Scaffold `holoassist_tasks` project via `isaaclab.bat --new` | ✅ Done |
| 3 | URDF → USD conversion | ✅ Done — `assets/usd/ur_onrobot_prepared/ur_onrobot_prepared.usd` validated via [scripts/robot_test_v0.py](scripts/robot_test_v0.py) |
| 4 | Port reach env (`DirectRLEnv` under `tasks/direct/reach/`) | ✅ Done — full strategy-module env wired; sanity test + smoke train both pass |
| 4b | Pick-and-place task (sibling under `tasks/direct/pick_place/`) | Future |
| 5 | Train PPO to convergence on reach task (4096 envs) | 🟡 In progress — 500 iters done (reward -100 → -20). Need ~1000-1500 more for full convergence + entropy_coef tuning |
| 6 | Evaluate + export trained policy to ONNX | Pending |
| 7 | Deploy ONNX policy back into ROS Gazebo for sim-to-real comparison | Pending |
| 8+ | Domain randomisation, real-robot deployment, additional tasks | Future |

See [ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md) §Status for the detailed
"last verified working" state, §11 for the Phase 4 mapping-by-mapping
narrative, and §12 for the Phase 5 initial training-run results.

---

## Locked-in decisions

| Decision | Choice | Why |
|---|---|---|
| Platform | Windows 11 native (Isaac Sim) + WSL2 Ubuntu 22.04 (xacro tooling only) | Isaac Sim runs natively on Windows; WSL is one-shot for URDF generation |
| GPU | RTX 4070 Ti SUPER (16 GB), driver **580.88** (not 595.xx — see Progress.md) | Validated combo |
| Isaac Sim | 5.1.0 Workstation installer at `C:\isaacsim` | NOT the pip install — binary works cleanly on Windows |
| Isaac Lab | `main` branch at `C:\Users\sebas\Github\IsaacLab` | Targets Isaac Sim 5.1; release/2.3.0 was for older Isaac Sim |
| RL framework | **RSL-RL** (Isaac Lab default) | Fully GPU-vectorised PPO; fastest with 4096 envs |
| Env API | `DirectRLEnv` | Closer to legacy Gym pattern; less manager indirection than `ManagerBasedRLEnv` |
| URDF source | `ros2_ws/src/ur_onrobot/ur_onrobot_description/urdf/ur_onrobot.urdf.xacro` | Articulated RG2; preserves Phase 4b pick-and-place option. NOT `ur3e_rg2_benchtop.urdf.xacro` (fixed gripper hack). |
| Mimic chain | Stripped from URDF in [scripts/prepare_urdf.py](scripts/prepare_urdf.py); gripper coupling replicated in Python | Isaac Sim 5.1 URDF importer loses finite limits on multi-level mimic chains. Strip-then-Python-couple is reliable. |
| Project package name | `holoassist_tasks` (plural) — umbrella for multiple sibling tasks under `tasks/direct/` | Future-proof for `reach`, `pick_place`, `insertion`, etc. |
| First task | **Reach** (`tasks/direct/reach/`) | Match what the legacy `UR3ePickPlaceEnv` actually trains (per [TRAINING_NOTES.md](../ur3e_rl_ws/TRAINING_NOTES.md): "Sufficient for the reach task"); evolve to pick-place in Phase 4b |
| Action space | 6-D **true joint-delta** (rad/step), `±0.24` clamp | Fixes legacy confusion (named "delta", implemented as absolute target slew-limited externally). True delta is cleaner. |
| Observation space | 12-D = `6 joint_pos + 3 EE_pos + 3 target_pos` (recommended; user confirmation pending) | Minimal sufficient state for reach. Drops noisy legacy fields (`grasped`, `gripper_state`, `bin_pos`). |
| Reward | Dense, reach-only for v1: `-1.0 * dist_to_target` always + `+10.0` impulse on success | Strip legacy's pick-place reward shape (grasp bonus, transport gradient); restore in Phase 4b |
| Episode length | ~200 steps × decimation × dt ≈ 13-20 s | Matches legacy |

The full Q1-Q5 design rationale lives in the project memory file
`project-phase4-design-decisions.md`.

---

## Target repo layout (current state + Phase 4 additions)

```
HoloAssist-AI/
├── ur3e_rl_ws/                          legacy ROS 2 + Gazebo stack (untouched, runs as comparison)
├── ros2_ws/                             legacy HoloAssist sim + ur_onrobot description (untouched)
├── clustering/                          legacy clustering pipeline (untouched)
│
└── isaac_rl/                            ⬅ THE ISAAC SIM RL WORK (this folder)
    ├── ISAAC_SIM_PLAN.md                ⬅ this file (forward plan)
    ├── ISAAC_SIM_Progress.md            setup log + gotchas + changelog
    ├── .gitignore
    │
    ├── assets/                          robot asset pipeline (Phase 3 output)
    │   ├── meshes/                      22 mesh files (14 UR3e + 8 RG2)
    │   ├── urdf/
    │   │   ├── ur_onrobot.urdf          raw xacro flatten
    │   │   └── ur_onrobot_prepared.urdf mimic-stripped, URIs rewritten
    │   └── usd/
    │       └── ur_onrobot_prepared/     imported USD asset (layered)
    │
    ├── scenes/
    │   └── ur_onrobot_test_stage.usda   interactive test stage (optional)
    │
    ├── scripts/                         standalone Python tools
    │   ├── prepare_urdf.py              URI rewrite + mimic strip (Phase 3)
    │   ├── inspect_urdf.py              structural inspection (Phase 3)
    │   └── robot_test_v0.py             Isaac Lab smoke test (Phase 3 validation)
    │
    └── holoassist_tasks/                Isaac Lab task project (Phase 2 scaffold)
        ├── pyproject.toml
        ├── scripts/                     project-local CLIs (random_agent.py, train.py, etc.)
        └── source/holoassist_tasks/holoassist_tasks/tasks/direct/
            ├── _template_cartpole/      original generator output, renamed for clarity (Phase 4a)
            ├── reach/                   ⬅ Phase 4 lands here
            │   ├── __init__.py             gym.register("Template-Holoassist-Reach-Direct-v0")
            │   ├── reach_env.py             HoloassistReachEnv(DirectRLEnv)
            │   ├── reach_env_cfg.py         HoloassistReachEnvCfg
            │   └── agents/rsl_rl_ppo_cfg.py PPO hyperparameters
            ├── pick_place/              ⬅ Phase 4b sibling
            │   └── (same shape)
            └── (future siblings: reach_oriented, insertion, ...)
```

**Two key structural points:**

1. **Multi-task support is built in.** `tasks/direct/` is auto-discovered by
   `import_packages()` in `tasks/__init__.py`. Dropping a new `<task_name>/`
   folder under it (with its own `__init__.py + gym.register(...)`)
   auto-registers a new Isaac Lab task. No central manifest to edit.

2. **Strict dual-stack isolation.** `isaac_rl/` and the legacy
   `ur3e_rl_ws/`+`ros2_ws/` are read-only references for each other. No imports
   across the boundary. Assets are *copied*, not symlinked. The legacy stack
   stays runnable for sim-to-real comparison in Phase 7.

---

## Phased plan

### Phase 1 — Install ✅ Done
NVIDIA driver 580.88 → Isaac Sim 5.1.0 Workstation install → Isaac Lab clone
+ `isaaclab.bat --install rsl_rl`. Verified end-to-end via cartpole train.
Full step-by-step + gotchas (driver crash, `_isaac_sim` junction, tensordict
0.7.2 + h5py 3.11.0 pins) in [ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md) §§ 1–8.

### Phase 2 — Scaffold project ✅ Done
`isaaclab.bat --new` with: External / Direct / single-agent / rsl_rl. Generated
`holoassist_tasks/` with a cartpole template inside. Verified end-to-end by
training the template task. Kept the scaffold structure; only renaming the
inner cartpole folder in Phase 4a.

### Phase 3 — URDF → USD ✅ Done
WSL2 + ROS 2 Humble + `ur_description` + `ur_onrobot_description` toolchain →
xacro flatten → mesh mirror + URI rewrite + `<mimic>` strip via
[scripts/prepare_urdf.py](scripts/prepare_urdf.py) → Isaac Sim URDF Importer
with Static Base + Default Density 1000 + Natural Frequency 50 + Damping 1.0
+ Convex Hull → output `assets/usd/ur_onrobot_prepared/`.
Verified by [scripts/robot_test_v0.py](scripts/robot_test_v0.py): 13 actuated
joints, parked pose held under gravity, no oscillation, no drift.
Full narrative + 7 troubleshooting entries + lessons in
[ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md) § 10.

### Phase 4 — Reach env port 🟡 In progress

Port the legacy reach env (from `ur3e_rl_ws/`) into a fresh `DirectRLEnv` at
`tasks/direct/reach/`. The legacy is read as a *reference only* — new code is
written from scratch in Isaac Lab idioms.

**Sub-step decomposition:**

| Sub-step | What | Owner |
|---|---|---|
| 4a | Rename `tasks/direct/holoassist_tasks/` → `_template_cartpole/`; create empty `tasks/direct/reach/` skeleton (5 stub files); pip reinstall; verify both tasks register | Me |
| 4b | **Mapping #1**: port `constants.py` → fields in `reach_env_cfg.py`. Workspace bounds, action scale, home pose become typed `@configclass` attributes. Uses real UR3e joint limits (not loose ±2π). | Me, with user review |
| 4c | **Mapping #3**: port `ros_interface.build_observation` → `_get_observations`. Torch-vectorised. 12-D output: `6 joint_pos + 3 EE_pos + 3 target_pos`. Drop noisy legacy fields. | Me, with user review |
| 4d | **Mapping #2**: port `reward.py` → `_get_rewards`. Reach-only: `-1.0 * dist_to_target` always + `+10.0` impulse on success. Drop grasp bonus + bin transport (restore in 4b for pick-place). | Me, with user review |
| 4e | **Mapping #4**: port `safety_checker.make_safe_target` → inline in `_pre_physics_step`. 3 lines: clip action to ±scale, add to current joint pos, clamp to URDF limits. True joint-delta semantics. | Me, with user review |
| 4f | **Mapping #5**: write `HoloassistReachEnv` class structure — reset (randomise target pose), done conditions (success / EE z too low / max steps), assemble methods from 4b-4e | Me, with user review |
| 4g | **Mapping #6+#7**: `gym.register("Template-Holoassist-Reach-Direct-v0", ...)` in `__init__.py`; tune `rsl_rl_ppo_cfg.py` for 6-DOF arm (actor `[256, 256, 128]`, more env steps per iter) | Me, with user review |
| 4h | **Sanity test**: `reach_test_v0.py` spawns env + steps with random actions for 100 steps. Verifies obs/action shapes, no NaNs, episodes terminate. | Me |
| 4i | **Smoke train**: `train.py --task Template-Holoassist-Reach-Direct-v0 --num_envs 64 --max_iterations 50 --headless`. Don't expect convergence — verify training loop runs end-to-end. | Me |

Each Mapping pauses for user review: I summarise (LEGACY / NEW ISAAC / KEEP /
CHANGE / IMPROVE), user confirms, I write the code.

**Files NOT created in Phase 4** (deferred to 4b pick-place):
- Cube as rigid body (target marker is a kinematic visual prim for v1)
- Gripper open/close helper (gripper stays fixed-closed in v1)
- Grasp detection logic
- Phase-conditional reward shaping (reach is single-phase)

### Phase 4b — Pick-and-place env (sibling task)

Once reach trains to convergence:
- New folder `tasks/direct/pick_place/` (sibling, not a fork)
- Cube becomes a `RigidObjectCfg` with mass + friction matching legacy
- Action space gains gripper dimension (7-D total: 6 arm-delta + 1 gripper open/close binary)
- Observation gains EE-cube proximity + cube-bin distance fields
- Reward gains phase-conditional shaping (pre-grasp → distance to cube; post-grasp → distance to bin; impulse on grasp transition; +10 on cube-in-bin)
- Gripper controlled via Python helper that drives all 7 joints together (per Phase 3 mimic workaround)

Reach env stays untouched. Both tasks coexist in `tasks/direct/`.

### Phase 5 — Train to convergence

```powershell
.\isaaclab.bat -p holoassist_tasks\scripts\rsl_rl\train.py `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 4096 --headless
```

- 4070 Ti SUPER: estimated 30-100 k steps/s
- Expected 5–30 M steps to convergence (10–60 min wall time)
- Drop `--num_envs` to 1024 on OOM
- Monitor via TensorBoard at `logs/rsl_rl/template_holoassist_reach/<run>/`
- Triage flat reward via same playbook as legacy [TRAINING_NOTES.md](../ur3e_rl_ws/TRAINING_NOTES.md)

### Phase 6 — Evaluate + export to ONNX

```powershell
.\isaaclab.bat -p holoassist_tasks\scripts\rsl_rl\play.py `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 16 `
    --checkpoint logs\rsl_rl\template_holoassist_reach\<run>\model_<N>.pt
```

Produces `exported/policy.pt` (TorchScript) and `exported/policy.onnx`.
Quantitative eval: success rate at distance tolerance 4 cm, mean steps to
success, action smoothness.

### Phase 7 — Deploy ONNX policy back into ROS Gazebo

Adapt the legacy [rl_policy_node.py](../ur3e_rl_ws/src/ur3e_policy_controller/ur3e_policy_controller/rl_policy_node.py):
```python
import onnxruntime as ort
session = ort.InferenceSession("policy.onnx", providers=["CPUExecutionProvider"])
action = session.run(None, {"obs": obs.astype(np.float32)})[0]
```

**Critical**: the observation built by the ROS node must match `_get_observations`
exactly — order, units, frames, normalisation. Add a sanity assert on
observation shape at both ends. This is the #1 silent sim-to-real failure mode.

Test sequence:
1. Replay Isaac Lab observations through the ROS node → verify identical actions
2. Run policy in legacy Gazebo sim ([ur3e_pick_place_world.launch.py](../ur3e_rl_ws/src/ur3e_gazebo_sim/launch/ur3e_pick_place_world.launch.py))
3. Compare success rate vs the Isaac-trained version evaluated in Isaac Sim
4. Run on real UR3e with `SafetyChecker` re-enabled, `max_delta_rad=0.05` (conservative for first real-robot run)

### Phase 8+ — Future work

- **Domain randomisation** in `_reset_idx`: joint friction ±20%, action noise σ=0.01 rad, joint pos noise σ=0.001 rad, target pos noise σ=0.005 m (matches D435i depth noise)
- **Vision-based observation**: add Isaac Sim camera + replace ground-truth target pose with cube detection from camera image. Pre-requires reach-from-pixels works.
- **Curriculum learning**: tighten success tolerance over training (4 cm → 2 cm → 1 cm)
- **Real-robot fine-tuning**: short on-robot training run with the sim-trained policy as initialisation
- **Additional task variants** (sibling folders under `tasks/direct/`): insertion, stacking, handover

---

## Time budget (revised based on Phase 1-3 actuals)

| Phase | Budget | Actual / estimate |
|---|---|---|
| 1 — Install | 1-2 days | 1 day (incl. driver downgrade troubleshooting) |
| 2 — Scaffold | 2-4 hours | ~2 hours |
| 3 — URDF → USD | 1 week (planning) | ~3 days (mimic workaround took half a day) |
| 4 — Reach env port | 1 week | In progress |
| 4b — Pick-place env | 1 week | Future |
| 5 — Train reach | < 1 hour wall | TBD |
| 6 — Eval + export | 1-2 hours | TBD |
| 7 — ROS deployment | 2-4 hours | TBD |

---

## What stays from the legacy stack, what's replaced

**Stays (referenced read-only during port; runs unchanged for Phase 7 comparison):**
- [reward.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/reward.py) logic — re-implemented torch-vectorised in `_get_rewards`
- [ros_interface.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/ros_interface.py) observation layout — re-implemented in `_get_observations`
- [SafetyChecker](../ur3e_rl_ws/src/ur3e_safety_layer/ur3e_safety_layer/safety_checker.py) — replicated inline in `_pre_physics_step` (not as a separate package)
- [Gazebo sim](../ur3e_rl_ws/src/ur3e_gazebo_sim/) — kept as sim-to-real test bench for Phase 7
- [rl_policy_node.py](../ur3e_rl_ws/src/ur3e_policy_controller/ur3e_policy_controller/rl_policy_node.py) — model loader swapped to ONNX Runtime in Phase 7
- URDF source (`ur_onrobot.urdf.xacro` from `ros2_ws/`) — flattened once in Phase 3, USD lives in `isaac_rl/assets/`

**Replaced (legacy version no longer used in Isaac side):**
- [ur3e_pick_place_env.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/envs/ur3e_pick_place_env.py) → `tasks/direct/reach/reach_env.py` (DirectRLEnv)
- [train_ppo.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_ppo.py) / [train_ppo_parallel.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/train_ppo_parallel.py) → `holoassist_tasks/scripts/rsl_rl/train.py`
- SB3 PPO → RSL-RL PPO
- 4 × Gazebo `SubprocVecEnv` → 4096-env GPU vec
- MoveIt collision checking during training → reward penalty + termination
- Fixed-gripper hack in `rg2_fixed.xacro` → articulated 7-DOF gripper from `ur_onrobot.urdf.xacro`

---

## Open questions (revisit at each milestone)

- **Behaviour-cloning warm-start** (legacy [pretrain_from_demos.py](../ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/pretrain_from_demos.py)): RSL-RL doesn't support BC out of the box. Skip for v1; revisit if PPO struggles to converge.
- **Camera-based observation**: Defer until ground-truth policy converges reliably. Then add Isaac Sim camera + replicate the legacy clustering pipeline as a perception module.
- **Pick-and-place vs reach**: Reach in Phase 4 (matches what legacy actually trains); pick-place in Phase 4b once reach works end-to-end.
- **Real-robot transfer**: Will be the third major effort after Phase 7 sim-to-sim validation. Out of scope for this plan revision.

---

## References

- [ISAAC_SIM_Progress.md](ISAAC_SIM_Progress.md) — setup log, version pins, troubleshooting, lessons learned (the **full** command reference lives in §Quick commands)
- [../ur3e_rl_ws/TRAINING_NOTES.md](../ur3e_rl_ws/TRAINING_NOTES.md) — legacy training empirical lessons
- [../ARCHITECTURE.md](../ARCHITECTURE.md) — broader project context
- [../CLAUDE.md](../CLAUDE.md) — agent project instructions (covers legacy stack only)
- [Isaac Lab docs](https://isaac-sim.github.io/IsaacLab/main/)
- [Isaac Sim 5.1 URDF Importer docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/importer_exporter/import_urdf.html)

---

## Command cheat sheet (most-used)

For the full annotated set with every flag explained, see [ISAAC_SIM_Progress.md § Quick commands](ISAAC_SIM_Progress.md#quick-commands-current-toolchain).
This is the short version — what you'll run day to day.

All commands assume you're in `C:\Users\sebas\Github\IsaacLab`.
`<HOLO>` = `C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI`.

| Task | Command |
|---|---|
| **Sanity-test the reach env** (random actions, headless) | `.\isaaclab.bat -p "<HOLO>\isaac_rl\scripts\reach_test_v0.py" --headless --num_envs 16 --num_steps 100` |
| **Smoke train** (~1 min) | `.\isaaclab.bat -p "<HOLO>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" --task Template-Holoassist-Reach-Direct-v0 --num_envs 64 --max_iterations 50 --headless` |
| **Convergence train** (~30-45 min) | `.\isaaclab.bat -p "<HOLO>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" --task Template-Holoassist-Reach-Direct-v0 --num_envs 4096 --max_iterations 1500 --headless` |
| **Play trained policy** (GUI) — auto-finds latest checkpoint | See multi-line PowerShell snippet in Progress.md § Quick commands |
| **TensorBoard** (separate terminal) | `.\_isaac_sim\python.bat -m tensorboard.main --logdir logs\rsl_rl\holoassist_reach_direct` → open http://localhost:6006 |
| **Robot USD smoke test** (no env) | `.\isaaclab.bat -p "<HOLO>\isaac_rl\scripts\robot_test_v0.py" --headless` |
| **Inspect URDF** (WSL) | `wsl -d Ubuntu-22.04 -- bash -c "python3 ~/holoassist/isaac_rl/scripts/inspect_urdf.py"` |
| **Re-prepare URDF after xacro change** (WSL → Isaac Sim) | Run `prepare_urdf.py` in WSL, then re-run the URDF Importer GUI step (Phase 3) |
| **Pip reinstall holoassist_tasks** (after folder/setup.py changes) | `.\_isaac_sim\python.bat -m pip install -e "<HOLO>\isaac_rl\holoassist_tasks\source\holoassist_tasks"` |

### Common gotchas (quick reference — full versions in Progress.md § Troubleshooting)

- **`play.py --checkpoint`**: needs the full path to a `model_N.pt` file, NOT a run folder
- **PowerShell + Python**: always `print(..., flush=True)` or output may be buffered
- **`gripper_tcp` not found**: merged out by URDF import — use `left_inner_finger` as EE proxy
- **Robot at ground level instead of z=1**: set `ArticulationCfg.init_state.pos=(0, 0, 1.0)` explicitly
- **Episodes terminate at ~10 steps**: bump `cfg.min_ee_clearance_below_base_m`, or move home pose higher
- **Action scale check**: `scale × control_rate ≤ 75% of joint velocity limit` (UR3e: ~3 rad/s)
