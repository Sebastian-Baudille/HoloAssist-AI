# isaac_rl — Isaac Sim RL workspace

This folder is where HoloAssist-AI's RL stack is being ported from Gazebo to
**NVIDIA Isaac Sim 5.1 + Isaac Lab** on native Windows. It contains the
project scaffold, task code, assets, and run scripts.

For the forward plan and phased roadmap, see [ISAAC_SIM_PLAN.md](ISAAC_SIM_PLAN.md).
For the broader project context, see [../ARCHITECTURE.md](../ARCHITECTURE.md).

---

## Status — 2026-05-31

| Stage | Status |
|---|---|
| Phase 1 — Isaac Sim + Isaac Lab install, cartpole verified | **Done** |
| Phase 2 — Scaffold `holoassist_tasks` via `isaaclab.bat --new` | **Done** |
| Phase 3 — UR3e + RG2 URDF → USD | **Done** |
| Phase 4 — Port reach env (`DirectRLEnv` under `tasks/direct/reach/`) | **Done** — full env wired, all 9 sub-mappings landed, sanity test + 50-iter smoke train both pass |
| Phase 5 — Train PPO to convergence | **In progress** — initial 500-iter run @ 4096 envs shows clear learning (reward -100 → -20, mean episode length stabilises at ~110/200). Needs more iters + tuning for full convergence |
| Phase 6 — Eval + export policy to ONNX | Pending |
| Phase 7 — Deploy ONNX back into ROS Gazebo | Pending |

**Last verified working**: 500-iteration PPO training run @ 4096 envs, ~20 min wall-clock on RTX 4070 Ti SUPER. Reward improves monotonically. Policy partially converged — reaches targets ~60-70% of the time. Checkpoint saved at `IsaacLab/logs/rsl_rl/holoassist_reach_direct/<timestamp>/model_499.pt`. `play.py` evaluation in GUI also works (with the path-resolution gotcha noted in troubleshooting).

---

## Hardware + software stack (the combo that works)

| Component | Value | Notes |
|---|---|---|
| GPU | NVIDIA GeForce RTX 4070 Ti SUPER (16 GB) | Ada Lovelace, sm_89 |
| CPU | Intel Core i7-14700K (16 logical cores) | |
| RAM | 32 GB | |
| OS | Windows 11 Pro, 25H2, build **26200** | The 26200 build is relevant — driver 595.79 has known crashes on this combo |
| **NVIDIA driver** | **580.88** | NVIDIA's officially tested version for Isaac Sim 5.1. **Do not use 595.xx** |
| Isaac Sim | **5.1.0** Workstation Installer at `C:\isaacsim` | Binary install (not pip) |
| Isaac Lab | **main branch** at `C:\Users\sebas\Github\IsaacLab` | NOT `release/2.3.0` — main targets Isaac Sim 5.1 |
| Python | 3.11 (Isaac Sim bundled) | Not the system Python; runs via `_isaac_sim\python.bat` |

---

## Critical version pins (must-use, do not auto-upgrade)

These pins are required because the latest pip wheels have Windows + Isaac Sim 5.1 ABI issues that aren't caught by pip's dependency resolver.

```powershell
.\_isaac_sim\python.bat -m pip install "tensordict==0.7.2" "h5py==3.11.0" --force-reinstall --no-deps
```

| Package | Required version | Why |
|---|---|---|
| `tensordict` | **0.7.2** | 0.12.x built against PyTorch 2.8 ABI; crashes on import in `_C.pyd` with access violation when used with PyTorch 2.7.0 |
| `h5py` | **3.11.0** | 3.16.x has Windows HDF5 DLL incompatibility — `ImportError: DLL load failed while importing _errors: The specified procedure could not be found` |
| `torch` | **2.7.0+cu128** | Pulled in by `isaaclab.bat --install`. Do not bump. |
| `rsl-rl-lib` | **5.0.1** | The version pulled in by `isaaclab.bat --install rsl_rl` on main branch |

Re-pin after any `isaaclab.bat --install` re-run, since pip may upgrade them.

---

## How it all wires together

```
C:\isaacsim\                              Isaac Sim 5.1.0 (Workstation installer)
└── python.bat                            ← bundled Python 3.11

C:\Users\sebas\Github\IsaacLab\           Isaac Lab (cloned, main branch)
├── isaaclab.bat                          ← all-purpose launcher
├── _isaac_sim → C:\isaacsim              ← JUNCTION (created manually, required)
├── source\
│   ├── isaaclab\                         ← editable installs into _isaac_sim's Python
│   ├── isaaclab_assets\
│   ├── isaaclab_contrib\
│   ├── isaaclab_mimic\
│   ├── isaaclab_rl\
│   └── isaaclab_tasks\
└── scripts\reinforcement_learning\rsl_rl\
    ├── train.py
    └── play.py

%USERPROFILE%\Github\41118 ...\HoloAssist-AI\isaac_rl\   ← THIS REPO
├── holoassist_tasks\                     project package (Phase 2 onward; cartpole scaffold to be gutted in Phase 4)
├── assets\                               UR3e + RG2 robot asset pipeline (Phase 3 output)
│   ├── meshes\                           22 mesh files (14 UR3e .dae/.stl + 8 RG2 .stl)
│   ├── urdf\
│   │   ├── ur_onrobot.urdf               raw xacro flatten
│   │   └── ur_onrobot_prepared.urdf      mimic-stripped, URIs rewritten — Isaac Sim input
│   └── usd\
│       └── ur_onrobot_prepared\          imported USD asset (layered: base, physics, robot, sensor)
├── scenes\
│   └── ur_onrobot_test_stage.usda        (optional) interactive scene referencing the USD
├── scripts\
│   ├── prepare_urdf.py                   URI rewrite + mimic strip
│   ├── inspect_urdf.py                   link/joint/inertia structural report
│   └── robot_test_v0.py                  Isaac Lab smoke test (parked pose verification)
├── ISAAC_SIM_PLAN.md                     forward roadmap
├── ISAAC_SIM_Progress.md                 ← this file
└── .gitignore
```

**Two environment variables** are set per-user (already done — `[System.Environment]::SetEnvironmentVariable(..., "User")`):
- `ISAACSIM_PATH = C:\isaacsim`
- `ISAACLAB_PATH = C:\Users\sebas\Github\IsaacLab`

---

## Installation log — what we actually did

Chronological, with each gotcha and the fix.

### 1. Isaac Sim 5.1.0 install (via Workstation installer)

Followed [official guide](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html).
Installed to `C:\isaacsim`. Ran `post_install.bat`.

### 2. First launch — Isaac Sim crashed mid-shader-compile

Symptom: GUI window opened, viewport stayed blank, Crashreporter popped after ~20 s.

Log showed:
```
rtx.scenedb.plugin.dll!carbOnPluginStartup+0x252db
carb.scenerenderer-rtx.plugin.dll
omni.hydra.rtx.plugin.dll
```

Tried: `clear_caches.bat`, `--reset-user`, `--/rtx/ecoMode/enabled=True`, `--no-window`. All still crashed.

**Root cause**: NVIDIA driver **595.79** has a known regression with Isaac Sim 5.1's Vulkan RTX path on Ada Lovelace GPUs. Confirmed by [NVIDIA forum thread on RTX 5070 Ti Blackwell](https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74) — affects multiple GPUs across 595.xx branch.

**Fix**: Downgrade NVIDIA driver to **580.88** (NVIDIA's officially tested version for Isaac Sim 5.1). Used the NVIDIA download page, "Custom Install" → "Perform clean installation". Required restart.

After downgrade: Isaac Sim 5.1 launched successfully on the second attempt (first one still rebuilt shader cache against the new driver).

### 3. Isaac Lab clone

```powershell
cd C:\Users\sebas\Github
git clone https://github.com/isaac-sim/IsaacLab.git
```

Initially considered `release/2.3.0` branch based on a forum post, but the official Isaac Lab docs confirm **main targets Isaac Sim 5.1.0**, so main is correct.

### 4. `.\isaaclab.bat --install rsl_rl` — failed: Python not found

Symptom:
```
Python was not found; run without arguments to install from the Microsoft Store
[ERROR] Unable to find any Python executable at path:
2. Python executable is not available at the default path: C:\Users\sebas\Github\IsaacLab\\_isaac_sim\python.bat
```

**Root cause**: `isaaclab.bat` doesn't honour the `ISAACSIM_PATH` env var on Windows — it requires a directory symlink at `IsaacLab\_isaac_sim` pointing at the Isaac Sim install.

**Fix**: Created a directory junction (works without admin):
```powershell
cd C:\Users\sebas\Github\IsaacLab
New-Item -ItemType Junction -Path "_isaac_sim" -Target "C:\isaacsim"
```

After the junction: `isaaclab.bat --install rsl_rl` proceeded.

### 5. Install hung mid-rsl_rl (or completed silently)

Symptom: terminal output stopped at the rsl_rl install phase. Prompt didn't return. Ctrl+C'd.

Verification showed most packages installed but the install hadn't finalized. Re-running `isaaclab.bat --install rsl_rl` later picked up where it left off and finished cleanly.

### 6. First training attempt — `tensordict._C.pyd` access violation

Symptom: cartpole training crashed during `import` chain:
```
Windows fatal exception: access violation
File "...\tensordict\utils.py", line 44 in <module>
...
File "...\rsl_rl\algorithms\distillation.py"
```

**Root cause**: pip installed the latest `tensordict==0.12.4`, which is built against PyTorch 2.8+ ABI. Isaac Lab installs PyTorch 2.7.0. The C extension binary is incompatible.

**Fix**:
```powershell
.\_isaac_sim\python.bat -m pip install "tensordict==0.7.2" --force-reinstall --no-deps
```

After: cartpole headless training worked. Reached reward ~226 in 50 PPO iterations within 12 seconds. Throughput ~4500 steps/s with 64 envs.

### 7. First `play.py` (GUI) attempt — `h5py` DLL error

Symptom: extension `isaaclab_tasks` failed to startup:
```
ImportError: DLL load failed while importing _errors: The specified procedure could not be found.
File "...\h5py\__init__.py", line 25, in <module>
    from . import _errors
```

**Root cause**: `h5py==3.16.0` (latest at install time) has a Windows HDF5 native DLL that needs a Windows API function not available in Isaac Sim's Python's DLL search context. Headless training didn't load `recorder_manager` (which imports h5py) so it never tripped this.

**Fix**:
```powershell
.\_isaac_sim\python.bat -m pip install "h5py==3.11.0" --force-reinstall --no-deps
```

After: `play.py` loaded past the import. GUI viewport opened, 16 cartpoles rendered, policy played back correctly.

### 8. Cartpole verified end-to-end

Both confirmed working:
- `train.py --task Isaac-Cartpole-Direct-v0 --num_envs 64 --headless` — PPO converges
- `play.py --task Isaac-Cartpole-Direct-v0 --num_envs 16` — viewport renders trained policy

This is the **green-light state**. Phase 1 complete.

### 9. WSL2 + ROS 2 Humble setup (Phase 3 prep)

The UR3e + RG2 robot is defined as a chain of xacro macros under
`ur3e_rl_ws/src/ur3e_gazebo_sim/urdf/`. xacro is a ROS-only tool, so we need
Linux + ROS 2 Humble to flatten it into a plain URDF before Isaac Sim's
URDF importer can read it. The conversion is a one-shot operation — we don't
intend to *run* anything in WSL beyond xacro and a small mesh-prep helper.

**WSL2 install** (PowerShell, admin):
```powershell
wsl --install -d Ubuntu-22.04
```
After reboot + first launch (set Linux username/password), confirmed GPU
passthrough works inside WSL — `nvidia-smi` reports driver 580.88, RTX 4070 Ti
SUPER with full 16 GB visible, CUDA 13 stack ready. GPU passthrough isn't
strictly needed for URDF conversion, but it's nice to have for future
experiments.

**ROS 2 Humble install** (inside Ubuntu):
Standard install-from-apt flow per [ROS 2 docs](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html):
locale → universe repo → apt key → `sudo apt install ros-humble-desktop ros-dev-tools ros-humble-xacro`.
Auto-source `/opt/ros/humble/setup.bash` from `~/.bashrc`.

Then `sudo apt install ros-humble-ur-description` for the UR robot models
(installs under `/opt/ros/humble/share/ur_description/`).

**Bridge to repo**: Symlinked the Windows repo into WSL home for short paths
(the actual repo path has spaces which are painful in shell):
```bash
ln -s "/mnt/c/Users/sebas/Github/41118 Artificial Intelligence in Robotics/HoloAssist-AI" ~/holoassist
```

**Build local ROS workspace** (so xacro can resolve `$(find ur3e_gazebo_sim)`):
```bash
cd ~/holoassist/ur3e_rl_ws
colcon build --packages-up-to ur3e_gazebo_sim
source install/setup.bash
```

Three packages get built: `ur3e_rl_env`, `ur3e_safety_layer`, `ur3e_gazebo_sim`
— in dependency order. The first two are pulled in because
`ur3e_gazebo_sim/package.xml` lists them as `exec_depend`s; colcon-cmake
insists on building them before the dependent package even though it's a
runtime-only dep.

**Gotcha — broken CMakeLists.txt**: `ur3e_gazebo_sim/CMakeLists.txt`
referenced `scripts/pointcloud_cube_detector.py` in its `install(PROGRAMS ...)`
block, but that `.py` file no longer exists in the repo (only the binary
`pointcloud_cube_detector` does). Build failed with:
```
CMake Error: file INSTALL cannot find "scripts/pointcloud_cube_detector.py"
```
**Fix**: removed the dangling line from CMakeLists.txt. Real change to tracked
source — should be committed.

After the fix: `ros2 pkg prefix ur3e_gazebo_sim` returns
`~/holoassist/ur3e_rl_ws/install/ur3e_gazebo_sim`. Xacro can now find the
package, and we're ready to flatten the top-level xacro
(`ur3e_rg2_benchtop.urdf.xacro`) into a plain URDF.

**Note on `--symlink-install`**: Tried this flag first to avoid copying files
into `install/`. It failed on `/mnt/c/` because Windows NTFS doesn't honour
Linux symlink semantics the way colcon expects. Dropped the flag — normal
copy install works fine for a one-shot build.

### 10. URDF → USD pipeline (Phase 3 completion)

After getting the ROS toolchain ready in step 9, an indexing pass over the
repo revealed **two** UR3e+RG2 xacros — the simple `ur3e_rg2_benchtop.urdf.xacro`
(fixed gripper, used by the RL stack) and the articulated
`ros2_ws/src/ur_onrobot/ur_onrobot_description/urdf/ur_onrobot.urdf.xacro`
(used by MoveIt). Chose the articulated one to preserve gripper modelling
options in Phase 4+ — even if we initially train without gripper control,
having proper joints in the USD beats baking in a hack.

This shifted the source-of-truth ROS workspace from `ur3e_rl_ws/` to
`ros2_ws/`. Built the description packages:
```bash
sudo apt install -y ros-humble-ur-client-library ros-humble-ur-robot-driver
cd ~/holoassist/ros2_ws
colcon build --packages-up-to ur_onrobot_description
source install/setup.bash
```
The two apt installs satisfy `$(find ur_client_library)` and
`$(find ur_robot_driver)` resolution in the xacro. They're only used for
real-hardware driver paths and never run in Isaac Sim; we install them anyway
because it's standard ROS hygiene and they're small.

**Flatten** (WSL):
```bash
mkdir -p ~/holoassist/isaac_rl/assets/urdf
xacro ~/holoassist/ros2_ws/src/ur_onrobot/ur_onrobot_description/urdf/ur_onrobot.urdf.xacro \
  ur_type:=ur3e onrobot_type:=rg2 base_yaw_rad:=3.14159 \
  -o ~/holoassist/isaac_rl/assets/urdf/ur_onrobot.urdf
```
Output: 958-line URDF referencing 22 unique mesh files via `package://` URIs
across two packages (`ur_description` for the 14 UR3e meshes,
`onrobot_description` for the 8 RG2 STLs).

**Mesh mirror** (also WSL): copied `/opt/ros/humble/share/ur_description/meshes/ur3e/`
and `~/holoassist/ros2_ws/install/onrobot_description/share/onrobot_description/meshes/rg2/`
into `isaac_rl/assets/meshes/{ur_description,onrobot_description}/`. 13 MB total.

**`prepare_urdf.py`** does two transforms on the raw flatten:
1. Rewrites every `package://ur_description/meshes/...` →
   `../meshes/ur_description/...` (and same for `onrobot_description`).
   Isaac Sim's URDF importer can't resolve `package://`; relative paths work.
2. **Strips `<mimic>` tags entirely** — see "Mimic chain workaround" below.
Output: `ur_onrobot_prepared.urdf`, fully self-contained inside `isaac_rl/assets/`.

**Structural validation**: `check_urdf` (from `liburdfdom-tools`) confirms the
URDF parses cleanly — 26 links, 25 joints. Custom `inspect_urdf.py` walks the
tree and prints joint types/limits/mimic info — helpful for catching issues
before Isaac Sim sees them.

**Isaac Sim URDF Importer settings** (File → Import →
`ur_onrobot_prepared.urdf`):

| Section | Setting | Value | Why |
|---|---|---|---|
| Model | Referenced Model | ⦿ | Keeps imported asset as external USD that scenes can reference, not dumped prims |
| Model | USD Output | `isaac_rl/assets/usd/` | Where the asset lands |
| Links | Static Base | ⦿ | `world` link is fixed to ground |
| Links | Default Density | `1000` kg/m³ | Fallback for 2 `cable_connector` links missing `<inertial>` |
| Joints & Drives | Ignore Mimic | ❌ | Moot — we already stripped `<mimic>` in `prepare_urdf.py` |
| Joints & Drives | Joint Configuration | Natural Frequency | Tutorial-recommended formulation |
| Joints & Drives | Drive Type | Force | Default |
| Per-joint | Natural Frequency | `50.0` (overrides default 25.0) | Stiffer drive — settles in ~30 ms, fits 10 Hz RL control step |
| Per-joint | Damping Ratio | `1.0` (overrides default 0.005) | Critical damping, no oscillation |
| Colliders | Collision From Visuals | ❌ | Explicit collision STLs present in URDF |
| Colliders | Collider Type | Convex Hull | Sufficient for parallel-jaw grasping; faster than decomposition |
| Colliders | Allow Self-Collision | ❌ | RG2 fingers touch when closing — would trip false positives |

Output: `isaac_rl/assets/usd/ur_onrobot_prepared/` — a layered USD asset
(separate USDs for base geometry, physics, robot articulation, sensors).
Entry point for Isaac Lab Python: `ur_onrobot_prepared.usd`.

**Mimic chain workaround**: The RG2 URDF uses a two-level mimic chain
(`finger_width → finger_joint → 5 other joints`). Isaac Sim 5.1's URDF
importer mishandles multi-level mimics in two ways:
1. The `Ignore Mimic` checkbox does NOT fully strip mimic info — it just
   skips multipliers but leaves joints typed as mimic.
2. The mimic'd joints get created without finite limits in the output USD,
   which PhysX then rejects with errors like
   `the revolute joint needs a finite limit set to be used by the mimic
   joint feature`. The joints become uncontrolled and spin freely.

**Fix**: removed `<mimic>` tags from the URDF in `prepare_urdf.py`. The 6
linkage joints become plain revolute with their finite URDF limits intact.
We replicate the gripper's mechanical coupling in Python — `open_gripper()` /
`close_gripper()` helpers set all 7 joints together. Less elegant than relying
on PhysX joint coupling but reliable.

**`robot_test_v0.py`** (Isaac Lab smoke test) — minimal scene with
ground + dome light + robot. Verifies:
- USD loads as an Isaac Lab `Articulation`
- All 13 actuated joints readable via the API
- Drives hold the parked pose (UR3e `HOME_JOINTS = [0, -π/2, 0, -π/2, 0, 0]`,
  gripper closed via 7 explicit joint targets)
- 10 seconds of physics, no flopping, no oscillation, no joint drift

Discovered two more Isaac Lab idioms during debugging:

- **Drive targets aren't seeded from `init_state`.** `ArticulationCfg.InitialStateCfg.joint_pos`
  sets the *initial joint positions* on reset, but drive targets default to
  zero. Without explicitly calling
  `robot.set_joint_position_target(robot.data.default_joint_pos)` after
  `sim.reset()`, drives immediately pull joints to zero — arm settles in the
  all-zeros horizontal pose instead of the parked one. Phase 4's env will
  set targets every step anyway (that's the action), so this only matters for
  the smoke test's initial seeding.

- **Init joint positions must be *strictly inside* limits.** Setting
  `finger_joint = 0.785398` rad fails validation when the URDF upper limit
  is also `0.785398` (`pos not in [..., upper]` uses strict `<`). Back off
  by ~0.005 rad when at a limit. Visually identical to "fully closed".

Phase 3 complete. Asset is ready for Phase 4 env work.

### 11. Phase 4 — reach env port (DirectRLEnv)

Ported the legacy reach env from `ur3e_rl_ws/` into a fresh `DirectRLEnv` at
`tasks/direct/reach/`, then refactored into a **strategy-module split** so
observation / reward / action implementations can be swapped without touching
the env class. Final layout:

```
holoassist_tasks/.../tasks/direct/reach/
├── __init__.py                          gym.register("Template-Holoassist-Reach-Direct-v0")
├── reach_env.py                         HoloassistReachEnv + 4 thin delegators (~180 lines)
├── reach_env_cfg.py                     HoloassistReachEnvCfg + UR_ONROBOT_CFG (~240 lines)
├── observations/ground_truth_12d.py     build(env) -> {"policy": (num_envs, 12)}
├── rewards/dense_reach.py               compute(env) -> (num_envs,)
├── actions/joint_delta.py               process(env, action) + apply(env)
└── agents/rsl_rl_ppo_cfg.py             PPORunnerCfg
```

The env class only contains lifecycle code (`__init__`, `_setup_scene`,
`_reset_idx`, `_get_dones`) + four 1-line delegators. To A/B test a new
reward or obs shape, drop a new module in the matching subpackage and edit
one import line in `reach_env.py` (or subclass the env for permanent
side-by-side comparison via a second `gym.register`).

**Sequencing of the 9 sub-mappings** (each one separately validated before
moving on):

1. **4a — scaffold rename + skeleton.** Renamed `tasks/direct/holoassist_tasks/` (the
   `--new` generator's cartpole template) → `tasks/direct/_template_cartpole/`
   to avoid name collision. Created empty `tasks/direct/reach/` with 5 stub
   files that register the gym ID but raise `NotImplementedError` if instantiated.
   Verified `import_packages()` auto-discovery picks up both task folders.
2. **4b — constants.py → reach_env_cfg.py.** Hoisted legacy workspace bounds,
   action scale, joint limits, etc. into `@configclass` fields. Renamed
   `JOINT_DELTA_ACTION_SCALE_RAD` → `action_scale_rad` since we committed to
   true joint-delta semantics. Discovered the action-scale-vs-control-rate
   gotcha (see troubleshooting): legacy's 0.24 rad/step at 10 Hz = 2.4 rad/s
   max joint speed, but at our 30 Hz control rate it would be 7.2 rad/s
   (above hardware limit). Set to **0.08** rad/step instead.
3. **4c — observation builder.** `ros_interface.build_observation` (numpy, 13-D
   with broken `grasped`/`gripper_state` fields) → `observations/ground_truth_12d.py`
   (torch, vectorised). Slimmed to **12-D: 6 joint_pos + 3 EE_pos + 3 target_pos**.
   Dropped noisy/broken legacy fields. RSL-RL's `actor_obs_normalization=True`
   handles running mean/std normalisation, so no manual normalisation here.
4. **4d — reward.** `reward.py::compute_reward` (numpy, mixed reach/grasp/place
   terms) → `rewards/dense_reach.py::compute()` (torch, vectorised, reach-only).
   Dropped grasp stipend (`+5.0`), transport gradient (`-0.5 * dist_to_bin`),
   and collision penalty (`-0.5`) — all pick-place-specific, restored in
   Phase 4b. Kept dense reach gradient, success bonus, time penalty, action
   penalty at legacy scales. Q2 chose option C (**terminate on success**), so
   the +10.0 bonus is effectively one-shot.
5. **4e — safety_checker.** Inlined the slew-limit + joint-clamp into
   `actions/joint_delta.py::process()`. 101-line `SafetyChecker` class
   collapses to ~5 lines of `torch.clip`. Joint limits no longer in cfg —
   read from `robot.data.joint_pos_limits` (single source of truth: URDF →
   USD → PhysX → API). Refactored env into the strategy-module split
   described above.
6. **4f — env structure.** `HoloassistReachEnv.__init__` + `_setup_scene` +
   `_reset_idx` + `_get_dones`. ~150 lines total. Key choices:
   - Robot mounted via `InitialStateCfg.pos=(0, 0, 1.0)` (the URDF's z=1.0
     offset got merged out by the URDF importer's "Merge Fixed Joints" pass —
     have to put it back in the cfg).
   - **EE proxy: `left_inner_finger`** (the URDF's `gripper_tcp` link got
     merged into `onrobot_base_link` during import; `left_inner_finger` is
     the closest available body to the actual grasp point).
   - Targets randomised per env in `_reset_idx` using torch tensors on the
     device; world-frame positions = local random + `scene.env_origins[i]`.
   - Optional scene elements (ground plane, table pedestal, target marker)
     all gated by `cfg.add_*` bool flags — toggleable for ablation runs.
7. **4g — PPO tuning.** Replaced cartpole template defaults
   (`[32, 32]` net, `entropy_coef=0.005`, `actor_obs_normalization=False`)
   with manipulation-appropriate values: `[256, 128, 64]` actor + critic,
   `entropy_coef=0.01`, **`actor_obs_normalization=True`** (Q5 decision —
   RSL-RL handles running mean/std for our mixed-scale obs).
8. **4h — sanity test.** Wrote `isaac_rl/scripts/reach_test_v0.py` — drives
   the env with 100 random actions, asserts no NaN/inf, reports per-step
   reward + termination counts. Found a print-buffering gotcha in PowerShell
   that hid script output (the env was actually running fine; output wasn't
   reaching the file). Fix: `print(..., flush=True)` consistently.
9. **4i — smoke train.** 64 envs × 50 iters headless, ~48 seconds. All RSL-RL
   metrics produce cleanly: value loss decreasing, surrogate loss small,
   entropy decreasing slightly, mean reward inching up. Checkpoint saved at
   `IsaacLab/logs/rsl_rl/holoassist_reach_direct/<timestamp>/model_49.pt`.

After 4i: discovered episodes were terminating at ~10 steps (out of max 200)
because the original "parked" home pose put the EE only 0.19 m above the
`min_ee_z=0.5` termination threshold. Bumped the threshold (`min_ee_clearance_below_base_m`
0.5 → 0.9 → `min_ee_z = 0.1`) and changed the home pose three times:

  - parked `[0, -π/2, 0, -π/2, 0, 0]` (legacy HOME_JOINTS) — EE z ≈ 0.69 m, too low
  - ready `[0, -π/2, π/2, -π/2, -π/2, 0]` (UR-Lab UR10e style) — EE z ≈ 1.1 m, much better
  - **vertical zero `[0, 0, 0, 0, 0, 0]`** (UR3e factory zero) — EE z ≈ 1.55 m, well clear of any threshold ← landed here

Scene visual layout also evolved through three iterations: thin table plate
under robot → thin plate offset forward → **full pedestal (0.7×0.7×1.0 m
solid block)** under the robot with workspace on top. Dome light bumped 2000 →
3500 + added a `DistantLightCfg` sun for better shading.

Final reward shape gains a **5th term**: soft below-base-plane penalty
(`-10 * max(0, threshold - ee_z)` where `threshold = robot_base_height - 0.02`).
Keeps the policy from learning to dip the EE through/under the workspace
surface. Continuous gradient — doesn't terminate, just penalises depth.

Phase 4 complete. Env loads, resets, steps, trains end-to-end.

### 12. Phase 5 — initial training run (in progress)

First convergence-scale run: **4096 envs × 500 iters @ 30 Hz control rate**,
~20 minutes wall-clock on the RTX 4070 Ti SUPER. Throughput: ~80,000 sim
steps/sec.

**Headline metrics** (from `Train/` TensorBoard cards):

| Metric | At iter 0 | At iter 499 |
|---|---|---|
| `mean_reward` | ~-100 | ~-20 |
| `mean_episode_length` | 200 (timeout) | ~110 |
| Total env steps | 0 | ~49 M |

The policy is **partially converged**: reaches the target ~60-70% of the
time, takes ~110 control steps (3.7 s wall) when it does. Reward improvement
is monotone and clean — no instability, no crashes, no NaN. Several
checkpoints saved (`model_0.pt` … `model_499.pt`, every 50 iters).

Eval via `play.py` works (with a path-resolution gotcha — see troubleshooting
"`play.py --checkpoint` requires a `.pt` file path"). Visualises the trained
policy controlling 4 robots in GUI mode.

Next: train longer (target 1500-2000 iters total) and/or tune `entropy_coef`
down from 0.01 → 0.003 to encourage exploitation in the latter half of
training. The current entropy curve climbs over training, suggesting the
policy is still exploring multiple near-optimal strategies rather than
committing — fine for reach but worth tuning before Phase 6.

---

## Quick commands (current toolchain)

All `isaaclab.bat` commands are run from `C:\Users\sebas\Github\IsaacLab`.
For brevity below, `<HOLOASSIST>` = `C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI`.

### Verify versions
```powershell
cd C:\Users\sebas\Github\IsaacLab
.\_isaac_sim\python.bat -m pip list | Select-String -Pattern "isaaclab|torch|rsl|tensordict|h5py|gymnasium|onnx"
```
Expected: `tensordict 0.7.2`, `h5py 3.11.0`, `torch 2.7.0+cu128`, `rsl-rl-lib 5.0.1`,
`isaaclab 0.54.3` + 5 more editable.

### List registered Holoassist tasks
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\scripts\robot_test_v0.py" --headless
# the script prints joint_names/body_names on startup; useful for verifying registration
```

### Robot USD smoke test (no env, just spawn + step physics)
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\scripts\robot_test_v0.py" --headless
```
Confirms `ur_onrobot_prepared.usd` loads as an Isaac Lab Articulation. Useful
after re-importing the URDF or changing the USD assets.

### Reach env sanity test (random agent)
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\scripts\reach_test_v0.py" `
    --headless --num_envs 16 --num_steps 100
```
100 random actions × 16 envs. Reports per-step shapes, NaN counts,
termination counts, reward range. Useful after cfg changes to confirm the
env still steps cleanly. Add `--no_marker` to disable target marker.

### Smoke train (fast, low-fidelity verification of the training loop)
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 64 --max_iterations 50 --headless
```
~1 minute. Just confirms PPO collects rollouts, computes advantages, updates
networks, saves checkpoint. Not enough for convergence.

### Convergence train (Phase 5 target)
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 4096 --max_iterations 1500 --headless
```
~30-45 minutes on RTX 4070 Ti SUPER. Expect mean_reward to climb from
~-100 → ~-5 ish (if reward shape is healthy). Drop `--max_iterations` to 500
for a quicker first-pass; bump to 3000+ for sim-to-real-ready policy.

### Play a trained checkpoint (GUI, watch the policy)
```powershell
# Find the latest run
$run = (Get-ChildItem "C:\Users\sebas\Github\IsaacLab\logs\rsl_rl\holoassist_reach_direct" `
    | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$ckpt = (Get-ChildItem "$run\model_*.pt" `
    | Sort-Object { [int]($_.BaseName -replace 'model_','') } -Descending | Select-Object -First 1).FullName
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\holoassist_tasks\scripts\rsl_rl\play.py" `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 4 --checkpoint $ckpt
```
**IMPORTANT**: `--checkpoint` is a full path to a `.pt` file, NOT a run
directory (see troubleshooting). The PowerShell snippet above auto-picks the
latest run + highest-numbered checkpoint.

### Launch TensorBoard (separate terminal, leave running)
```powershell
cd C:\Users\sebas\Github\IsaacLab
.\_isaac_sim\python.bat -m tensorboard.main --logdir logs\rsl_rl\holoassist_reach_direct
```
Open `http://localhost:6006`. Key cards under **Train/**: `mean_reward` (should
trend up), `mean_episode_length` (should grow then plateau as policy succeeds
faster). Loss cards (under **Loss/**) show training health. Can run live
during training — refreshes on its own.

### Re-prepare URDF (if Phase 3 source files change)
```bash
# In WSL
wsl -d Ubuntu-22.04 -- bash -c "source /opt/ros/humble/setup.bash && \
    source ~/holoassist/ros2_ws/install/setup.bash && \
    xacro ~/holoassist/ros2_ws/src/ur_onrobot/ur_onrobot_description/urdf/ur_onrobot.urdf.xacro \
        ur_type:=ur3e onrobot_type:=rg2 base_yaw_rad:=3.14159 \
        -o ~/holoassist/isaac_rl/assets/urdf/ur_onrobot.urdf && \
    python3 ~/holoassist/isaac_rl/scripts/prepare_urdf.py"
```
Then re-run the Isaac Sim URDF Importer GUI step (see ISAAC_SIM_PLAN.md Phase 3).

### Inspect prepared URDF structure
```bash
wsl -d Ubuntu-22.04 -- bash -c "python3 ~/holoassist/isaac_rl/scripts/inspect_urdf.py"
```
Prints link counts, mass distribution, joint types/limits, mimic chains,
`<gazebo>` and `<ros2_control>` block counts.

### Validate URDF topology
```bash
wsl -d Ubuntu-22.04 -- bash -c "check_urdf ~/holoassist/isaac_rl/assets/urdf/ur_onrobot_prepared.urdf"
```
Quick "does the URDF parse + tree connectivity" check from `liburdfdom-tools`.

### Pip-reinstall the editable holoassist_tasks package (after structural changes)
```powershell
cd C:\Users\sebas\Github\IsaacLab
.\_isaac_sim\python.bat -m pip install -e `
    "<HOLOASSIST>\isaac_rl\holoassist_tasks\source\holoassist_tasks"
```
Needed if you rename a task folder, add a new task, or change `setup.py`.
Not needed for editing Python inside the existing package (editable mode
follows the source).

### Resume training from a checkpoint
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 4096 --max_iterations 1500 --headless `
    --resume --checkpoint "<full .pt path>"
```
Continues from where the saved policy left off. Useful for staged training
or recovering from interrupted runs.

### Export trained policy to ONNX (Phase 6)
```powershell
.\isaaclab.bat -p "<HOLOASSIST>\isaac_rl\holoassist_tasks\scripts\rsl_rl\play.py" `
    --task Template-Holoassist-Reach-Direct-v0 --num_envs 16 --headless `
    --checkpoint "<full .pt path>" --export_io
```
Produces `policy.onnx` + `policy.pt` (TorchScript) under the run's
`exported/` subfolder. The ONNX file is what Phase 7 loads in the ROS node.

---

## Troubleshooting

### "Python was not found" / "Unable to find any Python executable"
→ The `_isaac_sim` junction is missing. Recreate it (see step 4 above).

### `[Fatal] rtx.scenedb.plugin.dll!carbOnPluginStartup`
→ Wrong NVIDIA driver. Downgrade to 580.88 (or any pre-595 production branch). See step 2.

### `Windows fatal exception: access violation` in `tensordict._C.pyd`
→ tensordict was upgraded above 0.7.x. Re-pin (see Critical version pins above).

### `ImportError: DLL load failed while importing _errors` (h5py)
→ h5py was upgraded above 3.11. Re-pin.

### Viewport timeout during `play.py` first launch
→ Shader cache still compiling. Run the command 2-3 times — each run extends the cache. Subsequent runs are fast.

### `tensorboard` command not found
→ TensorBoard is in Isaac Sim's Python, not on PATH. Run it via `.\_isaac_sim\python.bat -m tensorboard.main ...`.

### After re-running `isaaclab.bat --install ...`, training crashes again
→ Pip may have re-upgraded `tensordict` or `h5py`. Re-apply the pins.

### "Recursive unloadAllPlugins() detected!" in shutdown log
→ Benign. Just teardown noise after the script exits — not a crash indicator.

### `play.py` runs for minutes then crashes after I touched render settings
→ Known USD race condition. Modifying the scene graph (toggling DLSS, restoring render
defaults, creating / copying prims like light rigs) while physics is actively stepping can
invalidate iterators in the C++ render layer, producing `Iterator past-the-end` warnings
followed by an access violation. **Don't mutate the scene or change render settings while
the sim is running.** To change render quality: `Ctrl+C` first, then relaunch with the
new setting. For training, `--headless` sidesteps this entirely.

### "h5py is running against HDF5 1.14.6 when it was built against 1.14.2"
→ Benign warning. h5py 3.11.0's bundled HDF5 (1.14.2) is slightly older than what Isaac
Sim's environment exposes (1.14.6). It loads fine. Suppress with
`export HDF5_DISABLE_VERSION_CHECK=2` if it bothers you.

### `colcon build` fails: "Failed to find ... install/<pkg>/share/<pkg>/package.sh"
→ colcon-cmake checks all in-workspace `exec_depend`s, not just build deps. The fix is
`colcon build --packages-up-to <target>` instead of `--packages-select <target>` — the
former builds the whole dep chain in order.

### `colcon build` fails with `--symlink-install` on `/mnt/c/`
→ NTFS doesn't honour Linux symlink semantics the way colcon expects on the Windows
filesystem. Drop the flag and use a normal copy install.

### `CMake Error: file INSTALL cannot find "scripts/foo.py"` during colcon build
→ The package's `CMakeLists.txt` lists a script that's been deleted from the source
tree. Either restore the file from git or remove the dangling line from
`install(PROGRAMS ...)`. (Phase 3 prep: this happened with
`ur3e_gazebo_sim/scripts/pointcloud_cube_detector.py`.)

### Isaac Sim file dialog can't browse into `C:\Users\...` — red padlock on Users folder
→ The padlock is **cosmetic**, not an ACL. Omniverse file dialogs visually flag
known "system" folders (`Program Files`, `Users`) but navigation still works on
double-click. If actually blocked, verify with PowerShell that `sebas` has
FullControl on the user profile and that the `kit.exe` process owner is
`DESKTOP-VAP2BJE\sebas` (not an admin token). Workaround if all else fails:
copy URDF + meshes to `C:\holoassist_import\` and import from there.

### Isaac Sim file dialog interprets pasted Windows path as Omniverse server
→ The address bar defaults to `omniverse://`. Pasting `C:\...` over it gets
re-parsed as `omniverse://c` and triggers a "Login Required" dialog. Clear the
bar first, then paste; or navigate via the My Computer tree on the left.
Forward slashes (`C:/Users/...`) are more reliable than backslashes.

### `[Error] Usd Physics: the revolute joint ... needs a finite limit set to be used by the mimic joint feature`
→ Isaac Sim 5.1's URDF importer mishandles multi-level mimic chains —
creates joints typed as mimic but loses their finite limits. The `Ignore Mimic`
checkbox doesn't fully strip mimic info. **Fix**: remove `<mimic>` tags from
the URDF source (Phase 3 does this in `prepare_urdf.py`); the joints then
import as plain revolute with limits intact. Replicate the mimic relationship
in Python instead.

### `ValueError: The following joints have default positions out of the limits`
→ Isaac Lab does **strict** containment on init joint positions (`pos < upper`,
not `pos <= upper`). A value exactly at the URDF's limit fails validation. Back
off the init value by ~0.005 rad. Visually identical to the limit.

### Robot loads but arm flops to all-zeros pose under gravity instead of staying at init
→ `ArticulationCfg.InitialStateCfg.joint_pos` sets initial positions but NOT
drive targets — targets default to 0. After `sim.reset()`, call
`robot.set_joint_position_target(robot.data.default_joint_pos)` and
`scene.write_data_to_sim()` to seed targets to match init positions. In an RL
env this is moot (policy sets new targets every step), but matters for
init/smoke tests.

### Can't interact with viewport after a script-launched Isaac Sim finishes
→ Script-launched Isaac Sim is meant to be ephemeral (open → run → close).
Sometimes `simulation_app.close()` leaves an orphaned post-simulation window
that won't respond to gizmos or selection. **Just close it** (X or Task
Manager kill `kit.exe`) and use a separately-launched Isaac Sim for
interactive editing.

### `Unresolved reference prim path ... visuals/world` / `visuals/finger_width_mock_link` on USD load
→ Benign cosmetic warnings. Frame-only links (`world`, `finger_width_mock_link`,
etc.) have no geometry, so the importer creates an empty visuals reference for
them. No impact on physics or rendering.

### `ValueError: Prim path '{ENV_REGEX_NS}/Robot' is not global. It must start with '/'.`
→ The `{ENV_REGEX_NS}` placeholder is only expanded when the cfg is
auto-discovered by `InteractiveSceneCfg` as a class attribute. When you
instantiate `Articulation(self.cfg.robot_cfg)` directly in `_setup_scene`,
use the explicit regex pattern instead: `prim_path="/World/envs/env_.*/Robot"`.
The cartpole template uses this form too.

### `ValueError: Not all regular expressions are matched! gripper_tcp: []`
→ The URDF's `gripper_tcp` (and any other frame-only link with no inertia)
gets merged into its parent during Isaac's "Merge Fixed Joints" import pass.
Pick an available body as the EE proxy instead — `left_inner_finger` is
closest to the actual grasp point. Visible body list: `world`, `base_link_inertia`,
`shoulder_link`, `upper_arm_link`, `forearm_link`, `wrist_1_link` through
`wrist_3_link`, `onrobot_base_link`, `left/right_outer_knuckle`, `left/right_inner_knuckle`,
`left/right_inner_finger`, `finger_width_mock_link`. Cable connectors merged.

### Robot mounted at ground level instead of URDF z=1.0
→ The URDF's `<origin xyz="0 0 1"/>` macro offset gets merged out during
the URDF importer's "Merge Fixed Joints" pass — the world-link transform
collapses into the base_link. The resulting USD's articulation root is at
its local origin. To put the robot at z=1.0 again, set
`ArticulationCfg.init_state.pos=(0.0, 0.0, 1.0)` explicitly.

### Action scale exceeds hardware joint velocity limit
→ Easy to miss: legacy used `JOINT_DELTA_ACTION_SCALE_RAD = 0.24` at
`control_dt = 0.1 s` (10 Hz). At our Isaac control rate of 30 Hz
(`decimation=4, sim.dt=1/120`), the same 0.24 rad/step would be **7.2 rad/s**
— well above UR3e's ~3 rad/s hardware ceiling. Set
`cfg.action_scale_rad = 0.08` instead → 2.4 rad/s, matching legacy max.
General rule: `action_scale * control_rate ≤ 75% of hardware velocity limit`.

### Arm settles in all-zeros pose instead of init/home pose
→ `ArticulationCfg.InitialStateCfg.joint_pos` sets the initial *positions*
but does NOT seed the *drive targets*. The first physics step has
`set_joint_position_target = 0` (default), so position-controlled drives pull
the arm to all-zeros. In an env's `_reset_idx`, call
`robot.set_joint_position_target(default_joint_pos, env_ids=env_ids)` right
after `write_joint_state_to_sim` to seed the targets. (In an RL env, the
policy writes new targets every step via `_pre_physics_step`, so this only
matters for init/reset.)

### `ValueError: The following joints have default positions out of the limits`
→ Isaac Lab does **strict** containment on init joint positions (`pos < upper`,
not `pos <= upper`). A value exactly at the URDF's limit fails validation.
Pad init values inward from the limit by ~0.005 rad. (We hit this with the
RG2 finger linkage where the "closed" config landed exactly at the URDF
upper limit `0.785398`.)

### Python script output is missing or truncated when run via Isaac Lab launcher
→ Native command output through PowerShell can be block-buffered when stdout
is redirected to a file. Fix in the Python script: pass `flush=True` to
every `print()` call. Or use a small helper:
```python
def log(msg):
    print(msg, flush=True)
    sys.stdout.flush()
```
Without this, your script may appear to "hang" or "exit early" when it's
actually running fine.

### Multi-env episodes terminate after just ~10 steps
→ Symptom in TensorBoard: `Train/mean_episode_length` drops fast then
plateaus far below the max. Usually means the EE-too-low termination is
firing aggressively. Two fixes:
1. Loosen the threshold: bump `cfg.min_ee_clearance_below_base_m` (the
   distance below base where termination fires) up — e.g. `0.5 → 0.9`.
2. Change the home pose so the EE starts higher. The legacy parked pose
   `[0, -π/2, 0, -π/2, 0, 0]` puts EE at z ≈ 0.69 m (right at the threshold).
   UR3e zero pose `[0, 0, 0, 0, 0, 0]` puts EE at z ≈ 1.55 m (plenty of room).

### `play.py --checkpoint` PermissionError trying to load from `%TEMP%`
→ `--checkpoint` expects the **full path to a specific `.pt` file**, not the
run directory. If you pass a folder path, Hydra interprets the folder's
basename as a relative checkpoint name and prepends `%TEMP%` — producing
the cryptic permission error. Use:
```powershell
--checkpoint "C:\...\logs\rsl_rl\holoassist_reach_direct\<timestamp>\model_499.pt"
```
not
```powershell
--checkpoint "C:\...\logs\rsl_rl\holoassist_reach_direct\<timestamp>"
```

### `FabricManager::initializePointInstancer mismatched prototypes` warning
→ Cosmetic. Issued by `VisualizationMarkers` during scene init when the
marker prototypes are first instanced for N envs. Doesn't crash the sim;
markers render correctly.

---

## Lessons learned

1. **Don't trust "latest" pip wheels on Windows + Isaac Sim's Python.** The bundled Python has a non-standard DLL search context, and many native packages (`tensordict`, `h5py`) ship wheels that break in that context. Pin to known-good versions.
2. **Driver version matters more than you'd expect.** NVIDIA validates Isaac Sim against specific driver versions. Newer ≠ better. Check the [Isaac Sim Requirements page](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) before grabbing the latest driver.
3. **The `_isaac_sim` junction is mandatory** on Windows for the binary-install path. The env var alone isn't enough.
4. **Headless ≠ GUI** in terms of what extensions get loaded. Some bugs (like the h5py one) only show up when the viewport experience boots.
5. **Pip is idempotent.** If an install hangs and you Ctrl+C, just run it again. It picks up where it left off.
6. **Crashreports without `[Fatal]` lines in the log are usually timeouts, not crashes.** Look for the actual error markers before assuming the worst.
7. **Don't poke the viewport while a sim is running.** Toggling DLSS, restoring render defaults, or creating/copying prims through the GUI while physics is stepping triggers a USD iterator invalidation that crashes the C++ render layer. Let `play.py` just play. Use `--headless` for training to sidestep this entirely.
8. **Strip URDF features rather than work around importer bugs.** The Isaac Sim 5.1 URDF importer drops multi-level mimic chains and ignores its own `Ignore Mimic` checkbox. Removing `<mimic>` tags from the URDF source via `prepare_urdf.py` was less work than fighting the importer's broken handling, and yielded a cleaner USD. Same philosophy applies to over-declared `exec_depend`s, dangling `install(PROGRAMS ...)` references, etc. — fix at the source, not in workarounds downstream.
9. **Isaac Sim file dialogs are not Windows file dialogs.** They have their own quirks — Omniverse URL prefix on the address bar, cosmetic "system folder" padlocks, less-tolerant path parsing. Navigate via the tree pane when in doubt; clear the address bar before pasting paths; prefer forward slashes.
10. **`InitialStateCfg.joint_pos` ≠ drive targets.** Isaac Lab sets initial joint positions on reset but defaults drive targets to zero. Always seed `set_joint_position_target(default_joint_pos)` after reset (or, in an RL env, rely on the per-step action writing targets every iteration).
11. **Script-launched and interactive Isaac Sim are separate processes.** Don't try to use a script's ephemeral Isaac Sim window for interactive editing after the script exits — it's in a half-dead state. Use a fresh standalone Isaac Sim for any GUI work.
12. **Strategy-module split beats monolithic env class for iteration speed.** Pulling `_get_observations`, `_get_rewards`, `_pre_physics_step` out of the env class into `observations/`, `rewards/`, `actions/` sub-packages (each with a `compute(env)` or `build(env)` function) lets you A/B test new reward shapes / obs layouts by adding a new module + changing one import line. Lossy for tightly-coupled code (env class drops to ~180 lines), but the lifecycle methods (init/setup_scene/reset_idx/get_dones) genuinely don't share much with the stateless strategy functions.
13. **PowerShell + Python + Isaac Sim native launcher = output buffering surprises.** Always `print(..., flush=True)` from scripts that may be redirected. Saved us hours of "is the script hung?" confusion when in fact it was running fine but no output was reaching the file until process exit.
14. **The URDF importer is opinionated.** Two specific gotchas we hit: (a) frame-only links (no inertia) get merged out, so don't rely on them existing in the imported USD — pick a real body as the EE proxy. (b) The URDF's world→base transform also gets merged into the articulation root, dropping any z-offset baked into a `<xacro:ur_onrobot><origin xyz="..."/>`. Put it back via `ArticulationCfg.init_state.pos`.
15. **Drive targets are stateful, not snapshot-from-init.** `InitialStateCfg.joint_pos` populates the initial joint positions on reset but not the drive targets. The drives default to "target = 0" which immediately pulls the arm to all-zeros on the next physics step. Seed the targets explicitly in `_reset_idx`. (In a normal RL loop the policy overwrites them every step, so this only bites at init.)
16. **Action scale × control rate = effective velocity.** Easy to copy a per-step scale from a different sim and forget the dt changed. Always sanity-check: `max_velocity = action_scale / (sim.dt × decimation)` should match the real robot's safe velocity envelope.
17. **PPO on a 6-DOF arm with 12-D obs converges visibly within 500 iters at 4096 envs (~20 min on RTX 4070 Ti SUPER).** Reward improvement is monotone if reward shape is healthy and entropy_coef isn't too high. Higher quality (95%+ success) needs 1500-2000 iters total. Sim-to-real-ready needs 3000-8000 iters + domain randomisation.

---

## References

- [Isaac Sim 5.1 installation docs](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html)
- [Isaac Lab Quickstart](https://isaac-sim.github.io/IsaacLab/main/source/setup/quickstart.html)
- [NVIDIA forum: Isaac Sim 5.1 + 595 driver crash](https://forums.developer.nvidia.com/t/isaac-sim-5-1-gui-crash-access-violation-on-rtx-5070-ti-blackwell-fixed-by-driver-downgrade-to-591-74)
- [Isaac Lab compatibility matrix](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

---

## Changelog (append as work progresses)

- **2026-05-28** — Phase 1 complete. Cartpole training + playback verified. README created.
- **2026-05-28** — Added troubleshooting entries for mid-run GUI crashes (USD race when modifying scene during sim step) and benign h5py HDF5 version-skew warning.
- **2026-05-28** — Phase 2 complete. Generated `holoassist_tasks` project scaffold via `isaaclab.bat --new` (External, Direct, single-agent, rsl_rl). Pip installed editable into Isaac Sim's Python. Verified end-to-end by training the auto-generated `Template-Holoassist-Tasks-Direct-v0` task through project-local scripts. Note: generator copies cartpole as a template — to be replaced in Phase 4. Deleted nested `.git` repo inside `holoassist_tasks/` to avoid submodule conflict.
- **2026-05-29** — Phase 3 setup: WSL2 + Ubuntu 22.04 installed (GPU passthrough confirmed via `nvidia-smi`). ROS 2 Humble + `xacro` + `ur_description` installed via apt. Repo symlinked into WSL at `~/holoassist`. Built `ur3e_gazebo_sim` ROS package via `colcon build --packages-up-to`; fixed a dangling `install(PROGRAMS scripts/pointcloud_cube_detector.py ...)` line in its `CMakeLists.txt` (script was deleted from the source tree, but the line wasn't removed). Toolchain is now ready to flatten `ur3e_rg2_benchtop.urdf.xacro` → URDF.
- **2026-05-30** — Phase 3 complete. After repo-wide indexing surfaced two competing UR3e+RG2 xacros, picked the articulated `ur_onrobot.urdf.xacro` from `ros2_ws/src/ur_onrobot/` over the fixed-gripper benchtop variant — preserves Phase 4 gripper-modelling options. Installed `ros-humble-ur-client-library` + `ros-humble-ur-robot-driver` to satisfy xacro `$(find ...)` resolution, built `ur_onrobot_description` via colcon, flattened to URDF. Wrote `scripts/prepare_urdf.py` (mesh URI rewrite + `<mimic>` tag strip) and `scripts/inspect_urdf.py` (structural report). Re-imported into Isaac Sim with Static Base, Default Density 1000, Natural Freq 50, Damping Ratio 1.0, Convex Hull colliders — output `assets/usd/ur_onrobot_prepared/`. Wrote and ran `scripts/robot_test_v0.py` — Isaac Lab smoke test confirms USD loads as Articulation, all 13 joints readable, drives hold parked pose for 10 s with no oscillation. Discovered three Isaac Lab idioms: (a) drive targets aren't auto-seeded from `InitialStateCfg.joint_pos` — need explicit `set_joint_position_target` after `sim.reset()`; (b) init positions must be strictly inside joint limits (`pos < upper`, not `<=`); (c) two-level URDF mimic chains don't import — strip `<mimic>` at the URDF level instead. Asset ready for Phase 4.
- **2026-05-31** — Phase 4 complete + Phase 5 in progress. Ported the legacy reach env into a fresh `DirectRLEnv` at `tasks/direct/reach/`, refactored into a strategy-module split (`observations/`, `rewards/`, `actions/` sub-packages). 9 sub-mappings, each separately validated. New gotchas surfaced: prim_path needs `/World/envs/env_.*/Robot` literal (not `{ENV_REGEX_NS}/Robot`) when constructing Articulation() directly; `gripper_tcp` link merged out by URDF import → use `left_inner_finger` as EE proxy; URDF's z=1.0 base offset gets merged out → restore via `InitialStateCfg.pos=(0,0,1)`; action scale 0.24 at 30 Hz exceeds UR3e velocity limit (use 0.08); PowerShell stdout block-buffers Python prints (always `flush=True`). Initial home pose put EE only 0.19 m above the EE-too-low termination threshold → episodes terminated at ~10 steps; iterated through 3 home poses (parked → ready → vertical zero) and bumped the threshold from 0.5 → 0.1. Visual scene: thin plate → forward-offset plate → full 0.7×0.7×1.0 m pedestal. Added a 5th reward term (below-base-plane penalty, soft linear). Phase 5: 500 iters × 4096 envs @ ~80K steps/sec headless. Reward improvement -100 → -20 monotone over training. Episode length stable at ~110/200 (policy reaches target ~60-70% of the time). Checkpoint saved + visualised via `play.py`. Needs another 1000-1500 iters + entropy_coef tuning for full convergence.
