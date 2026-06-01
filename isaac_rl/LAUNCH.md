# Isaac Sim Launch Commands

Quick-reference commands for the HoloAssist Isaac Lab RL stack. Companion to the
root [LAUNCH.md](../LAUNCH.md) (ROS sim + clustering).

All commands run from **PowerShell on Windows**. Isaac Lab launcher is
`.\isaaclab.bat -p <script>` from the IsaacLab repo root.

## Paths

| Thing | Path |
|---|---|
| Isaac Lab repo | `C:\Users\sebas\Github\IsaacLab` |
| Project repo | `c:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI` |
| Train script | `<project>\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py` |
| Play script | `<project>\isaac_rl\holoassist_tasks\scripts\rsl_rl\play.py` |
| Training logs | `<IsaacLab>\logs\rsl_rl\holoassist_reach_direct\<timestamp>\` |

## Registered task IDs

| ID | Reward shape | When to use |
|---|---|---|
| `Template-Holoassist-Reach-Direct-v0` | 5 terms (dense_reach.py): reach, success, action, time, below_plane | Baseline |
| `Template-Holoassist-Reach-V1-Direct-v0` | 8 terms (dense_reach_v1.py): v0 plus down-incentive, action_rate, joint_vel | Smoother motion |

Registered in [reach/__init__.py](holoassist_tasks/source/holoassist_tasks/holoassist_tasks/tasks/direct/reach/__init__.py).

---

## 1. Train

### Smoke train (verify env builds + loop runs)

```powershell
cd C:\Users\sebas\Github\IsaacLab

.\isaaclab.bat -p "C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" `
  --task Template-Holoassist-Reach-V1-Direct-v0 `
  --num_envs 64 `
  --max_iterations 50 `
  --headless
```

Finishes in ~1 minute. Use this after any env edit before launching a full run.

### Full train (Phase 5)

```powershell
.\isaaclab.bat -p "C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI\isaac_rl\holoassist_tasks\scripts\rsl_rl\train.py" `
  --task Template-Holoassist-Reach-V1-Direct-v0 `
  --num_envs 4096 `
  --max_iterations 2000 `
  --headless
```

~30-35 min on a 4070 Ti SUPER. Drop `--headless` to watch the rollout (~5-10x slower).

### Common train flags

| Flag | Default | Purpose |
|---|---|---|
| `--task` | required | Gym ID (see table above) |
| `--num_envs` | 64 (cfg) | Parallel envs; sweet spot ~4096 on this GPU |
| `--max_iterations` | 200 (cfg) | PPO updates; ~1500-2000 for convergence |
| `--headless` | off | No GUI; required for full runs |
| `--seed` | -1 | RNG seed; -1 = random |
| `--resume` | off | Resume from `--load_run <name> --checkpoint <file>` |

PPO hyperparams (lr, entropy, network size, save_interval, ...) live in
[agents/rsl_rl_ppo_cfg.py](holoassist_tasks/source/holoassist_tasks/holoassist_tasks/tasks/direct/reach/agents/rsl_rl_ppo_cfg.py).

---

## 2. Switching strategy modules (reward / obs / action)

The env class delegates to swappable modules. Three ways to swap them:

### A. Edit the env's import (changes default behavior)

In [reach_env.py](holoassist_tasks/source/holoassist_tasks/holoassist_tasks/tasks/direct/reach/reach_env.py):

```python
from .actions      import joint_delta        as action_strategy
from .observations import ground_truth_12d   as obs_strategy
from .rewards      import dense_reach_v1     as reward_strategy
```

Repoint any `... as ..._strategy` line at a different module file in the same
subpackage. Required function signatures:

| Subpackage | Required function(s) |
|---|---|
| `observations/<name>.py` | `build(env) -> dict` |
| `rewards/<name>.py` | `compute(env) -> Tensor` |
| `actions/<name>.py` | `process(env, action)`, `apply(env)` |

### B. Subclass + override (per-gym-ID swap — recommended for A/B compares)

This is the v1 pattern in [__init__.py](holoassist_tasks/source/holoassist_tasks/holoassist_tasks/tasks/direct/reach/__init__.py):

```python
class HoloassistReachV1Env(HoloassistReachEnv):
    def _get_rewards(self) -> torch.Tensor:
        return _v1_reward.compute(self)

gym.register(
    id="Template-Holoassist-Reach-V1-Direct-v0",
    entry_point=f"{__name__}:HoloassistReachV1Env",
    ...
)
```

Register one gym ID per variant. Both/all variants then coexist; pick which
to train via `--task`.

### C. Add a new strategy file

1. Drop `rewards/dense_reach_v2.py` (or `observations/foo.py` etc.) into the
   right subpackage with the required function signature.
2. Either repoint `reach_env.py` (option A) or register a new gym ID (option B).
3. Re-run train with the chosen `--task`.

No package reinstall needed (`--symlink-install` style; strategy files are
picked up at import time).

---

## 3. TensorBoard live monitoring

In a **separate** PowerShell window while training runs:

```powershell
cd C:\Users\sebas\Github\IsaacLab

.\isaaclab.bat -p -m tensorboard.main --logdir logs\rsl_rl\holoassist_reach_direct --port 6006
```

Open <http://localhost:6006>. New scalars stream in as RSL-RL writes them
(every iteration; first event flush takes ~30 sec after train start).

### Key scalars

| Scalar | Healthy pattern |
|---|---|
| `Train/mean_reward` | Monotonically increases, then plateaus |
| `Train/mean_episode_length` | Decreases as policy learns to reach faster |
| `Loss/value_function` | Decreases then stabilises |
| `Loss/surrogate` | Small magnitude, no spikes |
| `Policy/mean_noise_std` | Slowly decreases (exploration shrinks) |
| `Train/learning_rate` | Adaptive — drops on KL spikes |

---

## 4. Comparing TensorBoard runs

Multiple runs under the same `--logdir` are auto-discovered.

1. Launch TensorBoard with the **same command as step 3** (point at the parent
   `holoassist_reach_direct\` directory, not a single run subfolder).
2. Left sidebar lists every run with a checkbox. Untick all, then tick the runs
   you want to overlay.
3. Each plot now overlays the ticked runs in distinct colors.
4. Bump the **Smoothing** slider (top-left) to 0.6-0.9 to read noisy scalars.

### Labelling runs for easier identification

Runs are timestamp-named (`2026-05-31_15-17-06`) by default. To make a run easy
to find later, rename the folder under
`logs\rsl_rl\holoassist_reach_direct\` after training completes (safe while
TensorBoard is closed). Example: `2026-05-31_15-17-06_v1_baseline\`.

### What to compare across variants

| Comparison | What it tells you |
|---|---|
| `mean_reward` final plateau | Best-performing reward shape |
| `mean_episode_length` final plateau | Fastest reach on average |
| Iter at which reward plateaus | Sample efficiency |
| Curve noise late in training | Policy stability |

Note: raw reward isn't comparable across variants with different weights — use
episode length + visual play.py for cross-variant judgement.

---

## 5. Play mode (load trained model with GUI)

### Auto-find latest checkpoint

```powershell
cd C:\Users\sebas\Github\IsaacLab

$run = (Get-ChildItem "logs\rsl_rl\holoassist_reach_direct" `
  | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
$ckpt = (Get-ChildItem "$run\model_*.pt" `
  | Sort-Object { [int]($_.BaseName -replace 'model_','') } -Descending `
  | Select-Object -First 1).FullName
Write-Output "Using checkpoint: $ckpt"

.\isaaclab.bat -p "C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI\isaac_rl\holoassist_tasks\scripts\rsl_rl\play.py" `
  --task Template-Holoassist-Reach-V1-Direct-v0 `
  --num_envs 4 `
  --checkpoint $ckpt
```

### Manual checkpoint path

```powershell
.\isaaclab.bat -p "C:\Users\sebas\Github\41118 Artificial Intelligence in Robotics\HoloAssist-AI\isaac_rl\holoassist_tasks\scripts\rsl_rl\play.py" `
  --task Template-Holoassist-Reach-V1-Direct-v0 `
  --num_envs 4 `
  --checkpoint "C:\Users\sebas\Github\IsaacLab\logs\rsl_rl\holoassist_reach_direct\2026-05-31_15-17-06\model_1900.pt"
```

### Critical details

- `--checkpoint` is the full path to a **`.pt` file**, NOT a run folder. Passing
  a folder triggers `PermissionError`.
- `--task` should match the task the model was trained on. v0 and v1 share
  obs + action shape so a v1 policy *runs* in either env, but use the matching
  one for honest evaluation.
- Do **NOT** pass `--headless` — you want the viewport.
- `--num_envs 4` (or up to 16) keeps the scene readable; defaults are higher.

### Camera controls in the Isaac Sim viewport

| Input | Action |
|---|---|
| Alt + Left-drag | Orbit |
| Alt + Middle-drag | Pan |
| Scroll | Zoom |
| `F` (with prim selected in Stage panel) | Frame on it |
| Spacebar | Pause / resume sim |
| Numpad 1 / 3 / 7 | Front / side / top view |

### Useful play flags

| Flag | Default | Purpose |
|---|---|---|
| `--task` | required | Must match training task ID |
| `--checkpoint` | required (.pt path) | Trained weights to load |
| `--num_envs` | 32 | 4-16 for clear viewing |
| `--video` | off | Record viewport to `<run>\videos\play\` |
| `--video_length` | 200 | Frames per recording |

---

## 6. Log directory layout

```
IsaacLab\logs\rsl_rl\holoassist_reach_direct\
  2026-05-31_14-59-21\               <- run timestamp (auto)
    model_0.pt                       <- save_interval=50 in rsl_rl_ppo_cfg.py
    model_50.pt
    ...
    model_1900.pt
    nn\                              <- TensorBoard event files
    git\                             <- code snapshot at train time
    params\                          <- env_cfg + agent_cfg yaml dumps
    videos\play\                     <- if --video was used during play
```

---

## 7. Gotchas

- **First train after env edits** — Isaac caches USD assets. If you changed
  scene cfg and see stale geometry, kill leftover Isaac processes
  (`taskkill /F /IM isaac-sim.exe`) before relaunching.
- **`--num_envs` too high** — symptom: silent CUDA OOM crash on startup.
  4070 Ti SUPER tops out around 4096-8192 envs for this task; back off if it
  crashes.
- **TensorBoard "no scalar data"** — refresh after ~30 sec (first event flush
  takes one save interval).
- **play.py shows immobile arm** — wrong task ID, env loaded a different action
  dim. Confirm `--task` matches training.
- **Action scale ceiling** — `cfg.action_scale_rad = 0.08` at 30 Hz = 2.4 rad/s,
  near UR3e hardware limit. Don't raise above 0.1 without rechecking joint vel
  limits in [reach_env_cfg.py](holoassist_tasks/source/holoassist_tasks/holoassist_tasks/tasks/direct/reach/reach_env_cfg.py).
- **TensorBoard via system python** — if you'd rather not boot Isaac for
  TensorBoard, `pip install tensorboard` in any system venv and run
  `tensorboard --logdir <full path to logs>` directly. Same UI.
