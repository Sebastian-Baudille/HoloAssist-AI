"""Sanity test (v0) for the HoloassistReachEnv.

Drives the env with random actions for N steps and reports:
    - shape / dtype of obs, reward, done tensors per step
    - any NaN / inf in obs or reward (would indicate a bug in our strategies)
    - termination counts (success vs EE-too-low vs truncation)
    - mean / min / max reward over the run

Counterpart to scripts/robot_test_v0.py (which validated the USD asset);
this validates the env-level Python wiring (cfg - strategies - stepping).

Run from the IsaacLab directory:
    cd C:\\Users\\sebas\\Github\\IsaacLab
    .\\isaaclab.bat -p "C:\\Users\\sebas\\Github\\41118 Artificial Intelligence in Robotics\\HoloAssist-AI\\isaac_rl\\scripts\\reach_test_v0.py"

Useful flags (forwarded to AppLauncher):
    --headless           run without GUI (faster)
    --device cpu         force CPU physics (default cuda:0)
    --num_envs 16        override cfg's num_envs=64 default
    --num_steps 200      override default step count (default 100)
    --seed 0             override RNG seed (default 0)
    --no_marker          disable the target marker (workaround for a current PhysX warning)
"""

import argparse
import sys

from isaaclab.app import AppLauncher

# -----------------------------------------------------------------------------
# CLI parsing — must happen before isaaclab modules are imported
# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="HoloassistReachEnv random-agent sanity test (v0)")
parser.add_argument("--num_envs", type=int, default=None, help="Override env cfg's num_envs (default uses cfg)")
parser.add_argument("--num_steps", type=int, default=100, help="How many env steps to run")
parser.add_argument("--seed", type=int, default=0, help="RNG seed for action sampling")
parser.add_argument("--no_marker", action="store_true", help="Disable the target visualization marker")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# -----------------------------------------------------------------------------
# Below this line, isaaclab can be imported
# -----------------------------------------------------------------------------
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import holoassist_tasks  # noqa: F401,E402  (triggers gym.register via auto-import)
from holoassist_tasks.tasks.direct.reach.reach_env_cfg import HoloassistReachEnvCfg  # noqa: E402

TASK_ID = "Template-Holoassist-Reach-Direct-v0"


def log(msg: str) -> None:
    """Print + flush so output survives Isaac Sim's warning spam in redirected streams."""
    print(msg, flush=True)
    sys.stdout.flush()


def main() -> None:
    # Build env cfg, optionally overriding num_envs + target_marker
    cfg = HoloassistReachEnvCfg()
    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs
    if args_cli.no_marker:
        cfg.add_target_marker = False
    num_envs = cfg.scene.num_envs

    log(f"[reach_test_v0] Task: {TASK_ID}")
    log(f"[reach_test_v0] num_envs = {num_envs}")
    log(f"[reach_test_v0] num_steps = {args_cli.num_steps}")
    log(f"[reach_test_v0] device = {cfg.sim.device}")
    log(f"[reach_test_v0] action_scale_rad = {cfg.action_scale_rad}")
    log(f"[reach_test_v0] success_tolerance_m = {cfg.success_tolerance_m}")
    log(f"[reach_test_v0] add_target_marker = {cfg.add_target_marker}")

    # Create env
    env = gym.make(TASK_ID, cfg=cfg)
    unwrapped = env.unwrapped
    device = unwrapped.device

    log(f"[reach_test_v0] env loaded — obs_space {env.observation_space}, action_space {env.action_space}")

    # Reset
    obs, _info = env.reset()
    log(f"[reach_test_v0] reset OK — obs['policy'].shape = {tuple(obs['policy'].shape)}")
    log(f"[reach_test_v0] sample obs (env 0): {obs['policy'][0].cpu().numpy().round(3)}")

    # Random-action loop
    rng = torch.Generator(device=device).manual_seed(args_cli.seed)

    reward_total = torch.zeros(num_envs, device=device)
    reward_min = torch.full((num_envs,), float("inf"), device=device)
    reward_max = torch.full((num_envs,), float("-inf"), device=device)
    success_count = 0
    too_low_count = 0
    timeout_count = 0
    nan_obs_steps = 0
    nan_reward_steps = 0

    log(f"[reach_test_v0] Stepping {args_cli.num_steps} random actions ...")
    for step in range(args_cli.num_steps):
        action = torch.rand((num_envs, 6), generator=rng, device=device) * 2.0 - 1.0
        obs, reward, terminated, truncated, _info = env.step(action)

        if not torch.isfinite(obs["policy"]).all():
            nan_obs_steps += 1
        if not torch.isfinite(reward).all():
            nan_reward_steps += 1

        reward_total += reward
        reward_min = torch.minimum(reward_min, reward)
        reward_max = torch.maximum(reward_max, reward)

        if terminated.any() or truncated.any():
            ee_pos = unwrapped._robot.data.body_link_state_w[:, unwrapped._ee_body_idx, :3]
            dist = torch.linalg.norm(ee_pos - unwrapped._target_pos, dim=-1)
            success_mask = (dist <= cfg.success_tolerance_m) & terminated
            too_low_mask = (ee_pos[:, 2] < unwrapped._min_ee_z) & terminated
            success_count += int(success_mask.sum().item())
            too_low_count += int(too_low_mask.sum().item())
            timeout_count += int(truncated.sum().item())

        if step % 20 == 0 or step == args_cli.num_steps - 1:
            log(
                f"  step {step:>3d}  "
                f"obs={tuple(obs['policy'].shape)}  "
                f"r=[{reward.min().item():+.3f}, {reward.max().item():+.3f}]  "
                f"term={int(terminated.sum().item())} trunc={int(truncated.sum().item())}"
            )

    # Summary
    log("[reach_test_v0] === Summary ===")
    log(f"  Total steps: {args_cli.num_steps}")
    log(f"  Mean cumulative reward / env: {reward_total.mean().item():+.3f}")
    log(f"  Per-step reward range across run: [{reward_min.min().item():+.3f}, {reward_max.max().item():+.3f}]")
    log("  Termination counts (across all envs and steps):")
    log(f"    success     : {success_count}")
    log(f"    EE too low  : {too_low_count}")
    log(f"    timeout     : {timeout_count}")
    log("  NaN/inf detected:")
    log(f"    obs steps   : {nan_obs_steps}")
    log(f"    reward steps: {nan_reward_steps}")

    ok = (nan_obs_steps == 0) and (nan_reward_steps == 0)
    if ok:
        log("[reach_test_v0] OK — env steps cleanly with random actions, no NaNs.")
    else:
        log("[reach_test_v0] FAIL — NaN/inf detected. See counts above.")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
