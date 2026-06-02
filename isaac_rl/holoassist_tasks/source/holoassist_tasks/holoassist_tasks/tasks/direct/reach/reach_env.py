# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""HoloAssist UR3e reach task — DirectRLEnv.

This file is the env *shell* — lifecycle methods (init, setup_scene, reset,
dones) plus thin delegators to the active observation / reward / action
strategies. Strategy implementations live in sibling subpackages:

    observations/<strategy>.py    - build(env) -> dict
    rewards/<strategy>.py         - compute(env) -> Tensor
    actions/<strategy>.py         - process(env, action) + apply(env)

To A/B test a different strategy, edit the `as ..._strategy` import lines
below (or subclass this env and override the delegator method).
"""

from __future__ import annotations

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from holoassist_tasks.common import kinematics

from .actions import joint_delta as action_strategy
from .observations import ground_truth_12d as obs_strategy
from .reach_env_cfg import HoloassistReachEnvCfg
from .rewards import dense_reach_v1 as reward_strategy


class HoloassistReachEnv(DirectRLEnv):
    """UR3e + RG2 reach task — drive end-effector to a randomised target pose.

    Action space (6-D true joint-delta, per action_strategy):
        action[i] in [-1, 1] for each of 6 arm joints; multiplied by
        cfg.action_scale_rad to produce per-step joint position deltas.

    Observation space (per obs_strategy; default 12-D):
        See observations/ground_truth_12d.py for the layout.

    Reward (per reward_strategy; default dense_reach):
        See rewards/dense_reach.py for the term breakdown.

    Termination (this file's _get_dones):
        success — EE within cfg.success_tolerance_m of target
        failure — EE below cfg.robot_base_height_m - cfg.min_ee_clearance_below_base_m
        truncation — episode_length_buf reached cfg.episode_length_s
    """

    cfg: HoloassistReachEnvCfg

    # ------------------------------------------------------------------ lifecycle
    def __init__(self, cfg: HoloassistReachEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        # super().__init__ calls _setup_scene which assigns self._robot, then
        # _reset_idx for all envs. After it returns, the scene is fully alive.
        super().__init__(cfg, render_mode, **kwargs)

        # Resolve articulation indices once — strategy modules use these every step.
        # EE proxy: `left_inner_finger`. The URDF's `gripper_tcp` link was
        # frame-only (no inertia) and got merged into `onrobot_base_link` during
        # Isaac's "merge fixed joints" pass — not available as a separate body.
        # `left_inner_finger` is the closest available body to the actual grasp
        # point. Slight asymmetry vs. the right finger (~few mm) is well within
        # the 4 cm success tolerance. Phase 4b pick-place may refine to a true
        # TCP by averaging both fingertips + applying a local Z offset.
        self._ee_body_idx = self._robot.find_bodies("left_inner_finger")[0][0]
        self._arm_joint_ids = self._robot.find_joints(
            [
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ]
        )[0]

        # Per-env tensors used by strategy modules. Allocated once; written in
        # _reset_idx (target) / _pre_physics_step (joint_pos_target + actions
        # + prev_actions).
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._joint_pos_target = torch.zeros((self.num_envs, 6), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        # Previous action — cached at the start of each _pre_physics_step BEFORE
        # the new action is written. Used by reward strategies that need an
        # action-rate term (e.g. dense_reach_v1). Zero is a safe default for
        # step 0 of each episode (also re-zeroed in _reset_idx per env).
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        # Action from TWO steps ago — rolled one step BEFORE prev_actions in
        # _pre_physics_step (chain: prev_prev_actions <- prev_actions <- actions).
        # Used by reward strategies that need a jerk / second-difference term
        # (e.g. dense_reach_v2). Zero on episode steps 0 and 1 (reset clears it),
        # so the jerk term is undefined for those two steps — bounded spurious
        # contribution that washes out across the rollout.
        self.prev_prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Per-env IK reference joints — written in _reset_idx via nearest-
        # neighbour lookup against the precomputed grid built below.
        # Read by dense_reach_v3_ik for the IK tracking reward term.
        # Allocated for all task variants; v0/v1/v2 ignore it.
        self._ik_reference = torch.zeros((self.num_envs, 6), device=self.device)

        # Precompute the IK grid once at env init (see _build_ik_grid).
        # ~2 sec for default 20x20=400 IK calls; trivial cost amortised
        # over training. Populates self._ik_grid_xy and self._ik_grid_joints.
        self._build_ik_grid()

        # Cached min-EE-z scalar — used in _get_dones every step.
        self._min_ee_z = self.cfg.robot_base_height_m - self.cfg.min_ee_clearance_below_base_m

    def _setup_scene(self) -> None:
        """Build the scene: robot + ground + table + target marker + sky light."""
        # Robot — always spawned, one per env via {ENV_REGEX_NS}
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Ground plane — toggleable (cfg.add_ground_plane)
        if self.cfg.add_ground_plane:
            sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())

        # Table — toggleable (cfg.add_table); kinematic rigid body so the EE
        # can collide with it for natural episode termination on table strikes,
        # but it never moves itself
        if self.cfg.add_table:
            table_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(
                        self.cfg.table_size_xy_m[0],
                        self.cfg.table_size_xy_m[1],
                        self.cfg.table_thickness_m,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.45, 0.32)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(
                        0.0,
                        self.cfg.table_offset_y_m,
                        self.cfg.robot_base_height_m - self.cfg.table_thickness_m / 2.0,
                    ),
                ),
            )
            self._table = RigidObject(table_cfg)
            self.scene.rigid_objects["table"] = self._table

        # Sky light + directional sun — always; the sky light alone wasn't
        # bright enough to make the ground plane visible from steep camera angles.
        sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.90)).func(
            "/World/SkyLight",
            sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.90)),
        )
        sim_utils.DistantLightCfg(
            intensity=1500.0,
            angle=0.53,                      # ~Sun-like angular size
        ).func(
            "/World/SunLight",
            sim_utils.DistantLightCfg(intensity=1500.0, angle=0.53),
        )

        # Clone envs (Isaac Lab pattern — replicate the per-env prims across num_envs)
        self.scene.clone_environments(copy_from_source=False)
        # Filter cross-env collisions so robots in adjacent envs don't interact
        self.scene.filter_collisions(global_prim_paths=[])

        # Target marker — toggleable (cfg.add_target_marker). VisualizationMarkers
        # is a render-only prim batch (no physics, no per-env clone) — we hand it
        # all num_envs target positions in one call from _reset_idx.
        if self.cfg.add_target_marker:
            marker_cfg = VisualizationMarkersCfg(
                prim_path="/Visuals/TargetMarkers",
                markers={
                    "target": sim_utils.SphereCfg(
                        radius=self.cfg.target_marker_size_m / 2.0,
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
                    ),
                },
            )
            self._target_marker = VisualizationMarkers(marker_cfg)
        else:
            self._target_marker = None

    def _build_ik_grid(self) -> None:
        """Precompute IK solutions over the target spawn grid.

        Called once from __init__. Walks a `cfg.ik_grid_resolution`-square
        grid over (target_pos_range_x, target_pos_range_y) at
        target_pos_z, runs scipy IK for each grid point via
        holoassist_tasks.common.kinematics.compute_ik_reference, and
        caches the joint solutions + grid positions as device tensors.

        Trade-off: ~1-2 cm of IK reference error (from nearest-neighbour
        rounding) vs running fresh scipy IK per reset which would 20-30x
        training time at 4096 envs. The IK term is a soft pull, not a
        target — that error is well within the noise it tolerates.

        Populates:
            self._ik_grid_xy     : (n_grid, 2) float tensor on device —
                                    local-frame XY coordinates
            self._ik_grid_joints : (n_grid, 6) float tensor on device —
                                    IK joint solutions for each grid cell
        """
        n = self.cfg.ik_grid_resolution
        x_min, x_max = self.cfg.target_pos_range_x
        y_min, y_max = self.cfg.target_pos_range_y
        z = self.cfg.target_pos_z

        xs = torch.linspace(x_min, x_max, n)
        ys = torch.linspace(y_min, y_max, n)
        grid_xy = torch.stack(torch.meshgrid(xs, ys, indexing="ij"), dim=-1).reshape(-1, 2)

        print(
            f"[reach_env] Precomputing {n}x{n}={n*n} IK solutions over reach spawn zone...",
            flush=True,
        )

        ik_joints_list = []
        n_failed = 0
        for i in range(grid_xy.shape[0]):
            target_world = np.array([float(grid_xy[i, 0]), float(grid_xy[i, 1]), z])
            ok, joints, fk_err = kinematics.compute_ik_reference(target_world)
            if not ok:
                n_failed += 1
                print(
                    f"  WARN: IK failed at ({target_world[0]:+.2f}, {target_world[1]:+.2f}), "
                    f"err={fk_err*100:.1f}cm; using approach seed as fallback",
                    flush=True,
                )
                joints = kinematics._approach_seed(target_world)
            ik_joints_list.append(joints)

        self._ik_grid_xy = grid_xy.to(self.device)
        self._ik_grid_joints = torch.tensor(
            np.array(ik_joints_list), dtype=torch.float32, device=self.device
        )

        if n_failed > 0:
            print(
                f"[reach_env] IK grid built with {n_failed}/{n*n} fallback seeds. "
                "Consider tightening target_pos_range_* if many cells are unreachable.",
                flush=True,
            )
        else:
            print(f"[reach_env] IK grid built: {n*n} solutions, all converged", flush=True)

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset the given envs: joints to home pose, target to a new random pos."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # Reset robot joints. Arm goes to cfg.home_joint_pos (the v3+
        # ready pose: [0, -π/2, 0, -π/2, 0, 0]); gripper joints stay at
        # the articulation default (closed; see UR_ONROBOT_CFG.init_state).
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()
        home_arm = torch.tensor(self.cfg.home_joint_pos, device=self.device, dtype=joint_pos.dtype)
        joint_pos[:, self._arm_joint_ids] = home_arm.unsqueeze(0).expand(n, -1)
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        # Seed drive targets so drives hold the pose (legacy bug: targets default
        # to 0 which would drive the arm horizontal — see robot_test_v0.py)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # Seed our cached arm-joint target so the first _apply_action holds home
        self._joint_pos_target[env_ids] = joint_pos[:, self._arm_joint_ids]

        # Zero per-env action caches on reset so the first step of a new
        # episode doesn't see a stale "previous action" left over from the
        # previous episode (which would produce a spurious action-rate penalty
        # on step 0 of every new episode). Same reasoning applies to
        # prev_prev_actions (used by v2's jerk term).
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_prev_actions[env_ids] = 0.0

        # Randomise target position per env (uniform within configured bounds)
        target_x = (
            torch.rand(n, device=self.device)
            * (self.cfg.target_pos_range_x[1] - self.cfg.target_pos_range_x[0])
            + self.cfg.target_pos_range_x[0]
        )
        target_y = (
            torch.rand(n, device=self.device)
            * (self.cfg.target_pos_range_y[1] - self.cfg.target_pos_range_y[0])
            + self.cfg.target_pos_range_y[0]
        )
        target_z = torch.full((n,), self.cfg.target_pos_z, device=self.device)
        local_target = torch.stack([target_x, target_y, target_z], dim=-1)  # (n, 3)

        # Translate to world frame by adding the per-env origin (envs are cloned
        # with spacing cfg.scene.env_spacing — env_origins captures the offsets)
        env_origins = self.scene.env_origins[env_ids]  # (n, 3)
        self._target_pos[env_ids] = local_target + env_origins

        # Look up IK reference for each just-reset env by nearest-
        # neighbour against the precomputed grid (in local frame).
        # local_target is already in local frame (env_origin not yet
        # added), which matches the frame the grid was built in.
        # Read by dense_reach_v3_ik for the IK tracking reward term.
        local_target_xy = local_target[:, :2]                                  # (n, 2)
        dist_to_grid = torch.cdist(local_target_xy, self._ik_grid_xy)          # (n, n_grid)
        nearest_idx = dist_to_grid.argmin(dim=1)                               # (n,)
        self._ik_reference[env_ids] = self._ik_grid_joints[nearest_idx]

        # Update marker visuals (skip if disabled in cfg)
        if self._target_marker is not None:
            self._target_marker.visualize(translations=self._target_pos)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (terminated, truncated) per env.

        Terminated:
            success — EE within cfg.success_tolerance_m of target (per Q2 = C, episode ends on success)
            failure — EE world-z below self._min_ee_z (table-crash guard)
        Truncated:
            episode_length_buf has reached max — Isaac Lab base class handles
            the auto-reset that follows, we just report the bool tensor.
        """
        ee_pos = self._robot.data.body_link_state_w[:, self._ee_body_idx, :3]
        dist = torch.linalg.norm(ee_pos - self._target_pos, dim=-1)

        success = dist <= self.cfg.success_tolerance_m
        too_low = ee_pos[:, 2] < self._min_ee_z
        terminated = success | too_low

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, time_out

    # ------------------------------------------------------------------ delegators
    # These forward to the active strategy module. Strategy selection happens
    # at import time at the top of this file. Subclass + override to swap
    # dynamically per env class.

    def _get_observations(self) -> dict:
        return obs_strategy.build(self)

    def _get_rewards(self) -> torch.Tensor:
        return reward_strategy.compute(self)

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        # Roll the action history BEFORE the strategy overwrites self.actions:
        #     prev_prev_actions <- prev_actions   (action from 2 steps ago)
        #     prev_actions      <- actions        (action from previous step)
        #     actions           <- (new value written by action_strategy.process)
        # First-difference (v1 action_rate) reads env.prev_actions; second-
        # difference (v2 jerk) reads env.prev_prev_actions. Both are zero on
        # episode reset, so step-0 and step-1 contributions are bounded.
        self.prev_prev_actions = self.prev_actions.clone()
        self.prev_actions = self.actions.clone()
        action_strategy.process(self, action)

    def _apply_action(self) -> None:
        action_strategy.apply(self)
