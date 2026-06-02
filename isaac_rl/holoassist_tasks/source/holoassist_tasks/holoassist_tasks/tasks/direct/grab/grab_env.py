# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""HoloAssist UR3e grab task — DirectRLEnv.

Mirror of the reach task's structure (env shell + strategy-module delegators),
extended for the grab task:
  - Scene gets a rigid-body cube (vs reach's visualization marker)
  - 7-D action (6 arm deltas + 1 gripper signal) per actions/joint_delta_gripper.py
  - 16-D observation per observations/ground_truth_16d.py
  - 6-term gated reward per rewards/dense_grab.py
  - Termination on success (cube lifted > 5 cm) or EE-too-low (table-crash guard)

Env-level setup beyond reach's pattern:
  1. Resolve indices for BOTH inner-finger bodies (gripper center = midpoint)
  2. Resolve indices for all 7 gripper joints (1 master + 5 followers + finger_width)
  3. At init, bump the linkage drive stiffness to cfg.linkage_drive_stiffness
     via write_joint_stiffness_to_sim — required for visual linkage rigidity
     under contact (control-level mimic; PhysxMimicJointAPI is metadata-only)
  4. At reset, optionally add ±cfg.joint_noise_rad noise to home pose
     (default 0 = strictly fixed; hook for Phase 6 robustness training)
  5. At reset, teleport cube to random XY within spawn zone + identity quat
"""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv

from .actions import joint_delta_gripper as action_strategy
from .grab_env_cfg import HoloassistGrabEnvCfg
from .observations import ground_truth_16d as obs_strategy
from .rewards import dense_grab, dense_grab_v1, dense_grab_v2, dense_grab_v3, dense_grab_v4, dense_grab_v5

# Reward-module dispatch: cfg.reward_module string -> module. Keeps all
# reward versions selectable per task registration without forking the
# env class. Add new versions here as they are introduced.
_REWARD_MODULES = {
    "dense_grab":    dense_grab,      # v0 — 6-term gated, side-sprawl approach works
    "dense_grab_v1": dense_grab_v1,   # v1 — 7-term, ungated orient (HOVER TRAP)
    "dense_grab_v2": dense_grab_v2,   # v2 — 8-term, rebalanced + time penalty (LATERAL MISALIGN)
    "dense_grab_v3": dense_grab_v3,   # v3 — v0 shape + elbow_up nudge at 10cm (FINGER DRAG)
    "dense_grab_v4": dense_grab_v4,   # v4 — v3 + anti-drag + grasp_act boost (SAFE-HOVER)
    "dense_grab_v5": dense_grab_v5,   # v5 — v0 shape + elbow_up at 5cm (conservative return)
}


class HoloassistGrabEnv(DirectRLEnv):
    """UR3e + RG2 grab task — reach to cube, grasp, lift > 5 cm above table.

    Action space (7-D, per actions/joint_delta_gripper.py):
        action[0:6] in [-1, 1] : per-arm-joint delta, scaled by cfg.action_scale_rad
        action[6]   in [-1, 1] : gripper signal (+1=open, -1=closed)

    Observation space (16-D, per observations/ground_truth_16d.py).

    Reward (per rewards/dense_grab.py):
        6 gated terms — see that module for the breakdown.

    Termination:
        success  — cube_z > table_top + cfg.success_lift_height
        failure  — EE z below cfg.robot_base_height_m - cfg.min_ee_clearance_below_base_m
        truncation — episode_length_buf reached max
    """

    cfg: HoloassistGrabEnvCfg

    # ------------------------------------------------------------------ lifecycle
    def __init__(self, cfg: HoloassistGrabEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        # super().__init__ calls _setup_scene which assigns self._robot + self._cube
        super().__init__(cfg, render_mode, **kwargs)

        # ---------- Resolve articulation indices ----------
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

        # Gripper linkage joints in canonical order — must match action strategy's
        # _LINKAGE_SIGNS and the gripper_coupling.FOLLOWER_GEARS order. Index 0 is
        # finger_joint (the master); the rest are mimic followers.
        linkage_names = (
            "finger_joint",
            "left_inner_knuckle_joint",
            "left_inner_finger_joint",
            "right_outer_knuckle_joint",
            "right_inner_knuckle_joint",
            "right_inner_finger_joint",
        )
        self._gripper_linkage_ids = [
            self._robot.find_joints(name)[0][0] for name in linkage_names
        ]
        self._gripper_linkage_ids_tensor = torch.tensor(
            self._gripper_linkage_ids, device=self.device, dtype=torch.long
        )
        self._finger_width_idx = self._robot.find_joints("finger_width")[0][0]

        # Body indices: BOTH inner fingers (we use their midpoint as the gripper
        # center for cube placement, alignment reward, and grasp detection).
        self._left_finger_idx  = self._robot.find_bodies("left_inner_finger")[0][0]
        self._right_finger_idx = self._robot.find_bodies("right_inner_finger")[0][0]

        # Forearm link — used by dense_grab_v3's elbow_up posture reward. The
        # forearm_link's body frame is at the elbow joint, so its world Z
        # approximates "how high is the elbow". Resolved here unconditionally
        # so any reward module can use it without re-resolving.
        self._forearm_idx = self._robot.find_bodies("forearm_link")[0][0]

        # ---------- Per-env buffers ----------
        self._joint_pos_target = torch.zeros((self.num_envs, 6), device=self.device)
        self._gripper_linkage_magnitude = torch.zeros((self.num_envs,), device=self.device)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # Cached failure threshold (table-crash guard)
        self._min_ee_z = self.cfg.robot_base_height_m - self.cfg.min_ee_clearance_below_base_m

        # ---------- Bump linkage drive stiffness ----------
        # PhysxMimicJointAPI is metadata-only in Isaac Sim 5.1 (proven in
        # Phase 4b-mimic), so we enforce mimic-like behaviour via stiff drives:
        # all 6 linkage joints get the same commanded position (computed in the
        # action strategy with the correct ±gear signs), and high stiffness
        # ensures they track the command rigidly under contact.
        stiff = torch.full(
            (self.num_envs, len(self._gripper_linkage_ids)),
            self.cfg.linkage_drive_stiffness,
            device=self.device,
        )
        damp = torch.full_like(stiff, self.cfg.linkage_drive_damping)
        self._robot.write_joint_stiffness_to_sim(stiff, joint_ids=self._gripper_linkage_ids_tensor)
        self._robot.write_joint_damping_to_sim(damp,   joint_ids=self._gripper_linkage_ids_tensor)

        print(
            f"[grab_env] {self.num_envs} envs initialised. "
            f"Linkage stiffness {self.cfg.linkage_drive_stiffness}, "
            f"gripper_closed_angle {self.cfg.gripper_closed_angle}, "
            f"joint_noise_rad {self.cfg.joint_noise_rad}",
            flush=True,
        )

    def _setup_scene(self) -> None:
        """Build the scene: robot + ground + table + cube + lights."""
        # Robot — always spawned, one per env via {ENV_REGEX_NS} regex in cfg
        self._robot = Articulation(self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self._robot

        # Ground plane
        if self.cfg.add_ground_plane:
            sim_utils.GroundPlaneCfg().func(
                "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
            )

        # Pedestal table (kinematic — never moves but the EE can collide with it)
        if self.cfg.add_table:
            table_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(
                        self.cfg.table_size_xy_m[0],
                        self.cfg.table_size_xy_m[1],
                        self.cfg.table_thickness_m,
                    ),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True, disable_gravity=True
                    ),
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

        # Cube (rigid body, dynamic, teleported in _reset_idx)
        cube_cfg = RigidObjectCfg(
            prim_path="/World/envs/env_.*/Cube",
            spawn=sim_utils.CuboidCfg(
                size=(self.cfg.cube_size_m, self.cfg.cube_size_m, self.cfg.cube_size_m),
                rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
                mass_props=sim_utils.MassPropertiesCfg(mass=self.cfg.cube_mass_kg),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=self.cfg.cube_friction,
                    dynamic_friction=self.cfg.cube_friction,
                    restitution=0.0,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.1, 0.1)),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, self.cfg.cube_pos_z)),
        )
        self._cube = RigidObject(cube_cfg)
        self.scene.rigid_objects["cube"] = self._cube

        # Lights — same as reach env
        sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.90)).func(
            "/World/SkyLight",
            sim_utils.DomeLightCfg(intensity=3500.0, color=(0.85, 0.85, 0.90)),
        )
        sim_utils.DistantLightCfg(intensity=1500.0, angle=0.53).func(
            "/World/SunLight",
            sim_utils.DistantLightCfg(intensity=1500.0, angle=0.53),
        )

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        """Reset given envs: robot to home pose (+ optional noise), cube to random pos."""
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES
        super()._reset_idx(env_ids)

        n = len(env_ids)

        # ---------- Robot: home pose (+ optional ±noise) for arm, gripper open ----------
        joint_pos = self._robot.data.default_joint_pos[env_ids].clone()

        # Arm: fixed ready pose plus optional uniform noise per joint
        home_arm = torch.tensor(self.cfg.home_joint_pos, device=self.device, dtype=joint_pos.dtype)
        home_arm = home_arm.unsqueeze(0).expand(n, -1).clone()
        if self.cfg.joint_noise_rad > 0.0:
            noise = (torch.rand_like(home_arm) * 2.0 - 1.0) * self.cfg.joint_noise_rad
            home_arm = home_arm + noise
        joint_pos[:, self._arm_joint_ids] = home_arm

        # Gripper: all linkage joints to 0 (open), finger_width to max
        for joint_idx in self._gripper_linkage_ids:
            joint_pos[:, joint_idx] = 0.0
        joint_pos[:, self._finger_width_idx] = self.cfg.gripper_max_width

        joint_vel = self._robot.data.default_joint_vel[env_ids]
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # Reset internal action caches
        self._joint_pos_target[env_ids] = joint_pos[:, self._arm_joint_ids]
        self._gripper_linkage_magnitude[env_ids] = 0.0
        self.actions[env_ids] = 0.0

        # ---------- Cube: random position within spawn zone ----------
        cube_x = (
            torch.rand(n, device=self.device)
            * (self.cfg.cube_pos_range_x[1] - self.cfg.cube_pos_range_x[0])
            + self.cfg.cube_pos_range_x[0]
        )
        cube_y = (
            torch.rand(n, device=self.device)
            * (self.cfg.cube_pos_range_y[1] - self.cfg.cube_pos_range_y[0])
            + self.cfg.cube_pos_range_y[0]
        )
        cube_z = torch.full((n,), self.cfg.cube_pos_z, device=self.device)
        local_cube = torch.stack([cube_x, cube_y, cube_z], dim=-1)

        # Translate to world frame (env_origins offsets each clone)
        env_origins = self.scene.env_origins[env_ids]
        world_cube = local_cube + env_origins

        # Pose tensor: (n, 7) = (xyz, quat_wxyz). Identity quat = (1, 0, 0, 0).
        cube_root_pose = torch.zeros((n, 7), device=self.device)
        cube_root_pose[:, :3] = world_cube
        cube_root_pose[:, 3] = 1.0
        cube_root_vel = torch.zeros((n, 6), device=self.device)
        self._cube.write_root_pose_to_sim(cube_root_pose, env_ids=env_ids)
        self._cube.write_root_velocity_to_sim(cube_root_vel, env_ids=env_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute (terminated, truncated) per env.

        Terminated:
            success — cube_z > table_top + cfg.success_lift_height
            failure — EE z below cached safety threshold (table-crash guard)
        Truncated:
            episode_length_buf has reached max — base class handles auto-reset.
        """
        cube_z = self._cube.data.root_pos_w[:, 2]
        table_z = self.cfg.robot_base_height_m
        success = cube_z > (table_z + self.cfg.success_lift_height)

        # Use left_inner_finger as the EE proxy for the table-crash guard
        # (same convention as reach task)
        ee_z = self._robot.data.body_link_state_w[:, self._left_finger_idx, 2]
        too_low = ee_z < self._min_ee_z

        terminated = success | too_low
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    # ------------------------------------------------------------------ delegators

    def _get_observations(self) -> dict:
        return obs_strategy.build(self)

    def _get_rewards(self) -> torch.Tensor:
        return _REWARD_MODULES[self.cfg.reward_module].compute(self)

    def _pre_physics_step(self, action: torch.Tensor) -> None:
        action_strategy.process(self, action)

    def _apply_action(self) -> None:
        action_strategy.apply(self)
