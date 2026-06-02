# Team Content — for index.html team section

Current site only shows Nic and Sebastian. Full team below.

---

## John Chen (John-A-Chen)
**Role:** Simulation, RL Training, Perception Foundation, Integration Lead

Started the project's Gazebo simulation environment and ported Moveit2 collision avoidance.
Designed and built the MuJoCo training backend (parallel to Isaac Sim) to iterate on PPO reward shaping fast.
Implemented the Jacobian IK module used to seed the policy with valid joint configurations.
Built the full staged RL pipeline: Reach → Transport coordinator state machine with separate trained sub-policies.
Designed and integrated the gripper (OnRobot RG2) into the RL environment and the ROS 2 stack.
Set up the OnRobot driver, description, and combined UR+OnRobot description packages.
Started the initial cube perception package (point cloud detection, ROS 2 node, RViz markers).

GitHub: John-A-Chen

---

## Guy Smith (GuyESmith)
**Role:** Perception & Clustering

Took ownership of the perception stack and rebuilt it for reliability.
Replaced the AprilTag 3 tracker approach (used in base HoloAssist) with a pure point cloud pipeline.
Built the Stage A dataset capture system (60 scenes, automated, reads settled poses from Gazebo).
Benchmarked K-Means at 2.65 cm centroid error, then replaced it with DBSCAN — achieving 1.63 cm with 82% exact-count rate.
Key DBSCAN parameters: eps=0.015, min_samples=20, size filter 50–1500 pts.
Fixed cube spacing bug (1.5× → 2.0× cube size) that caused DBSCAN to merge adjacent cubes.
Tightened UR3e joint limits to prevent self-collision during Phase A training (shoulder_lift, elbow, wrist_1).
Added configuration penalty reward term (soft ramp near joint limit) to give gradient signal away from limits.
Ran Phase A training: 14D obs (EE + cube + joints + height + timestep), reach-only reward, 99% success at 696k steps.
Contributed IK-seeded training approach (Jun 1): robot rewarded for being in valid joint configurations to reach the goal.
Documented TensorBoard/RViz monitoring workflow and fixed RG2 gripper mesh STL files (corrupt COLOR-header binary).

GitHub: GuyESmith

---

## Sebastian Baudille (Sebastian-Baudille)
**Role:** Isaac Sim & RL Control

Initiated the project (initial commit, Project Ignition).
Set up Isaac Sim 5.1 environment with the UR3e + RG2 as a layered USD asset (base, physics, robot, sensor).
Built the URDF→USD conversion pipeline (ur_onrobot_prepared.urdf → .usd).
Created the `holoassist_tasks` Isaac Lab extension with DirectRLEnv reach task.
Implemented `ground_truth_12d` observation module, `joint_delta` action module, RSL-RL PPO config.
Updated Gazebo/RViz sim files for dataset collection.
Added planning documents and repo refactoring.
Training: 4096 GPU-parallel envs, RTX 4070 Ti SUPER, ~20 min per 500 iterations.
Current status: ~60–70% reach success after 500 iterations, partially converged.

GitHub: Sebastian-Baudille

---

## Ollie Lau (ollie-lau / LazyTurtle)
**Role:** Gazebo Simulation Foundation & Safety Layer

Bootstrapped the Gazebo Classic environment with the UR3e and table (the scene everything else built on).
Wrote the original PPO training scripts (single-agent and parallel) and the UR3e safety layer package.
Safety layer includes: collision detection, joint limit checking, SafetyChecker, MoveIt collision checker node.
John later extended this with MoveIt integration, cube randomization, and parallel IGN_PARTITION isolation.
After handoff, Ollie took ownership of the Gazebo sim and Moveit2 collision avoidance subsystem.

GitHub: ollie-lau (LazyTurtle)

---

## Nicholas Sabatini (MafiaPineapple)
**Role:** Perception (original HoloAssist), Unity XR Dashboard, Website & RL Support

### Original HoloAssist perception system
Nic owns the full AprilTag 3 perception pipeline that was the precursor to the DBSCAN system:
- **Eye-to-hand calibration** (`eye_hand_calibration_node.py`): OpenCV `calibrateHandEye` (Park solver),
  observing AprilTag on end effector from multiple robot poses. Validated at 0.4 mm accuracy in sim,
  both sim mode and real hardware mode via ROS services.
- **`cube_pose_relay.py`**: TF-aware ROS bridge publishing cube poses from `workspace_frame` to `base_link`
- **`CubePoseSubscriber.cs`**: Unity C# subscriber for `/holoassist/unity/cube_N_pose` topics
- **Hardware validation**: `origin/nic` merged as "hardware-tested perception wiring" — Nic validated
  the full AprilTag pipeline on the real UR3e robot
- **`workspace_perception_params.yaml`**: calibration parameters tuned for real hardware
- The underlying workspace board node used SVD (Kabsch algorithm) + RANSAC depth plane fitting

### Unity XR Dashboard (base HoloAssist — Nic's primary subsystem)
Built the entire Unity Quest 3 mixed reality dashboard from scratch:
- Hand tracking → robot teleoperation (RMRC)
- Working gripper control
- Steam Deck dashboard app (WebSocket bridge, no Steam Link needed)
- Self-collision guard (ROS 2 velocity filter node)
- E-stop, MoveIt integration in dashboard

### HoloAssist-AI contributions
- Built this website and all detailed run guides
- Early RL support (May 12): keyboard teleoperation for training data, sim launch fixes,
  approach reward, pretrained model loading, home position, action normalization

GitHub: MafiaPineapple
