# HoloAssist-AI RL Guide

## Launch Sim And Train
```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Terminal 1: Gazebo + bridges
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py gui:=false

# Terminal 2: single-env training
UR3E_RL_TOTAL_TIMESTEPS=10000 ros2 run ur3e_rl_env train_ppo

# Terminal 2: parallel training
UR3E_RL_TOTAL_TIMESTEPS=500000 \
UR3E_RL_NUM_ENVS=4 \
UR3E_RL_PPO_N_STEPS=2048 \
UR3E_RL_PPO_BATCH_SIZE=256 \
ros2 run ur3e_rl_env train_ppo_parallel
```

## Observation Vector (13D)
`[0:3]` normalized EE XYZ, `[3:6]` normalized target-cube XYZ, `[6:9]` normalized bin XYZ, `[9]` grasped flag, `[10]` gripper state, `[11]` EE height normalized, `[12]` timestep normalized.

This keeps task-relevant state only and removes joint-level redundancy from the old 29D vector.

## Reward Phases
- Reach: penalize EE-to-cube distance.
- Grasp: bonus when grasped.
- Transport: when grasped, penalize cube-to-bin distance.
- Place: large bonus when `cube_in_bin` is true.
- Penalties: collision, time pressure, and action smoothness.

## Deploy To Real Robot
```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash

# Perception
ros2 launch realsense2_camera rs_launch.py
ros2 run ur3e_gazebo_sim pointcloud_cube_detector

# Policy deployment
UR3E_RL_MODEL_PATH=./rl_models/ppo_best \
ros2 run ur3e_policy_controller rl_policy_node
```

## TensorBoard Metrics
- `ep_rew_mean`: mean episodic reward trend (primary learning progress signal).
- `ep_len_mean`: average episode length.
- `rollout/success_rate` (if logged by wrappers): task-completion ratio.
- PPO losses (`policy_gradient_loss`, `value_loss`, `entropy_loss`) indicate optimization stability.
