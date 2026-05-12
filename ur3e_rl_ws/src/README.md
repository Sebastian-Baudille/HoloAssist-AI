# UR3e Gazebo Reinforcement Learning Workspace

This `src` folder contains the Gazebo simulation and first reinforcement learning
stack for the HoloAssist UR3e trolley workspace.

The current goal is to train a Stable-Baselines3 PPO policy that moves the UR3e
end effector towards a cube in Gazebo using joint-delta control.

## Packages

```text
src/
  ur3e_gazebo_sim/
    Gazebo Classic scene with the UR3e, fixed RG2 gripper, trolley, cubes, and bins.

  ur3e_rl_env/
    Gymnasium environment, ROS interface, reward function, PPO training script,
    evaluation script, and joint command smoke test.

  ur3e_safety_layer/
    Joint delta clamps, simple UR3e joint limits, collision/TCP-height safety checks,
    and placeholder safety nodes.

  ur3e_policy_controller/
    Runtime ROS 2 node that loads a trained PPO model and sends safe joint targets.
```

## What Works Now

- The Gazebo scene launches with the trolley, cubes, bins, UR3e, and fixed RG2.
- The RL packages build as ROS 2 `ament_python` packages.
- The PPO environment code creates the required action and observation spaces.
- The 29-value observation builder is implemented.
- Joint-delta safety clamping is implemented.
- PPO training/evaluation scripts are wired as `ros2 run` executables.

## Important Current Limitation

The RL code expects live ROS topics for joint control and object/goal/TCP poses.
The Gazebo scene is visually correct, but PPO training will only work once these
topics are being published and the joint trajectory controller accepts commands:

```text
/joint_states
/tcp_pose_broadcaster/pose
/cube/pose
/goal/pose
/collision_flag
/scaled_joint_trajectory_controller/joint_trajectory
```

For the first reach task, `/goal/pose` is optional in the code. If it is missing,
the cube pose is used as the goal. `/collision_flag` can be provided by the
placeholder:

```bash
ros2 run ur3e_safety_layer moveit_collision_checker
```

That placeholder always publishes `False`; replace it with a real MoveIt collision
checker before using this on hardware.

## Install Python RL Dependencies

ROS dependencies are handled by the workspace packages, but PPO needs extra Python
packages:

```bash
python3 -m pip install gymnasium stable-baselines3
```

Optional but useful for training logs:

```bash
python3 -m pip install tensorboard
```

## Build Everything

From the workspace root:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```

To rebuild only these packages:

```bash
colcon build --packages-select \
  ur3e_gazebo_sim \
  ur3e_safety_layer \
  ur3e_rl_env \
  ur3e_policy_controller
source install/setup.bash
```

## Launch The Gazebo Sim

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py
```

Verbose Gazebo logging:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py verbose:=true
```

Launch without the Gazebo GUI:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py gui:=false
```

Gazebo starts paused by default. Press play in Gazebo, or launch unpaused:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py paused:=false
```

## Scene Layout

The world file is:

```text
ur3e_gazebo_sim/worlds/pick_place_world.sdf
```

That file controls the trolley, cubes, and bins:

```xml
<pose>x y z roll pitch yaw</pose>
```

Current scene objects:

```text
bench_table  - trolley/bench mesh
cube_1       - cube object
cube_2       - cube object
cube_3       - cube object
cube_4       - cube object
bin_1        - bin object
bin_2        - bin object
```

To add another cube or bin, duplicate an `<include>` block in the world file,
give it a unique `<name>`, and change the pose.

## Change The Robot Position In Code

Edit:

```text
ur3e_gazebo_sim/launch/ur3e_pick_place_world.launch.py
```

Look for:

```python
robot_x_arg = DeclareLaunchArgument("robot_x", default_value="0.0")
robot_y_arg = DeclareLaunchArgument("robot_y", default_value="0.0")
robot_z_arg = DeclareLaunchArgument(
    "robot_z",
    default_value="1.078",
    description="UR3e base_link height on the trolley tabletop.",
)
robot_yaw_arg = DeclareLaunchArgument(
    "robot_yaw",
    default_value="3.141592653589793",
    description="Yaw of the bench-mounted UR3e base. Pi matches the Unity scene orientation.",
)
```

Change the `default_value` numbers, rebuild, source, and relaunch.

## PPO Design

The first PPO task is reach-to-cube:

- Control mode: joint-delta control.
- Action space: continuous `Box(-0.03, 0.03, shape=(6,))`.
- Each action is added to the six UR3e joint angles.
- Safety layer clamps any delta to max `0.05 rad`.
- First success condition: end effector is within `0.04 m` of the cube.

Observation shape is `(29,)`:

```text
6 joint positions
6 joint velocities
3 end-effector position values
3 cube/object position values
3 goal position values
3 object position minus end-effector position values
3 goal position minus object position values
1 grasped flag, currently always 0
1 collision flag
```

## RL Topics

Configured in:

```text
ur3e_rl_env/ur3e_rl_env/ros_interface.py
```

Current constants:

```python
JOINT_STATE_TOPIC = "/joint_states"
TCP_POSE_TOPIC = "/tcp_pose_broadcaster/pose"
CUBE_POSE_TOPIC = "/cube/pose"
GOAL_POSE_TOPIC = "/goal/pose"
COLLISION_FLAG_TOPIC = "/collision_flag"
JOINT_TRAJECTORY_TOPIC = "/scaled_joint_trajectory_controller/joint_trajectory"
```

Change these constants if your controller or pose publishers use different topic
names.

## Smoke Test Joint Commands

Use this before training. It reads the current UR3e joint positions, adds `0.1 rad`
to `shoulder_pan_joint`, publishes a `JointTrajectory`, and prints before/after
joint positions.

```bash
ros2 run ur3e_rl_env smoke_test_joint_command
```

If the numbers do not change, PPO training will not be able to move the robot yet.
Check that a trajectory controller is running and subscribed to:

```bash
ros2 topic info /scaled_joint_trajectory_controller/joint_trajectory
```

## Start Placeholder Collision Publisher

In a separate terminal:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur3e_safety_layer moveit_collision_checker
```

This publishes:

```text
/collision_flag = False
```

## Record Teleop Demonstrations

Collect human demonstrations from the Unity XR teleop system for
pretraining the PPO policy. The recorder runs alongside the teleop node
and passively samples robot state — it does not send any commands.

Prerequisites: UR driver running (`/joint_states` publishing), teleop
node active (Unity publishing to `/forward_velocity_controller/commands`).
TCP pose is read from TF (`base_link` → `tool0`), so no extra pose
publisher is needed.

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur3e_rl_env record_demo
```

Set a static object position if perception is not running:

```bash
ros2 run ur3e_rl_env record_demo --ros-args \
    -p object_xyz:="[0.3, 0.0, 0.05]"
```

Configurable parameters (via `--ros-args -p`):

```text
hz                 Sampling rate (default: 5.0)
max_episode_steps  Steps before auto-segmenting a new episode (default: 200)
base_frame         TF base frame (default: base_link)
tcp_frame          TF tool frame (default: tool0)
object_topic       Object pose topic (default: /cube/pose)
object_xyz         Static object fallback [x, y, z] (default: [0.3, 0.0, 0.05])
goal_topic         Goal pose topic (default: /goal/pose)
output_dir         Output directory (default: ./demo_data)
velocity_topic     Teleop velocity command topic (default: /forward_velocity_controller/commands)
```

Episode segmentation is automatic: episodes end on success (TCP within
4 cm of object), failure (collision or TCP below z=0.02 m), or after
`max_episode_steps`. Press Ctrl+C to save the current episode and exit.

Output files (one per episode):

```text
demo_data/
  demo_episode_001.npz
  demo_episode_002.npz
  ...
```

Each NPZ contains:

```text
observations       (T, 29)  float32  — PPO observation vector
actions            (T, 6)   float32  — joint delta (clipped to [-0.03, 0.03])
rewards            (T,)     float32  — reward from the RL reward function
next_observations  (T, 29)  float32  — observation after action
terminals          (T,)     bool     — True if episode ended (success/failure)
velocity_commands  (T, 6)   float32  — raw velocity commands from teleop
```

## Pretrain PPO From Demos

Behavior-clone the demo data into a PPO policy network, then fine-tune
with RL. No Gazebo or ROS needed — uses only the saved NPZ files.

```bash
ros2 run ur3e_rl_env pretrain_from_demos
```

Environment variables:

```text
UR3E_DEMO_INPUT_DIR     Demo directory (default: ./demo_data)
UR3E_BC_OUTPUT_PATH     Output model (default: ./rl_models/ppo_ur3e_pretrained)
UR3E_BC_EPOCHS          Training epochs (default: 100)
UR3E_BC_LR              Learning rate (default: 1e-3)
UR3E_BC_BATCH_SIZE      Batch size (default: 256)
```

Then fine-tune with PPO:

```python
from stable_baselines3 import PPO
model = PPO.load("./rl_models/ppo_ur3e_pretrained", env=your_env)
model.learn(total_timesteps=200000)
```

## Train PPO

Start Gazebo and the required pose/controller topics first, then run:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur3e_rl_env train_ppo
```

Training settings:

```text
Algorithm: Stable-Baselines3 PPO
Timesteps: 200000
Model output: ./rl_models/ppo_ur3e_reach_object
Checkpoints: ./rl_models/checkpoints/
TensorBoard logs: ./tb_logs/
```

## Train PPO With 4 Parallel Gazebo Sims

For faster training on a multi-core machine, use the parallel trainer. This
script launches four headless Gazebo instances for you, so stop any old Gazebo
launches first.

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run ur3e_rl_env train_ppo_parallel
```

Default parallel settings:

```text
UR3E_RL_NUM_ENVS=4
UR3E_RL_BASE_ROS_DOMAIN_ID=30
UR3E_RL_BASE_GAZEBO_MASTER_PORT=11400
UR3E_RL_CONTROL_DT=0.1
UR3E_RL_RESET_DURATION=0.4
UR3E_RL_TORCH_THREADS=2
```

Each worker uses a separate ROS domain and Gazebo master port:

```text
worker 0: ROS_DOMAIN_ID=30, GAZEBO_MASTER_URI=http://127.0.0.1:11400
worker 1: ROS_DOMAIN_ID=31, GAZEBO_MASTER_URI=http://127.0.0.1:11401
worker 2: ROS_DOMAIN_ID=32, GAZEBO_MASTER_URI=http://127.0.0.1:11402
worker 3: ROS_DOMAIN_ID=33, GAZEBO_MASTER_URI=http://127.0.0.1:11403
```

Gazebo logs are written to:

```text
./rl_models/gazebo_parallel_logs/
```

Quick 20k-step test:

```bash
UR3E_RL_TOTAL_TIMESTEPS=20000 ros2 run ur3e_rl_env train_ppo_parallel
```

If the laptop becomes sluggish, try two sims:

```bash
UR3E_RL_NUM_ENVS=2 ros2 run ur3e_rl_env train_ppo_parallel
```

If the laptop has spare CPU, try six sims:

```bash
UR3E_RL_NUM_ENVS=6 ros2 run ur3e_rl_env train_ppo_parallel
```

## Evaluate PPO

After training:

```bash
ros2 run ur3e_rl_env evaluate_policy
```

It runs 20 episodes and prints the success rate.

## Run A Trained Policy

After training:

```bash
ros2 run ur3e_policy_controller rl_policy_node
```

By default it loads:

```text
./rl_models/ppo_ur3e_reach_object
```

To use a different model:

```bash
UR3E_RL_MODEL_PATH=/path/to/model ros2 run ur3e_policy_controller rl_policy_node
```

The policy node reads the current observation, predicts a joint-delta action,
clamps it through `SafetyChecker`, and publishes a joint target every `0.2 s`.

## Useful Checks

List executables:

```bash
ros2 pkg executables ur3e_rl_env
ros2 pkg executables ur3e_safety_layer
ros2 pkg executables ur3e_policy_controller
```

Check topic availability:

```bash
ros2 topic list
ros2 topic echo /joint_states --once
ros2 topic echo /tcp_pose_broadcaster/pose --once
ros2 topic echo /cube/pose --once
ros2 topic echo /collision_flag --once
```

Check the trajectory command subscriber:

```bash
ros2 topic info /scaled_joint_trajectory_controller/joint_trajectory
```

## Troubleshooting

If ROS says a package cannot be found:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```

If Gazebo says the entity already exists or the address is already in use, an old
Gazebo instance is still running. Stop it with `Ctrl+C` in the terminal that
launched Gazebo.

If `train_ppo` cannot import `gymnasium` or `stable_baselines3`:

```bash
python3 -m pip install gymnasium stable-baselines3
```

If the RL environment waits forever, one of the required topics is missing. Start
by checking:

```bash
ros2 topic list
```

If `smoke_test_joint_command` publishes but the robot does not move, the Gazebo
robot is probably not connected to a joint trajectory controller yet. That is the
next integration step before PPO can actually learn.

## Recommended Development Order

1. Launch Gazebo and confirm the visual scene is correct.
2. Add or start publishers for TCP pose, cube pose, and collision flag.
3. Add or start the UR3e joint trajectory controller.
4. Run `smoke_test_joint_command` and confirm the robot moves.
5. Run `train_ppo`.
6. Evaluate with `evaluate_policy`.
7. Run the trained model with `rl_policy_node`.
