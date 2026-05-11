# ur3e_gazebo_sim

Gazebo Classic simulation for the HoloAssist UR3e pick/place workspace.

This package launches a UR3e with a fixed OnRobot RG2 gripper on the trolley/bench
workspace, with cubes and bins placed on top of the trolley. It is intended as the
visual and physics scene that will later be used by the reinforcement learning
environment.

## What It Launches

- Gazebo Classic `gzserver` and, by default, the `gzclient` GUI.
- The pick/place world from `worlds/pick_place_world.sdf`.
- A trolley/bench model using `models/table/meshes/UR3eTrolley_decimated.dae`.
- Four cube objects using `model://cube`.
- Two bin objects using `model://bin`.
- A UR3e robot description from `ur_description`.
- A fixed RG2 gripper model from `urdf/rg2_fixed.xacro`.
- `robot_state_publisher` and `joint_state_publisher` for TF/joint state output.

This package does not yet launch MoveIt, ros2_control, PPO, or any RL training code.
It is currently just the Gazebo scene for checking the workspace layout.

## Build

From the workspace root:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3e_gazebo_sim
source install/setup.bash
```

## Launch

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py
```

For extra Gazebo logging:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py verbose:=true
```

The launch file starts Gazebo paused by default so the scene can be inspected before
the objects start moving. Press the play button in Gazebo to unpause physics, or
launch with:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py paused:=false
```

To launch the server without the GUI:

```bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py gui:=false
```

## Launch Arguments

The main launch arguments are:

- `gui`: start the Gazebo GUI. Default: `true`.
- `paused`: start Gazebo paused. Default: `true`.
- `verbose`: print verbose Gazebo logs. Default: `false`.
- `spawn_robot`: spawn the UR3e/RG2. Default: `true`.
- `include_rg2`: attach the fixed RG2 gripper. Default: `true`.
- `robot_x`: robot base x position. Default: `0.0`.
- `robot_y`: robot base y position. Default: `0.0`.
- `robot_z`: robot base height on the trolley. Default: `1.078`.
- `robot_yaw`: robot base yaw in radians. Default: `3.141592653589793`.

## Changing Object Positions

Edit:

```text
src/ur3e_gazebo_sim/worlds/pick_place_world.sdf
```

Each included model has a pose:

```xml
<pose>x y z roll pitch yaw</pose>
```

Current world objects include:

```xml
<name>bench_table</name>
<pose>-0.031 0.0 0.29 0 0 0</pose>

<name>cube_1</name>
<pose>0.1 -0.40 1.11 0 0 0</pose>

<name>cube_2</name>
<pose>0.1 -0.25 1.11 0 0 0</pose>

<name>cube_3</name>
<pose>-0.1 -0.25 1.11 0 0 0</pose>

<name>cube_4</name>
<pose>-0.1 -0.40 1.11 0 0 0</pose>

<name>bin_1</name>
<pose>0.28 0.0 1.078 0 0 0</pose>

<name>bin_2</name>
<pose>-0.28 0.0 1.078 0 0 0</pose>
```

For extra bins or cubes, duplicate an `<include>` block, give it a unique `<name>`,
and change the pose.

## Changing The Robot Position In Code

Edit:

```text
src/ur3e_gazebo_sim/launch/ur3e_pick_place_world.launch.py
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

Change those `default_value` values, then rebuild the package.

## After Any Code Or World Edit

Always rebuild and source the workspace again:

```bash
cd /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3e_gazebo_sim
source install/setup.bash
```

## Troubleshooting

If ROS says the package cannot be found, source the workspace install:

```bash
source /home/ollie/git/RS2/main/HoloAssist/ur3e_rl_ws/install/setup.bash
```

If Gazebo says the entity already exists or the server address is already in use,
an old Gazebo launch is still running. Stop the old launch with `Ctrl+C` in its
terminal before launching again.

If Gazebo prints missing mesh errors, rebuild the package and make sure you launch
after sourcing `install/setup.bash`.
