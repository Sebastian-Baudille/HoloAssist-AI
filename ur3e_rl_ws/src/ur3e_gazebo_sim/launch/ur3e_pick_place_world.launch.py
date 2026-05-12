"""Launch the UR3e + RG2 pick/place scene in Ignition Fortress.

Pipeline:
  1. Set IGN_GAZEBO_RESOURCE_PATH so Ignition can find custom models.
  2. Start Ignition Fortress via ros_gz_sim.
  3. Spawn the UR3e+RG2 URDF into the running sim.
  4. Start robot_state_publisher (TF + /robot_description).
  5. Spawn and activate ros2_control controllers.
  6. Start gazebo_pose_bridge (publishes /cube/pose, /goal/pose, /tcp/pose for RL).

Usage:
    ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py
    ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py gui:=false
"""
from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _launch_setup(context, *args, **kwargs):
    pkg_share_dir = get_package_share_directory("ur3e_gazebo_sim")
    world_path = LaunchConfiguration("world").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"

    gz_args = f"-r -v 3 {world_path}"
    if not gui:
        gz_args = f"-s -r -v 3 {world_path}"

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("ros_gz_sim"),
                "launch",
                "gz_sim.launch.py",
            )
        ),
        launch_arguments={"gz_args": gz_args}.items(),
    )

    return [gz_sim]


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("ur3e_gazebo_sim")
    pkg_share_dir = get_package_share_directory("ur3e_gazebo_sim")

    world_default = PathJoinSubstitution([pkg_share, "worlds", "pick_place_world.sdf"])
    models_path = os.path.join(pkg_share_dir, "models")

    # Set resource paths in os.environ NOW so ign gazebo inherits them.
    # AppendEnvironmentVariable alone is unreliable for the SDF parser.
    for env_key in ("IGN_GAZEBO_RESOURCE_PATH", "GZ_SIM_RESOURCE_PATH"):
        existing = os.environ.get(env_key, "")
        paths = [p for p in existing.split(os.pathsep) if p]
        for p in (models_path, pkg_share_dir):
            if p not in paths:
                paths.append(p)
        os.environ[env_key] = os.pathsep.join(paths)
    controller_config = PathJoinSubstitution([pkg_share, "config", "ur3e_controllers.yaml"])
    robot_xacro = PathJoinSubstitution([pkg_share, "urdf", "ur3e_rg2_benchtop.urdf.xacro"])

    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    world_arg = DeclareLaunchArgument("world", default_value=world_default)
    spawn_robot_arg = DeclareLaunchArgument("spawn_robot", default_value="true")

    robot_x_arg = DeclareLaunchArgument("robot_x", default_value="0.0")
    robot_y_arg = DeclareLaunchArgument("robot_y", default_value="0.0")
    robot_z_arg = DeclareLaunchArgument(
        "robot_z",
        default_value="1.10",
        description="UR3e base_link height on the trolley tabletop.",
    )
    robot_yaw_arg = DeclareLaunchArgument(
        "robot_yaw",
        default_value="3.141592653589793",
        description="Yaw of the bench-mounted UR3e base.",
    )

    robot_description_content = Command(
        [
            FindExecutable(name="xacro"),
            " ",
            robot_xacro,
            " ",
            "name:=ur3e_rg2 ",
            "ur_type:=ur3e ",
            "base_x:=",
            LaunchConfiguration("robot_x"),
            " ",
            "base_y:=",
            LaunchConfiguration("robot_y"),
            " ",
            "base_z:=",
            LaunchConfiguration("robot_z"),
            " ",
            "base_yaw:=",
            LaunchConfiguration("robot_yaw"),
            " ",
            "include_rg2:=true ",
            "controllers_file:=",
            controller_config,
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": True}],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    spawn_robot = Node(
        package="ros_gz_sim",
        executable="create",
        name="spawn_ur3e_rg2",
        output="screen",
        arguments=[
            "-topic", "/robot_description",
            "-name", "ur3e_rg2",
        ],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="spawn_joint_state_broadcaster",
        output="screen",
        arguments=[
            "joint_state_broadcaster",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60",
        ],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        name="spawn_scaled_joint_trajectory_controller",
        output="screen",
        arguments=[
            "scaled_joint_trajectory_controller",
            "--controller-manager", "/controller_manager",
            "--controller-manager-timeout", "60",
            "--switch-timeout", "60",
        ],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    gazebo_pose_bridge = Node(
        package="ur3e_gazebo_sim",
        executable="gazebo_pose_bridge.py",
        name="gazebo_pose_bridge",
        output="screen",
    )

    return LaunchDescription(
        [
            gui_arg,
            world_arg,
            spawn_robot_arg,
            robot_x_arg,
            robot_y_arg,
            robot_z_arg,
            robot_yaw_arg,
            OpaqueFunction(function=_launch_setup),
            robot_state_publisher,
            TimerAction(period=4.0, actions=[spawn_robot]),
            TimerAction(period=8.0, actions=[joint_state_broadcaster_spawner]),
            TimerAction(period=10.0, actions=[trajectory_controller_spawner]),
            TimerAction(period=14.0, actions=[gazebo_pose_bridge]),
        ]
    )
