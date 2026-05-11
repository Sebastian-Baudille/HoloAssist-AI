from __future__ import annotations

import os

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    AppendEnvironmentVariable,
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    TimerAction,
)
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def _start_gazebo(context, *args):
    del args
    world = LaunchConfiguration("world").perform(context)
    gui = LaunchConfiguration("gui").perform(context).lower() == "true"
    paused = LaunchConfiguration("paused").perform(context).lower() == "true"
    verbose = LaunchConfiguration("verbose").perform(context).lower() == "true"
    gazebo_ros_prefix = get_package_prefix("gazebo_ros")
    gazebo_ros_lib_dir = os.path.join(gazebo_ros_prefix, "lib")

    server_cmd = ["gzserver"]
    if verbose:
        server_cmd.append("--verbose")
    if paused:
        server_cmd.append("-u")
    server_cmd.extend(
        [
            "-s",
            os.path.join(gazebo_ros_lib_dir, "libgazebo_ros_init.so"),
            "-s",
            os.path.join(gazebo_ros_lib_dir, "libgazebo_ros_factory.so"),
            world,
        ]
    )

    actions = [ExecuteProcess(cmd=server_cmd, output="screen")]
    if gui:
        actions.append(ExecuteProcess(cmd=["gzclient"], output="screen"))
    return actions


def generate_launch_description() -> LaunchDescription:
    pkg_share = FindPackageShare("ur3e_gazebo_sim")
    pkg_share_dir = get_package_share_directory("ur3e_gazebo_sim")
    package_share_root = os.path.dirname(pkg_share_dir)
    ros_share_root = os.path.dirname(get_package_share_directory("ur_description"))

    world_default = PathJoinSubstitution([pkg_share, "worlds", "pick_place_world.sdf"])
    models_path = PathJoinSubstitution([pkg_share, "models"])
    robot_xacro = PathJoinSubstitution([pkg_share, "urdf", "ur3e_rg2_benchtop.urdf.xacro"])

    gui_arg = DeclareLaunchArgument("gui", default_value="true")
    paused_arg = DeclareLaunchArgument(
        "paused",
        default_value="true",
        description="Start Gazebo paused so the visual scene can be inspected before controllers/RL are added.",
    )
    verbose_arg = DeclareLaunchArgument("verbose", default_value="false")
    world_arg = DeclareLaunchArgument("world", default_value=world_default)
    use_sim_time_arg = DeclareLaunchArgument("use_sim_time", default_value="true")
    include_rg2_arg = DeclareLaunchArgument("include_rg2", default_value="true")
    spawn_robot_arg = DeclareLaunchArgument("spawn_robot", default_value="true")

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

    joint_defaults = {
        "shoulder_pan_joint": "0.0",
        "shoulder_lift_joint": "-1.5708",
        "elbow_joint": "1.5708",
        "wrist_1_joint": "-1.5708",
        "wrist_2_joint": "-1.5708",
        "wrist_3_joint": "0.0",
    }

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
            "include_rg2:=",
            LaunchConfiguration("include_rg2"),
        ]
    )
    robot_description = {
        "robot_description": ParameterValue(robot_description_content, value_type=str)
    }

    joint_state_publisher = Node(
        package="joint_state_publisher",
        executable="joint_state_publisher",
        name="joint_state_publisher",
        output="screen",
        parameters=[
            robot_description,
            {
                "rate": 30,
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "zeros": {name: float(value) for name, value in joint_defaults.items()},
            },
        ],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": LaunchConfiguration("use_sim_time")}],
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    spawn_entity_args = [
        "-topic",
        "/robot_description",
        "-entity",
        "ur3e_rg2",
    ]

    spawn_robot = Node(
        package="gazebo_ros",
        executable="spawn_entity.py",
        name="spawn_ur3e_rg2",
        output="screen",
        arguments=spawn_entity_args,
        condition=IfCondition(LaunchConfiguration("spawn_robot")),
    )

    return LaunchDescription(
        [
            gui_arg,
            paused_arg,
            verbose_arg,
            world_arg,
            use_sim_time_arg,
            include_rg2_arg,
            spawn_robot_arg,
            robot_x_arg,
            robot_y_arg,
            robot_z_arg,
            robot_yaw_arg,
            AppendEnvironmentVariable(
                name="GAZEBO_MODEL_PATH",
                value=models_path,
                separator=os.pathsep,
            ),
            AppendEnvironmentVariable(
                name="GAZEBO_MODEL_PATH",
                value=package_share_root,
                separator=os.pathsep,
            ),
            AppendEnvironmentVariable(
                name="GAZEBO_MODEL_PATH",
                value=ros_share_root,
                separator=os.pathsep,
            ),
            OpaqueFunction(function=_start_gazebo),
            joint_state_publisher,
            robot_state_publisher,
            TimerAction(period=3.0, actions=[spawn_robot]),
        ]
    )
