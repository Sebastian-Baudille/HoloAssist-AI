from __future__ import annotations

from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _static_tf_from_calibration(context):
    publish_tf = LaunchConfiguration("publish_static_tf").perform(context).lower() == "true"
    if not publish_tf:
        return []

    calibration_file = Path(LaunchConfiguration("calibration_file").perform(context)).expanduser()
    if not calibration_file.exists():
        return [LogInfo(msg=f"[cube_perception] Calibration missing: {calibration_file}. Skipping TF.")]

    try:
        data = yaml.safe_load(calibration_file.read_text())
        transform = data["transform"]
        t = transform["translation"]
        r = transform["rotation"]
        base_frame = data["parameters"].get("robot_base_frame", "base_link")
        camera_frame = data["parameters"].get("tracking_base_frame", "camera_link")
        args = [
            str(t["x"]),
            str(t["y"]),
            str(t["z"]),
            str(r["x"]),
            str(r["y"]),
            str(r["z"]),
            str(r["w"]),
            str(base_frame),
            str(camera_frame),
        ]
    except Exception as exc:  # noqa: BLE001
        return [
            LogInfo(
                msg=f"[cube_perception] Failed parsing calibration {calibration_file}: {exc}. Skipping TF."
            )
        ]

    return [
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="camera_tf_from_calibration",
            arguments=args,
            output="screen",
        )
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="false"),
            DeclareLaunchArgument("start_camera", default_value="true"),
            DeclareLaunchArgument("publish_static_tf", default_value="true"),
            DeclareLaunchArgument(
                "calibration_file",
                default_value="~/.ros2/easy_handeye2/calibrations/holoassist_calibration.calib",
            ),
            OpaqueFunction(function=_static_tf_from_calibration),
            Node(
                package="realsense2_camera",
                executable="realsense2_camera_node",
                name="camera",
                condition=IfCondition(LaunchConfiguration("start_camera")),
                parameters=[
                    {
                        "enable_depth": True,
                        "enable_color": True,
                        "enable_pointcloud": True,
                        "pointcloud.enable": True,
                        "align_depth.enable": True,
                    }
                ],
                output="screen",
            ),
            Node(
                package="cube_perception",
                executable="perception_node",
                name="cube_perception",
                parameters=[
                    PathJoinSubstitution(
                        [FindPackageShare("cube_perception"), "config", "params.yaml"]
                    )
                ],
                output="screen",
            ),
        ]
    )
