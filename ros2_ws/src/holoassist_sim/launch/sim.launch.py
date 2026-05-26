"""Launch the holoassist_sim Gazebo + RViz scene.

Pipeline:
  1. Read config/sim_params.yaml.
  2. Render worlds/table_cubes.sdf.jinja2 -> $cache_dir/table_cubes.sdf.
  3. Render config/ros_gz_bridge.yaml.jinja2 -> $cache_dir/ros_gz_bridge.yaml.
  4. Spawn gz sim with the rendered world.
  5. Spawn ros_gz_bridge with the rendered bridge config.
  6. Publish static TF: map -> <camera>_link -> <camera>_optical.
  7. Spawn RViz with the bundled config.

Override the params file with:  ros2 launch holoassist_sim sim.launch.py params_file:=/abs/path.yaml
"""
import os
import subprocess
import sys
import tempfile

import yaml
from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


PKG = "holoassist_sim"


def _render(params_path: str, template_path: str, output_path: str) -> None:
    """Render a Jinja2 template using render_world.py from the installed package."""
    # ament_index gives the install prefix (e.g. .../install/holoassist_sim).
    # The script is installed to <prefix>/lib/<pkg>/render_world.py by CMakeLists.
    script = os.path.join(get_package_prefix(PKG), "lib", PKG, "render_world.py")
    if not os.path.isfile(script):
        raise FileNotFoundError(
            f"render_world.py not found at {script}.\n"
            "Run:  colcon build --packages-select holoassist_sim --symlink-install"
        )
    # Use the same interpreter that launched ros2 to guarantee jinja2/yaml are available.
    subprocess.check_call([sys.executable, script, params_path, template_path, output_path])


def _launch_setup(context, *args, **kwargs):
    pkg_share = get_package_share_directory(PKG)

    params_file = LaunchConfiguration("params_file").perform(context)
    if not params_file:
        params_file = os.path.join(pkg_share, "config", "sim_params.yaml")

    with open(params_file, "r") as f:
        params = yaml.safe_load(f)

    cache_dir = tempfile.mkdtemp(prefix="holoassist_sim_")
    world_sdf = os.path.join(cache_dir, "table_cubes.sdf")
    bridge_yaml = os.path.join(cache_dir, "ros_gz_bridge.yaml")

    _render(
        params_file,
        os.path.join(pkg_share, "worlds", "table_cubes.sdf.jinja2"),
        world_sdf,
    )
    _render(
        params_file,
        os.path.join(pkg_share, "config", "ros_gz_bridge.yaml.jinja2"),
        bridge_yaml,
    )

    cam = params["camera"]
    cx, cy, cz, croll, cpitch, cyaw = cam["pose"]
    cam_name = cam["name"]
    use_imu = bool(cam.get("imu", {}).get("enabled", False))

    # Gazebo Sim (Fortress) via ros_gz_sim launch wrapper.
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
        ),
        launch_arguments={"gz_args": f"-r -v 3 {world_sdf}"}.items(),
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"config_file": bridge_yaml, "use_sim_time": True}],
        output="screen",
    )

    # Ignition Fortress scopes the sensor frame as "<model>/<link>/<sensor_name>".
    # The rgbd sensor is mounted directly on d435i_link (sensor name "rgbd"),
    # so the published frame_id is "<cam_name>/<cam_name>_link/rgbd".
    gz_sensor_frame = f"{cam_name}/{cam_name}_link/rgbd"

    # map -> sensor frame.  The body-frame pose directly aims the camera at
    # the table (pitch tilts +X downward toward the scene).  Since the sensor
    # sits on d435i_link with no additional sub-link rotation, one TF is enough.
    tf_map_to_link = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_map_to_camera_link",
        arguments=[
            "--frame-id", "map",
            "--child-frame-id", f"{cam_name}_link",
            "--x", str(cx), "--y", str(cy), "--z", str(cz),
            "--roll", str(croll), "--pitch", str(cpitch), "--yaw", str(cyaw),
        ],
    )
    # d435i_link -> sensor scoped frame (identity — same physical location).
    tf_link_to_sensor = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="tf_camera_link_to_sensor",
        arguments=[
            "--frame-id", f"{cam_name}_link",
            "--child-frame-id", gz_sensor_frame,
            "--x", "0", "--y", "0", "--z", "0",
            "--roll", "0", "--pitch", "0", "--yaw", "0",
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", os.path.join(pkg_share, "config", "rviz.rviz")],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )

    # Owns the cubes in the running sim. Exposes /scene/randomize_cubes and
    # /scene/reset.  Parameters editable live via rqt_reconfigure.
    scene_controller = Node(
        package=PKG,
        executable="scene_controller.py",
        name="scene_controller",
        parameters=[{
            "world_name":  "table_cubes_world",
            "params_file": params_file,
        }],
        output="screen",
    )

    # Sliders / toggles for scene_controller's parameters. Dock the window
    # next to RViz to edit the scene without restarting the sim.
    rqt = Node(
        package="rqt_reconfigure",
        executable="rqt_reconfigure",
        name="rqt_reconfigure",
        output="screen",
    )

    actions = [gz_sim, bridge, tf_map_to_link, tf_link_to_sensor, rviz, scene_controller, rqt]
    # IMU has no influence on the launch graph beyond the bridge entry, which
    # is already templated. Nothing extra to do here.
    _ = use_imu
    return actions


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "params_file",
            default_value="",
            description="Path to sim_params.yaml. Empty = use bundled default.",
        ),
        OpaqueFunction(function=_launch_setup),
    ])
