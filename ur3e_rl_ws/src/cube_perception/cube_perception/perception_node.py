#!/usr/bin/env python3
"""Main ROS 2 node for cube detection and tracking."""
from __future__ import annotations

import numpy as np
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
import tf2_ros
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSPresetProfiles
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from ur3e_rl_env.constants import (
    TABLE_TOP_Z,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
    WORKSPACE_Z_MAX,
    WORKSPACE_Z_MIN,
)

from .detector import CubeDetector
from .tracker import CubeTracker
from .visualiser import CubeVisualiser


OUTPUT_CUBE_COUNT = 4


def quaternion_to_rotation_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Convert quaternion to a 3x3 rotation matrix."""
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class PerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("cube_perception")

        self.declare_parameters(
            "",
            [
                ("input_topic", "/camera/depth/color/points"),
                ("camera_frame", "camera_link"),
                ("world_frame", "base_link"),
                ("workspace_x_min", WORKSPACE_X_MIN),
                ("workspace_x_max", WORKSPACE_X_MAX),
                ("workspace_y_min", WORKSPACE_Y_MIN),
                ("workspace_y_max", WORKSPACE_Y_MAX),
                ("table_top_z", TABLE_TOP_Z),
                ("workspace_z_min", WORKSPACE_Z_MIN),
                ("workspace_z_max", WORKSPACE_Z_MAX),
                ("cube_edge_m", 0.040),
                ("plane_distance_threshold_m", 0.005),
                ("plane_ransac_n", 3),
                ("plane_num_iterations", 100),
                ("cluster_eps_m", 0.015),
                ("cluster_min_points", 20),
                ("cluster_size_min", 50),
                ("cluster_size_max", 1200),
                ("full_cube_points_min", 300),
                ("full_cube_points_max", 900),
                ("occluded_cube_points_min", 50),
                ("max_cubes", OUTPUT_CUBE_COUNT),
                ("tracker_max_distance_m", 0.08),
                ("tracker_max_missing_frames", 15),
                ("tracker_confidence_decay", 0.85),
                ("tracker_min_publish_confidence", 0.2),
                ("publish_rate_hz", 10.0),
                ("publish_debug_cloud", True),
                ("publish_markers", True),
            ],
        )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.camera_frame = str(self.get_parameter("camera_frame").value)
        self.world_frame = str(self.get_parameter("world_frame").value)

        self.detector_params = {
            "workspace_x_min": float(self.get_parameter("workspace_x_min").value),
            "workspace_x_max": float(self.get_parameter("workspace_x_max").value),
            "workspace_y_min": float(self.get_parameter("workspace_y_min").value),
            "workspace_y_max": float(self.get_parameter("workspace_y_max").value),
            "workspace_z_min": float(self.get_parameter("workspace_z_min").value),
            "workspace_z_max": float(self.get_parameter("workspace_z_max").value),
            "plane_distance_threshold_m": float(
                self.get_parameter("plane_distance_threshold_m").value
            ),
            "plane_ransac_n": int(self.get_parameter("plane_ransac_n").value),
            "plane_num_iterations": int(self.get_parameter("plane_num_iterations").value),
            "cluster_eps_m": float(self.get_parameter("cluster_eps_m").value),
            "cluster_min_points": int(self.get_parameter("cluster_min_points").value),
            "cluster_size_min": int(self.get_parameter("cluster_size_min").value),
            "cluster_size_max": int(self.get_parameter("cluster_size_max").value),
            "full_cube_points_min": int(self.get_parameter("full_cube_points_min").value),
            "full_cube_points_max": int(self.get_parameter("full_cube_points_max").value),
            "occluded_cube_points_min": int(self.get_parameter("occluded_cube_points_min").value),
            "max_cubes": int(self.get_parameter("max_cubes").value),
            "tracker_max_distance_m": float(
                self.get_parameter("tracker_max_distance_m").value
            ),
            "tracker_max_missing_frames": int(
                self.get_parameter("tracker_max_missing_frames").value
            ),
            "tracker_confidence_decay": float(
                self.get_parameter("tracker_confidence_decay").value
            ),
            "tracker_min_publish_confidence": float(
                self.get_parameter("tracker_min_publish_confidence").value
            ),
        }

        self.detector = CubeDetector(self.detector_params)
        self.tracker = CubeTracker(self.detector_params)
        self.visualiser = CubeVisualiser(
            self.world_frame,
            float(self.get_parameter("cube_edge_m").value),
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pose_pubs = [
            self.create_publisher(PoseStamped, f"/cube_{idx}/pose", 10)
            for idx in range(OUTPUT_CUBE_COUNT)
        ]
        self.conf_pubs = [
            self.create_publisher(Float32, f"/cube_{idx}/confidence", 10)
            for idx in range(OUTPUT_CUBE_COUNT)
        ]
        self.debug_pub = self.create_publisher(PointCloud2, "/cube_detections/debug", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/cube_detections/markers", 10)

        self.latest_tracks = []
        self.latest_object_points = np.empty((0, 3), dtype=np.float32)

        self.create_subscription(
            PointCloud2,
            self.input_topic,
            self._cloud_cb,
            QoSPresetProfiles.SENSOR_DATA.value,
        )

        publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        timer_period = 1.0 / max(publish_rate_hz, 1e-3)
        self.create_timer(timer_period, self._publish_outputs)

        self.get_logger().info(
            f"cube_perception ready. Listening on {self.input_topic}, "
            f"transforming {self.camera_frame} -> {self.world_frame}"
        )

    def _cloud_cb(self, msg: PointCloud2) -> None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                msg.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.1),
            )
        except Exception as exc:  # noqa: BLE001
            self.get_logger().warn(
                f"TF lookup failed for {msg.header.frame_id} -> {self.world_frame}: {exc}",
                throttle_duration_sec=5.0,
            )
            self.latest_tracks = self.tracker.update([])
            self.latest_object_points = np.empty((0, 3), dtype=np.float32)
            return

        points_cam = np.asarray(
            list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float32,
        )
        if points_cam.shape[0] < 10:
            self.latest_tracks = self.tracker.update([])
            self.latest_object_points = np.empty((0, 3), dtype=np.float32)
            return

        t = transform.transform.translation
        q = transform.transform.rotation
        rotation = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
        translation = np.array([t.x, t.y, t.z], dtype=np.float64)

        points_world = (rotation @ points_cam.T).T + translation
        points_world = points_world.astype(np.float32, copy=False)

        object_points = self.detector.extract_object_points(points_world)
        detections = self.detector.detect_from_object_points(object_points)
        self.latest_tracks = self.tracker.update(detections)
        self.latest_object_points = object_points

    def _publish_outputs(self) -> None:
        now = self.get_clock().now().to_msg()

        for idx in range(OUTPUT_CUBE_COUNT):
            if idx >= len(self.latest_tracks):
                continue

            track = self.latest_tracks[idx]
            pose = PoseStamped()
            pose.header.stamp = now
            pose.header.frame_id = self.world_frame
            pose.pose.position.x = float(track.position[0])
            pose.pose.position.y = float(track.position[1])
            pose.pose.position.z = float(track.position[2])
            pose.pose.orientation.w = 1.0
            self.pose_pubs[idx].publish(pose)

            confidence = Float32()
            confidence.data = float(track.confidence)
            self.conf_pubs[idx].publish(confidence)

        if bool(self.get_parameter("publish_debug_cloud").value):
            debug_msg = pc2.create_cloud_xyz32(
                header=self._header(now),
                points=self.latest_object_points.tolist(),
            )
            self.debug_pub.publish(debug_msg)

        if bool(self.get_parameter("publish_markers").value):
            marker_array = self.visualiser.build_markers(
                tracks=self.latest_tracks,
                stamp=now,
                max_cubes=OUTPUT_CUBE_COUNT,
            )
            self.marker_pub.publish(marker_array)

    def _header(self, stamp):
        hdr = Header()
        hdr.stamp = stamp
        hdr.frame_id = self.world_frame
        return hdr


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PerceptionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
