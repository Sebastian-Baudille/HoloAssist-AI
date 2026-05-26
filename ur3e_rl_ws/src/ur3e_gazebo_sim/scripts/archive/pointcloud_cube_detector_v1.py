#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
import open3d as o3d
import rclpy
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2

from ur3e_rl_env.constants import (
    TABLE_TOP_Z,
    WORKSPACE_X_MAX,
    WORKSPACE_X_MIN,
    WORKSPACE_Y_MAX,
    WORKSPACE_Y_MIN,
)


INPUT_POINTCLOUD_TOPIC = "/camera/depth/color/points"
DEBUG_POINTCLOUD_TOPIC = "/cube_detections/debug"
CUBE_POSE_TOPICS = ("/cube_0/pose", "/cube_1/pose", "/cube_2/pose", "/cube_3/pose")

CLUSTER_EPS_M = 0.015
CLUSTER_MIN_POINTS = 20
CLUSTER_SIZE_MIN = 200
CLUSTER_SIZE_MAX = 800
PLANE_DISTANCE_THRESHOLD_M = 0.005
PLANE_RANSAC_N = 3
PLANE_NUM_ITER = 100


class PointCloudCubeDetector(Node):
    def __init__(self) -> None:
        super().__init__("pointcloud_cube_detector")

        self.cube_publishers = [
            self.create_publisher(PoseStamped, topic, 10) for topic in CUBE_POSE_TOPICS
        ]
        self.debug_pub = self.create_publisher(PointCloud2, DEBUG_POINTCLOUD_TOPIC, 10)
        self.create_subscription(
            PointCloud2,
            INPUT_POINTCLOUD_TOPIC,
            self._pointcloud_cb,
            10,
        )
        self.get_logger().info(
            f"Listening on {INPUT_POINTCLOUD_TOPIC}; publishing {', '.join(CUBE_POSE_TOPICS)}"
        )

    def _pointcloud_cb(self, msg: PointCloud2) -> None:
        points_iter = pc2.read_points(
            msg,
            field_names=("x", "y", "z"),
            skip_nans=True,
        )
        points = np.asarray(list(points_iter), dtype=np.float32)
        if points.size == 0:
            return

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        min_bound = np.array([WORKSPACE_X_MIN, WORKSPACE_Y_MIN, TABLE_TOP_Z + 0.01], dtype=np.float64)
        max_bound = np.array([WORKSPACE_X_MAX, WORKSPACE_Y_MAX, TABLE_TOP_Z + 0.15], dtype=np.float64)
        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_bound, max_bound=max_bound)
        cropped = pcd.crop(bbox)
        if len(cropped.points) == 0:
            return

        _, inliers = cropped.segment_plane(
            distance_threshold=PLANE_DISTANCE_THRESHOLD_M,
            ransac_n=PLANE_RANSAC_N,
            num_iterations=PLANE_NUM_ITER,
        )
        objects_pcd = cropped.select_by_index(inliers, invert=True)
        if len(objects_pcd.points) == 0:
            return

        labels = np.asarray(
            objects_pcd.cluster_dbscan(
                eps=CLUSTER_EPS_M,
                min_points=CLUSTER_MIN_POINTS,
                print_progress=False,
            )
        )
        if labels.size == 0:
            return

        centroids: list[np.ndarray] = []
        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue
            indices = np.where(labels == label)[0]
            if indices.size < CLUSTER_SIZE_MIN or indices.size > CLUSTER_SIZE_MAX:
                continue
            cluster_points = np.asarray(objects_pcd.points)[indices]
            centroids.append(cluster_points.mean(axis=0))

        if not centroids:
            return

        centroids.sort(key=lambda c: float(np.linalg.norm(c)))
        for index, centroid in enumerate(centroids[:4]):
            stamped = PoseStamped()
            stamped.header = msg.header
            stamped.pose.position.x = float(centroid[0])
            stamped.pose.position.y = float(centroid[1])
            stamped.pose.position.z = float(centroid[2])
            stamped.pose.orientation.w = 1.0
            self.cube_publishers[index].publish(stamped)

        debug_points = np.asarray(objects_pcd.points, dtype=np.float32)
        debug_msg = pc2.create_cloud_xyz32(msg.header, debug_points.tolist())
        self.debug_pub.publish(debug_msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PointCloudCubeDetector()
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
