"""Geometric cube detection pipeline."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import open3d as o3d


@dataclass
class DetectionResult:
    centroid: np.ndarray
    n_points: int
    confidence: float
    bbox_volume: float
    is_occluded: bool


class CubeDetector:
    def __init__(self, params: dict):
        self.p = params

    def _workspace_bbox(self) -> o3d.geometry.AxisAlignedBoundingBox:
        min_b = np.array(
            [
                self.p["workspace_x_min"],
                self.p["workspace_y_min"],
                self.p["workspace_z_min"],
            ],
            dtype=np.float64,
        )
        max_b = np.array(
            [
                self.p["workspace_x_max"],
                self.p["workspace_y_max"],
                self.p["workspace_z_max"],
            ],
            dtype=np.float64,
        )
        return o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)

    def extract_object_points(self, points_world: np.ndarray) -> np.ndarray:
        """Return workspace-cropped points with the dominant plane removed."""
        if points_world.ndim != 2 or points_world.shape[1] != 3 or points_world.shape[0] < 10:
            return np.empty((0, 3), dtype=np.float32)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_world.astype(np.float64, copy=False))

        cropped = pcd.crop(self._workspace_bbox())
        if len(cropped.points) < 10:
            return np.empty((0, 3), dtype=np.float32)

        try:
            _, inliers = cropped.segment_plane(
                distance_threshold=self.p["plane_distance_threshold_m"],
                ransac_n=int(self.p["plane_ransac_n"]),
                num_iterations=int(self.p["plane_num_iterations"]),
            )
        except RuntimeError:
            return np.empty((0, 3), dtype=np.float32)

        objects_pcd = cropped.select_by_index(inliers, invert=True)
        if len(objects_pcd.points) < 10:
            return np.empty((0, 3), dtype=np.float32)

        return np.asarray(objects_pcd.points, dtype=np.float32)

    def detect_from_object_points(self, object_points: np.ndarray) -> list[DetectionResult]:
        """Run DBSCAN + cube filtering on plane-removed object points."""
        if object_points.ndim != 2 or object_points.shape[1] != 3 or object_points.shape[0] < 10:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(object_points.astype(np.float64, copy=False))

        labels = np.asarray(
            pcd.cluster_dbscan(
                eps=float(self.p["cluster_eps_m"]),
                min_points=int(self.p["cluster_min_points"]),
                print_progress=False,
            )
        )
        if labels.size == 0:
            return []

        results: list[DetectionResult] = []
        full_min = int(self.p["full_cube_points_min"])
        full_max = int(self.p["full_cube_points_max"])
        occluded_min = int(self.p["occluded_cube_points_min"])

        for label in sorted(set(labels.tolist())):
            if label < 0:
                continue

            idx = np.where(labels == label)[0]
            n_points = int(idx.size)
            if (
                n_points < int(self.p["cluster_size_min"])
                or n_points > int(self.p["cluster_size_max"])
            ):
                continue
            if n_points < occluded_min:
                continue

            cluster_pts = object_points[idx]
            centroid = cluster_pts.mean(axis=0)

            cluster_pcd = o3d.geometry.PointCloud()
            cluster_pcd.points = o3d.utility.Vector3dVector(cluster_pts.astype(np.float64))
            bb = cluster_pcd.get_axis_aligned_bounding_box()
            bbox_volume = float(np.prod(bb.get_extent()))

            if n_points >= full_min:
                confidence = min(1.0, n_points / max(full_max, 1))
                is_occluded = False
            else:
                confidence = 0.4 + 0.4 * (n_points / max(full_min, 1))
                is_occluded = True

            results.append(
                DetectionResult(
                    centroid=centroid.astype(np.float32),
                    n_points=n_points,
                    confidence=float(np.clip(confidence, 0.0, 1.0)),
                    bbox_volume=bbox_volume,
                    is_occluded=is_occluded,
                )
            )

        results.sort(key=lambda r: float(np.linalg.norm(r.centroid)))
        return results[: int(self.p["max_cubes"])]

    def detect(self, points_world: np.ndarray) -> list[DetectionResult]:
        """Run full detection pipeline on world-frame point cloud."""
        object_points = self.extract_object_points(points_world)
        return self.detect_from_object_points(object_points)
