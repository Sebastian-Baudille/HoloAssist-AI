"""RViz marker helpers for cube perception."""
from __future__ import annotations

from visualization_msgs.msg import Marker, MarkerArray


class CubeVisualiser:
    def __init__(self, world_frame: str, cube_edge_m: float):
        self.world_frame = world_frame
        self.cube_edge_m = float(cube_edge_m)

    def build_markers(self, tracks, stamp, max_cubes: int) -> MarkerArray:
        arr = MarkerArray()

        for idx, track in enumerate(tracks):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.world_frame
            marker.ns = "cube_detections"
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = float(track.position[0])
            marker.pose.position.y = float(track.position[1])
            marker.pose.position.z = float(track.position[2])
            marker.pose.orientation.w = 1.0
            marker.scale.x = self.cube_edge_m
            marker.scale.y = self.cube_edge_m
            marker.scale.z = self.cube_edge_m
            marker.color.a = 1.0
            if track.confidence > 0.7:
                marker.color.r = 0.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            elif track.confidence > 0.4:
                marker.color.r = 1.0
                marker.color.g = 1.0
                marker.color.b = 0.0
            else:
                marker.color.r = 1.0
                marker.color.g = 0.0
                marker.color.b = 0.0
            arr.markers.append(marker)

        for idx in range(len(tracks), int(max_cubes)):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = self.world_frame
            marker.ns = "cube_detections"
            marker.id = idx
            marker.action = Marker.DELETE
            arr.markers.append(marker)

        return arr
