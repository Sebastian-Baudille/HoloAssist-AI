"""Temporal tracker for cube poses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    position: np.ndarray
    confidence: float
    missing_frames: int = 0
    age_frames: int = 0
    is_occluded: bool = False


class CubeTracker:
    def __init__(self, params: dict):
        self.p = params
        self.tracks: list[Track] = []
        self._next_id = 0

    def update(self, detections) -> list[Track]:
        matched_track_ids = set()
        matched_detection_ids = set()

        if self.tracks and detections:
            cost = np.zeros((len(self.tracks), len(detections)), dtype=np.float32)
            for ti, track in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    cost[ti, di] = float(np.linalg.norm(track.position - det.centroid))

            while np.isfinite(cost).any():
                ti, di = np.unravel_index(np.argmin(cost), cost.shape)
                min_dist = float(cost[ti, di])
                if min_dist > float(self.p["tracker_max_distance_m"]):
                    break

                track = self.tracks[ti]
                det = detections[di]
                track.position = det.centroid.copy()
                track.confidence = float(max(track.confidence, det.confidence))
                track.missing_frames = 0
                track.age_frames += 1
                track.is_occluded = bool(det.is_occluded)

                matched_track_ids.add(ti)
                matched_detection_ids.add(di)
                cost[ti, :] = np.inf
                cost[:, di] = np.inf

        for ti, track in enumerate(self.tracks):
            if ti in matched_track_ids:
                continue
            track.missing_frames += 1
            track.age_frames += 1
            track.is_occluded = True
            track.confidence *= float(self.p["tracker_confidence_decay"])

        for di, det in enumerate(detections):
            if di in matched_detection_ids:
                continue
            self.tracks.append(
                Track(
                    track_id=self._next_id,
                    position=det.centroid.copy(),
                    confidence=float(det.confidence),
                )
            )
            self._next_id += 1

        self.tracks = [
            t
            for t in self.tracks
            if t.missing_frames <= int(self.p["tracker_max_missing_frames"])
            and t.confidence >= float(self.p["tracker_min_publish_confidence"])
        ]
        self.tracks.sort(key=lambda t: float(np.linalg.norm(t.position)))
        return self.tracks[: int(self.p["max_cubes"])]

    def reset(self) -> None:
        self.tracks = []
