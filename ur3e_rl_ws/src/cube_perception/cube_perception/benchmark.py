"""Measure detection consistency and confidence for cube perception."""
from __future__ import annotations

import argparse
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32


class PerceptionBenchmark(Node):
    def __init__(self, n_samples: int):
        super().__init__("perception_benchmark")
        self.n_samples = n_samples
        self.samples = {idx: [] for idx in range(4)}
        self.confidences = {idx: [] for idx in range(4)}

        for idx in range(4):
            self.create_subscription(
                PoseStamped,
                f"/cube_{idx}/pose",
                lambda msg, cube_idx=idx: self._pose_cb(msg, cube_idx),
                10,
            )
            self.create_subscription(
                Float32,
                f"/cube_{idx}/confidence",
                lambda msg, cube_idx=idx: self._conf_cb(msg, cube_idx),
                10,
            )

        self.get_logger().info(f"Collecting {n_samples} samples per cube...")

    def _pose_cb(self, msg: PoseStamped, idx: int) -> None:
        if len(self.samples[idx]) >= self.n_samples:
            return
        p = msg.pose.position
        self.samples[idx].append([p.x, p.y, p.z])

    def _conf_cb(self, msg: Float32, idx: int) -> None:
        if len(self.confidences[idx]) >= self.n_samples:
            return
        self.confidences[idx].append(float(msg.data))

    def is_done(self) -> bool:
        return all(len(v) >= self.n_samples for v in self.samples.values())

    def report(self) -> None:
        print("\n=== Perception Benchmark Results ===\n")
        for idx in range(4):
            pts = np.array(self.samples[idx], dtype=np.float64)
            confs = np.array(self.confidences[idx], dtype=np.float64)
            if pts.size == 0:
                print(f"cube_{idx}: NO DATA")
                continue

            mean_pos = pts.mean(axis=0)
            std_pos = pts.std(axis=0) * 1000.0
            mean_conf = float(confs.mean()) if confs.size > 0 else float("nan")

            print(f"cube_{idx}:")
            print(
                f"  Mean position:   x={mean_pos[0]:.4f}  "
                f"y={mean_pos[1]:.4f}  z={mean_pos[2]:.4f}"
            )
            print(
                f"  Std deviation:   x={std_pos[0]:.1f}mm  "
                f"y={std_pos[1]:.1f}mm  z={std_pos[2]:.1f}mm"
            )
            print(f"  Mean confidence: {mean_conf:.2f}")
            print(f"  Samples:         {len(pts)}/{self.n_samples}\n")

        print("Target: std deviation < 3mm for reliable pick-and-place")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    args = parser.parse_args()

    rclpy.init()
    node = PerceptionBenchmark(args.n_samples)

    start = time.time()
    try:
        while rclpy.ok() and not node.is_done():
            rclpy.spin_once(node, timeout_sec=0.1)
            if time.time() - start > 60.0:
                print("Timeout - not all cubes detected")
                break
    except (KeyboardInterrupt, ExternalShutdownException):
        pass

    node.report()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
