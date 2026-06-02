#!/usr/bin/env python3
"""dataset_capture.py — Automated dataset generation for cube detection training.

Orchestrates 60 scenes (50 train + 10 val), each with 2–4 randomly placed cubes.
For each scene:
  1. Sets cube_count parameter on the running scene_controller node
  2. Calls /scene/randomize_cubes to spawn cubes in Gazebo
  3. Waits for physics to settle (configurable)
  4. Reads actual settled poses from Gazebo via ign topic dynamic_pose/info
  5. Captures one PointCloud2 frame from /camera/points → saves as PLY
  6. Saves scene_NNNN.labels.json alongside the PLY with ground truth

Prerequisites (must be running before invoking this script):
  ros2 launch holoassist_sim sim.launch.py
  ros2 run holoassist_sim scene_controller --ros-args -p params_file:=<path/to/sim_params.yaml>

Usage:
  ros2 run holoassist_sim dataset_capture --params <path/to/sim_params.yaml>
  ros2 run holoassist_sim dataset_capture --params <yaml> --output ~/holoassist_dataset --scenes 60
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import yaml
from rcl_interfaces.msg import Parameter as RclParam
from rcl_interfaces.msg import ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_srvs.srv import Trigger


TRAIN_SCENES = 50
VAL_SCENES = 10
TOTAL_SCENES = TRAIN_SCENES + VAL_SCENES

SETTLE_WAIT_SEC = 1.0       # seconds to wait for physics after randomise
CLOUD_TIMEOUT_SEC = 10.0    # seconds to wait for a point cloud frame
SERVICE_TIMEOUT_SEC = 5.0   # seconds to wait for service response

CUBE_SIZE_MIN = 0.03
CUBE_SIZE_MAX = 0.05


# ── PLY writing (copied from save_pointcloud.py to avoid cross-imports) ──────

def _unpack_rgb(rgb_field: np.ndarray) -> np.ndarray:
    if rgb_field.dtype == np.float32:
        as_u32 = rgb_field.view(np.uint32)
    else:
        as_u32 = rgb_field.astype(np.uint32)
    r = ((as_u32 >> 16) & 0xFF).astype(np.uint8)
    g = ((as_u32 >> 8) & 0xFF).astype(np.uint8)
    b = (as_u32 & 0xFF).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


def _write_ply(path: str, xyz: np.ndarray, rgb: np.ndarray | None) -> None:
    n = xyz.shape[0]
    has_rgb = rgb is not None and rgb.shape[0] == n
    with open(path, "wb") as f:
        header = [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {n}",
            "property float x",
            "property float y",
            "property float z",
        ]
        if has_rgb:
            header += ["property uchar red", "property uchar green", "property uchar blue"]
        header.append("end_header")
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if has_rgb:
            buf = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("r", "u1"), ("g", "u1"), ("b", "u1")])
            buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
            buf["r"], buf["g"], buf["b"] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
        else:
            buf = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4")])
            buf["x"], buf["y"], buf["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        f.write(buf.tobytes())


# ── Ignition pose query ───────────────────────────────────────────────────────

def _query_gazebo_poses(world_name: str, cube_names: list[str]) -> dict[str, dict]:
    """Read actual settled cube poses from Gazebo via ign topic.

    Returns {name: {x, y, z, qx, qy, qz, qw}} for each requested cube name.
    Fields missing in the output default to 0 (qw defaults to 1).
    """
    try:
        result = subprocess.run(
            ["ign", "topic", "-e", "-n", "1",
             "-t", f"/world/{world_name}/dynamic_pose/info"],
            capture_output=True, text=True, timeout=8.0,
        )
        text = result.stdout
    except subprocess.TimeoutExpired:
        return {}

    poses: dict[str, dict] = {}
    # Split on pose-block boundaries
    blocks = re.split(r"\npose \{|\npose\{", "\n" + text)
    for block in blocks[1:]:
        name_m = re.search(r'name:\s*"([^"]+)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        if name not in cube_names:
            continue

        pos_m = re.search(r"position\s*\{([^}]+)\}", block)
        x = y = z = 0.0
        if pos_m:
            pt = pos_m.group(1)
            xm = re.search(r"\bx:\s*([-\d.e+]+)", pt)
            ym = re.search(r"\by:\s*([-\d.e+]+)", pt)
            zm = re.search(r"\bz:\s*([-\d.e+]+)", pt)
            x = float(xm.group(1)) if xm else 0.0
            y = float(ym.group(1)) if ym else 0.0
            z = float(zm.group(1)) if zm else 0.0

        ori_m = re.search(r"orientation\s*\{([^}]+)\}", block)
        qx = qy = qz = 0.0
        qw = 1.0
        if ori_m:
            ot = ori_m.group(1)
            qxm = re.search(r"\bx:\s*([-\d.e+]+)", ot)
            qym = re.search(r"\by:\s*([-\d.e+]+)", ot)
            qzm = re.search(r"\bz:\s*([-\d.e+]+)", ot)
            qwm = re.search(r"\bw:\s*([-\d.e+]+)", ot)
            qx = float(qxm.group(1)) if qxm else 0.0
            qy = float(qym.group(1)) if qym else 0.0
            qz = float(qzm.group(1)) if qzm else 0.0
            qw = float(qwm.group(1)) if qwm else 1.0

        poses[name] = {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}

    return poses


# ── ROS 2 node ────────────────────────────────────────────────────────────────

class DatasetCapture(Node):
    def __init__(self, params: dict, output_dir: str) -> None:
        super().__init__("dataset_capture")

        cam = params.get("camera", {})
        self._topic = f"{cam.get('topic', '/camera')}/points"
        self._output_dir = Path(os.path.expanduser(output_dir))
        self._output_dir.mkdir(parents=True, exist_ok=True)

        self._table_top_z: float = (
            params["table"]["pose"][2] + params["table"]["size"][2] / 2
        )
        self._world_name: str = "table_cubes_world"

        # Point cloud subscription
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self._sub = self.create_subscription(PointCloud2, self._topic, self._on_cloud, qos)

        # Service client for scene_controller
        self._randomize_cli = self.create_client(Trigger, "/scene/randomize_cubes")

        # Direct parameter service client (Humble-compatible, no subprocess)
        self._set_params_cli = self.create_client(
            SetParameters, "/scene_controller/set_parameters"
        )

        # Thread-safe cloud capture
        self._cloud_lock = threading.Lock()
        self._cloud_event = threading.Event()
        self._latest_cloud: PointCloud2 | None = None

        self.get_logger().info(f"Subscribing to {self._topic}")
        self.get_logger().info(f"Output → {self._output_dir}")

    def _on_cloud(self, msg: PointCloud2) -> None:
        with self._cloud_lock:
            self._latest_cloud = msg
        self._cloud_event.set()

    def _wait_for_service(self) -> bool:
        self.get_logger().info("Waiting for /scene/randomize_cubes ...")
        if not self._randomize_cli.wait_for_service(timeout_sec=15.0):
            self.get_logger().error("scene_controller not available — is it running?")
            return False
        return True

    def _remove_default_cubes(self, params: dict) -> None:
        """Remove the default SDF cubes (cube_red etc.) that are always in the world."""
        world = self._world_name
        default_names = [c["name"] for c in params.get("cubes", [])]
        for name in default_names:
            subprocess.run([
                "ign", "service", "-s", f"/world/{world}/remove",
                "--reqtype", "ignition.msgs.Entity",
                "--reptype", "ignition.msgs.Boolean",
                "--timeout", "1000",
                "--req", f'name: "{name}", type: MODEL',
            ], capture_output=True)
        if default_names:
            self.get_logger().info(f"Removed default cubes: {default_names}")
            time.sleep(0.5)

    def _set_params(self, **kwargs) -> bool:
        """Set scene_controller parameters via direct service call (Humble-compatible)."""
        req = SetParameters.Request()
        for name, value in kwargs.items():
            p = RclParam()
            p.name = name
            p.value = ParameterValue()
            if isinstance(value, int):
                p.value.type = ParameterType.PARAMETER_INTEGER
                p.value.integer_value = value
            else:
                p.value.type = ParameterType.PARAMETER_DOUBLE
                p.value.double_value = float(value)
            req.parameters.append(p)

        future = self._set_params_cli.call_async(req)
        deadline = time.monotonic() + 10.0
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn("set_parameters timed out")
                return False
            time.sleep(0.02)
        return True

    def _set_fixed_size(self, size_m: float = 0.04) -> None:
        self._set_params(cube_size_min=size_m, cube_size_max=size_m)
        self.get_logger().info(f"Cube size fixed to {size_m} m")

    def _set_cube_count(self, n: int) -> None:
        self._set_params(cube_count=n)

    def _call_randomize(self) -> bool:
        req = Trigger.Request()
        future = self._randomize_cli.call_async(req)
        deadline = time.monotonic() + SERVICE_TIMEOUT_SEC
        while not future.done():
            if time.monotonic() > deadline:
                self.get_logger().warn("Randomize service timed out")
                return False
            time.sleep(0.05)
        return future.result().success

    def _capture_cloud(self) -> PointCloud2 | None:
        self._cloud_event.clear()
        with self._cloud_lock:
            self._latest_cloud = None
        if not self._cloud_event.wait(timeout=CLOUD_TIMEOUT_SEC):
            self.get_logger().error("Timed out waiting for point cloud")
            return None
        with self._cloud_lock:
            return self._latest_cloud

    def _cloud_to_arrays(self, msg: PointCloud2):
        field_names = [f.name for f in msg.fields]
        wanted = ["x", "y", "z"] + (["rgb"] if "rgb" in field_names else [])
        records = pc2.read_points(msg, field_names=wanted, skip_nans=True)
        records = np.asarray(list(records)) if not isinstance(records, np.ndarray) else records
        if records.size == 0:
            return None, None
        xyz = np.stack([records["x"], records["y"], records["z"]], axis=1).astype(np.float32)
        rgb = _unpack_rgb(records["rgb"]) if "rgb" in wanted else None
        finite = np.isfinite(xyz).all(axis=1)
        xyz = xyz[finite]
        rgb = rgb[finite] if rgb is not None else None
        return xyz, rgb

    def run(self, total_scenes: int, train_scenes: int, params: dict) -> None:
        import random

        if not self._wait_for_service():
            return

        # One-time setup: fix cube size and remove default SDF cubes
        self._set_fixed_size(0.04)
        self._remove_default_cubes(params)

        self.get_logger().info(
            f"Starting dataset capture: {train_scenes} train + "
            f"{total_scenes - train_scenes} val scenes"
        )

        success_count = 0
        for scene_idx in range(1, total_scenes + 1):
            split = "train" if scene_idx <= train_scenes else "val"
            cube_count = random.randint(2, 4)
            scene_id = f"scene_{scene_idx:04d}"

            self.get_logger().info(
                f"[{scene_idx}/{total_scenes}] {scene_id} | {split} | {cube_count} cubes"
            )

            # 1. Set cube count and randomize scene
            self._set_cube_count(cube_count)
            if not self._call_randomize():
                self.get_logger().warn(f"  Randomize failed, skipping {scene_id}")
                continue

            # 2. Wait for physics to settle
            time.sleep(SETTLE_WAIT_SEC)

            # 3. Read actual settled poses from Gazebo
            cube_names = [f"cube_rand_{i:02d}" for i in range(cube_count)]
            poses = _query_gazebo_poses(self._world_name, cube_names)
            if len(poses) < cube_count:
                self.get_logger().warn(
                    f"  Only got {len(poses)}/{cube_count} poses from Gazebo, skipping"
                )
                continue

            # 4. Capture point cloud
            cloud = self._capture_cloud()
            if cloud is None:
                continue
            xyz, rgb = self._cloud_to_arrays(cloud)
            if xyz is None or len(xyz) == 0:
                self.get_logger().warn(f"  Empty cloud, skipping {scene_id}")
                continue

            # 5. Save PLY
            ply_path = self._output_dir / f"{scene_id}.ply"
            _write_ply(str(ply_path), xyz, rgb)

            # 6. Save labels.json — use settled poses, estimate size from z
            cubes_label = []
            for name in cube_names:
                p = poses[name]
                # Back-calculate size from settled z: z = table_top + size/2
                size = round((p["z"] - self._table_top_z) * 2, 4)
                size = float(np.clip(size, CUBE_SIZE_MIN, CUBE_SIZE_MAX))
                cubes_label.append({
                    "name": name,
                    "position": {"x": p["x"], "y": p["y"], "z": p["z"]},
                    "orientation": {
                        "x": p["qx"], "y": p["qy"],
                        "z": p["qz"], "w": p["qw"],
                    },
                    "size_m": size,
                })

            labels = {
                "scene_id": scene_id,
                "split": split,
                "cube_count": cube_count,
                "world_name": self._world_name,
                "table_top_z": self._table_top_z,
                "cubes": cubes_label,
            }
            labels_path = self._output_dir / f"{scene_id}.labels.json"
            with open(labels_path, "w") as f:
                json.dump(labels, f, indent=2)

            success_count += 1
            self.get_logger().info(
                f"  Saved {ply_path.name} + labels ({len(xyz)} pts)"
            )

        self.get_logger().info(
            f"\nDone. {success_count}/{total_scenes} scenes captured → {self._output_dir}"
        )
        self.get_logger().info(
            "Run: python3 clustering/verify_detection.py to evaluate accuracy."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture labelled dataset from Gazebo sim")
    parser.add_argument("--params", required=True, help="Path to sim_params.yaml")
    parser.add_argument(
        "--output", default="~/holoassist_dataset",
        help="Output directory (default: ~/holoassist_dataset)",
    )
    parser.add_argument(
        "--scenes", type=int, default=TOTAL_SCENES,
        help=f"Total scenes to capture (default: {TOTAL_SCENES})",
    )
    parser.add_argument(
        "--train-scenes", type=int, default=TRAIN_SCENES,
        help=f"Scenes labelled as train (default: {TRAIN_SCENES}, rest = val)",
    )
    args, ros_args = parser.parse_known_args()

    with open(args.params) as f:
        params = yaml.safe_load(f)

    rclpy.init(args=ros_args)
    node = DatasetCapture(params, args.output)

    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    try:
        node.run(total_scenes=args.scenes, train_scenes=args.train_scenes, params=params)
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        rclpy.shutdown()
        spin_thread.join(timeout=2.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
