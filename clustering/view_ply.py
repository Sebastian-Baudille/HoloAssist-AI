"""View a PLY file in Polyscope — no processing, just load and display.

Points are transformed from camera body frame to world frame using the camera
pose in sim_params.yaml so that the scene appears correctly oriented (Z = up).

Usage:
    python clustering/view_ply.py
    python clustering/view_ply.py ~/holoassist_pointclouds/default_4cubes_40mm_v001.ply
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import polyscope as ps
import yaml


DEFAULT_PARAMS = Path(__file__).parent.parent / "ros2_ws/src/holoassist_sim/config/sim_params.yaml"


def camera_to_world(points: np.ndarray, pose: list) -> np.ndarray:
    """Transform points from camera body frame to world frame.

    pose = [x, y, z, roll, pitch, yaw] from sim_params.yaml.
    Rotation convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    """
    x, y, z, roll, pitch, yaw = pose

    cr, sr = np.cos(roll),  np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw),   np.sin(yaw)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cr, -sr],
                   [0,  sr,  cr]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    R = Rz @ Ry @ Rx
    t = np.array([x, y, z])
    return (R @ points.T).T + t


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick PLY viewer for HoloAssist captures")
    parser.add_argument("path", nargs="?", default=None, help="Path to .ply file")
    parser.add_argument("--params", default=str(DEFAULT_PARAMS),
                        help="Path to sim_params.yaml (for camera pose)")
    args = parser.parse_args()

    if args.path is None:
        default_dir = Path.home() / "holoassist_pointclouds"
        plys = sorted(default_dir.glob("*.ply")) if default_dir.exists() else []
        if plys:
            args.path = str(plys[-1])
            print(f"Using most recent capture: {args.path}")
        else:
            args.path = str(Path(__file__).parent / "sample_data/default_4cubes_40mm_v001.ply")
            print(f"No captures found in ~/holoassist_pointclouds — using bundled sample")

    # Load point cloud
    pcd = o3d.io.read_point_cloud(args.path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors) if pcd.has_colors() else None
    print(f"{len(points):,} points  |  has_colors={pcd.has_colors()}")

    # Transform to world frame using camera pose from sim_params.yaml
    try:
        with open(args.params) as f:
            params = yaml.safe_load(f)
        pose = params["camera"]["pose"]
        points_world = camera_to_world(points, pose)
        print(f"Transformed to world frame using camera pose {pose}")
    except Exception as e:
        print(f"Warning: could not load camera pose ({e}), showing in camera frame")
        points_world = points

    print(f"X range: {points_world[:, 0].min():.3f} → {points_world[:, 0].max():.3f}")
    print(f"Y range: {points_world[:, 1].min():.3f} → {points_world[:, 1].max():.3f}")
    print(f"Z range: {points_world[:, 2].min():.3f} → {points_world[:, 2].max():.3f}")

    ps.init()
    ps.set_up_dir("z_up")
    ps.set_ground_plane_mode("shadow_only")

    cloud = ps.register_point_cloud("Captured scene", points_world, radius=0.0005)
    if colors is not None:
        cloud.add_color_quantity("RGB", colors, enabled=True)
    cloud.add_scalar_quantity("Height (world Z)", points_world[:, 2], enabled=False)

    ps.show()


if __name__ == "__main__":
    main()
