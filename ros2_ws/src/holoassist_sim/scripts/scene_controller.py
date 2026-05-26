#!/usr/bin/env python3
"""scene_controller.py — Manage cubes in a running Gazebo (Fortress) sim.

Exposes two ROS 2 services. Call them from the command line, an rqt plugin,
or (later) a button in our control GUI:

  /scene/randomize_cubes  — remove tracked cubes, spawn new ones with random
                            position / size / colour per current parameters
  /scene/reset            — remove tracked cubes, respawn the default layout
                            from sim_params.yaml

Settings are ROS 2 parameters editable live via `rqt_reconfigure`.  Run
`ros2 run rqt_reconfigure rqt_reconfigure` and dock the window next to RViz
to tweak the scene without restarting the sim.

Implementation notes
--------------------
We talk to Gazebo by shelling out to `ign service`.  It is the simplest path
that does not need extra Python bindings.  For each new cube we write an SDF
file to /tmp and pass its path via the EntityFactory `sdf_filename` field —
this avoids fighting with shell-escaping the SDF string inline.
"""
import os
import random
import subprocess
import tempfile
from pathlib import Path

import rclpy
import yaml
from rclpy.node import Node
from std_srvs.srv import Trigger


class SceneController(Node):
    def __init__(self):
        super().__init__("scene_controller")

        # Settings (edit via rqt_reconfigure)
        self.declare_parameter("world_name",      "table_cubes_world")
        self.declare_parameter("params_file",     "")
        self.declare_parameter("cube_count",      4)
        self.declare_parameter("cube_size_min",   0.03)
        self.declare_parameter("cube_size_max",   0.05)
        self.declare_parameter("x_min",          -0.15)
        self.declare_parameter("x_max",           0.15)
        self.declare_parameter("y_min",          -0.15)
        self.declare_parameter("y_max",           0.15)
        self.declare_parameter("randomize_yaw",   True)
        self.declare_parameter("randomize_color", True)

        # We need table_top_z to place cubes correctly. Compute from params_file.
        self._table_top_z = 0.50  # safe default; overwritten in _refresh_defaults
        self._default_cubes = []
        self._refresh_defaults()

        # Names of cubes we have spawned so we can clean them up later
        self._spawned = []

        # Services
        self.create_service(Trigger, "/scene/randomize_cubes", self._on_randomize)
        self.create_service(Trigger, "/scene/reset",           self._on_reset)

        self.get_logger().info(
            "scene_controller ready. Try: "
            "ros2 service call /scene/randomize_cubes std_srvs/srv/Trigger"
        )

    # ── service callbacks ────────────────────────────────────────────────

    def _on_randomize(self, request, response):
        try:
            self._remove_tracked()
            n = self.get_parameter("cube_count").value
            placed_xy = []  # track placed (x, y, size) for overlap check
            for i in range(n):
                self._spawn_random(f"cube_rand_{i:02d}", placed_xy)
            response.success = True
            response.message = f"Spawned {n} random cubes"
        except Exception as e:
            response.success = False
            response.message = f"randomize failed: {e}"
            self.get_logger().error(response.message)
        return response

    def _on_reset(self, request, response):
        try:
            self._remove_tracked()
            self._refresh_defaults()       # re-read params file in case it changed
            for cube in self._default_cubes:
                self._spawn(
                    name=cube["name"],
                    pose=cube["pose"],
                    size=cube["size"],
                    color=cube["color"],
                    mass=cube["mass"],
                )
            response.success = True
            response.message = f"Reset to {len(self._default_cubes)} default cubes"
        except Exception as e:
            response.success = False
            response.message = f"reset failed: {e}"
            self.get_logger().error(response.message)
        return response

    # ── scene helpers ─────────────────────────────────────────────────────

    def _refresh_defaults(self):
        """Reload sim_params.yaml so /scene/reset uses the latest defaults."""
        path = self.get_parameter("params_file").value
        if not path or not os.path.isfile(path):
            return
        with open(path) as f:
            params = yaml.safe_load(f)
        self._table_top_z = params["table"]["pose"][2] + params["table"]["size"][2] / 2
        self._default_cubes = params.get("cubes", [])

    def _spawn_random(self, name, placed_xy: list):
        p = lambda k: self.get_parameter(k).value
        size = round(random.uniform(p("cube_size_min"), p("cube_size_max")), 3)
        # Minimum centre-to-centre distance: cube diagonal to guarantee no overlap
        min_dist = size * 1.5

        # Retry until a non-overlapping position is found (max 50 attempts)
        for _ in range(50):
            x = round(random.uniform(p("x_min"), p("x_max")), 3)
            y = round(random.uniform(p("y_min"), p("y_max")), 3)
            if all(((x - px) ** 2 + (y - py) ** 2) ** 0.5 >= min_dist
                   for px, py, _ in placed_xy):
                break
        placed_xy.append((x, y, size))

        z = round(self._table_top_z + size / 2, 4)
        yaw = round(random.uniform(0, 2 * 3.14159), 3) if p("randomize_yaw") else 0.0
        if p("randomize_color"):
            color = [round(random.uniform(0.1, 0.95), 2) for _ in range(3)] + [1.0]
        else:
            color = [0.5, 0.5, 0.5, 1.0]

        self._spawn(name, pose=[x, y, z, 0.0, 0.0, yaw],
                    size=[size, size, size], color=color, mass=0.05)

    # ── Gazebo interaction (via `ign service`) ────────────────────────────

    def _spawn(self, name, pose, size, color, mass):
        sdf_path = self._write_cube_sdf(name, pose, size, color, mass)
        world = self.get_parameter("world_name").value
        subprocess.run([
            "ign", "service", "-s", f"/world/{world}/create",
            "--reqtype", "ignition.msgs.EntityFactory",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "1000",
            "--req", f'sdf_filename: "{sdf_path}"',
        ], check=True)
        self._spawned.append(name)

    def _remove_tracked(self):
        world = self.get_parameter("world_name").value
        for name in self._spawned:
            subprocess.run([
                "ign", "service", "-s", f"/world/{world}/remove",
                "--reqtype", "ignition.msgs.Entity",
                "--reptype", "ignition.msgs.Boolean",
                "--timeout", "1000",
                "--req", f'name: "{name}", type: MODEL',
            ], check=False)   # don't crash on already-removed
        self._spawned.clear()

    @staticmethod
    def _write_cube_sdf(name, pose, size, color, mass):
        sx, sy, sz = size
        r, g, b, a = color
        x, y, z, R, P, Y = pose
        # Solid cuboid inertia: I = m/12 * (a² + b²)
        ixx = mass / 12.0 * (sy * sy + sz * sz)
        iyy = mass / 12.0 * (sx * sx + sz * sz)
        izz = mass / 12.0 * (sx * sx + sy * sy)
        sdf = f"""<?xml version="1.0"?>
<sdf version="1.9">
  <model name="{name}">
    <pose>{x} {y} {z} {R} {P} {Y}</pose>
    <link name="link">
      <inertial>
        <mass>{mass}</mass>
        <inertia>
          <ixx>{ixx:.6e}</ixx><iyy>{iyy:.6e}</iyy><izz>{izz:.6e}</izz>
          <ixy>0</ixy><ixz>0</ixz><iyz>0</iyz>
        </inertia>
      </inertial>
      <collision name="collision">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>
      </collision>
      <visual name="visual">
        <geometry><box><size>{sx} {sy} {sz}</size></box></geometry>
        <material>
          <ambient>{r} {g} {b} {a}</ambient>
          <diffuse>{r} {g} {b} {a}</diffuse>
          <specular>0.2 0.2 0.2 1</specular>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""
        path = os.path.join(tempfile.gettempdir(), f"holoassist_{name}.sdf")
        with open(path, "w") as f:
            f.write(sdf)
        return path


def main():
    rclpy.init()
    node = SceneController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
