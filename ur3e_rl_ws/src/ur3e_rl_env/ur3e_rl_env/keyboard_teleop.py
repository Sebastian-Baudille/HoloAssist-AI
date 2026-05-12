"""Keyboard teleoperation for the UR3e in Gazebo.

Sends joint trajectory commands via the same interface the RL env uses.
Designed for testing the demo recording pipeline without real hardware
or the Unity XR teleop.

Usage:
    ros2 run ur3e_rl_env keyboard_teleop

Controls:
    1-6       Select joint (1=shoulder_pan ... 6=wrist_3)
    w / Up    Jog selected joint +
    s / Down  Jog selected joint -
    a / Left  Previous joint
    d / Right Next joint
    h         Home position (all zeros)
    + / =     Increase step size
    - / _     Decrease step size
    q         Quit
"""

from __future__ import annotations

import select
import sys
import termios
import tty

import rclpy
from rclpy.executors import ExternalShutdownException

from ur3e_rl_env.ros_interface import RosInterfaceNode, UR3E_JOINT_NAMES

JOINT_SHORT = ["pan", "lift", "elbow", "wr1", "wr2", "wr3"]
DEFAULT_STEP = 0.1
MIN_STEP = 0.005
MAX_STEP = 0.10

BANNER = """\
─────────────────────────────────────
  Keyboard Teleop — UR3e
─────────────────────────────────────
  1-6       Select joint
  w / ↑     Jog +        s / ↓  Jog -
  a / ←     Prev joint   d / →  Next
  h         Home         + / -  Step
  q         Quit
─────────────────────────────────────"""


def _read_key(timeout: float = 0.05) -> str | None:
    if not select.select([sys.stdin], [], [], timeout)[0]:
        return None
    ch = sys.stdin.read(1)
    if ch == "\x1b":
        if select.select([sys.stdin], [], [], 0.02)[0]:
            ch2 = sys.stdin.read(1)
            if ch2 == "[" and select.select([sys.stdin], [], [], 0.02)[0]:
                ch3 = sys.stdin.read(1)
                return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(
                    ch3
                )
        return None
    return ch


class KeyboardTeleop:
    def __init__(self) -> None:
        self.ros = RosInterfaceNode(node_name="keyboard_teleop")
        self.selected = 0
        self.step = DEFAULT_STEP

    def run(self) -> None:
        print("Waiting for /joint_states ...")
        if not self.ros.wait_for_joint_state(timeout_sec=30.0):
            print("Timed out — is the Gazebo sim running?")
            return

        print(BANNER)
        self._print_state()

        old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())

        try:
            while rclpy.ok():
                rclpy.spin_once(self.ros, timeout_sec=0.01)
                key = _read_key(timeout=0.05)
                if key is not None and self._handle(key):
                    break
        except KeyboardInterrupt:
            pass
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
            print("\n")

    def _handle(self, key: str) -> bool:
        if key == "q":
            return True
        if key in "123456":
            self.selected = int(key) - 1
        elif key in ("a", "LEFT"):
            self.selected = (self.selected - 1) % 6
        elif key in ("d", "RIGHT"):
            self.selected = (self.selected + 1) % 6
        elif key in ("w", "UP"):
            self._jog(self.step)
        elif key in ("s", "DOWN"):
            self._jog(-self.step)
        elif key in ("+", "="):
            self.step = min(self.step + 0.005, MAX_STEP)
        elif key in ("-", "_"):
            self.step = max(self.step - 0.005, MIN_STEP)
        elif key == "h":
            import math
            self.ros.send_joint_target([0.0, -math.pi / 2, 0.0, -math.pi / 2, 0.0, 0.0], duration_sec=1.0)
        else:
            return False
        self._print_state()
        return False

    def _jog(self, delta: float) -> None:
        positions = [
            self.ros.joint_positions_by_name.get(n, 0.0) for n in UR3E_JOINT_NAMES
        ]
        positions[self.selected] += delta
        self.ros.send_joint_target(positions, duration_sec=0.5)

    def _print_state(self) -> None:
        parts = []
        for i, short in enumerate(JOINT_SHORT):
            val = self.ros.joint_positions_by_name.get(UR3E_JOINT_NAMES[i], 0.0)
            marker = ">" if i == self.selected else " "
            parts.append(f"{marker}{short}:{val:+.3f}")
        line = "  ".join(parts)
        print(f"\r  {line}  step={self.step:.3f}    ", end="", flush=True)


def main() -> None:
    rclpy.init()
    teleop = KeyboardTeleop()
    try:
        teleop.run()
    finally:
        teleop.ros.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
