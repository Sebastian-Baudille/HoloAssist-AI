from __future__ import annotations

import time

import numpy as np
import rclpy

from ur3e_rl_env.ros_interface import RosInterfaceNode, UR3E_JOINT_NAMES


def _spin_for(node: RosInterfaceNode, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while rclpy.ok() and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = RosInterfaceNode("ur3e_smoke_test_joint_command")
    try:
        if not node.wait_for_joint_state(timeout_sec=10.0):
            raise RuntimeError("No complete UR3e joint state received.")

        before_state = node.get_state()
        if before_state is None:
            current_joints = np.array(
                [node.joint_positions_by_name[name] for name in UR3E_JOINT_NAMES],
                dtype=np.float32,
            )
        else:
            current_joints = np.asarray(before_state["joint_positions"], dtype=np.float32)

        target_joints = current_joints.copy()
        target_joints[0] += 0.1

        print("Before joint positions:")
        print(dict(zip(UR3E_JOINT_NAMES, current_joints.tolist())))

        node.send_joint_target(target_joints, duration_sec=1.0)
        _spin_for(node, 1.5)

        after_joints = np.array(
            [
                node.joint_positions_by_name.get(name, float("nan"))
                for name in UR3E_JOINT_NAMES
            ],
            dtype=np.float32,
        )
        print("After joint positions:")
        print(dict(zip(UR3E_JOINT_NAMES, after_joints.tolist())))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

