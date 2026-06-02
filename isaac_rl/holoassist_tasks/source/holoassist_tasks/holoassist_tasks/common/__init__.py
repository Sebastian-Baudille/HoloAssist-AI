# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Robot-agnostic utilities shared across HoloAssist tasks.

Currently exposed:
  - kinematics : FK + scipy IK reference solver (numpy + scipy, no ROS,
    no Isaac dependencies). Adapted from
    `ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/kinematics.py` with
    ROBOT_BASE_Z changed from 1.10 (ROS Gazebo) to 1.0 (Isaac).

Anything genuinely task-specific (rewards, observation layouts, action
strategies) stays in the task subpackage. Only utilities that would be
duplicated across reach / grasp / transport / pick_place belong here.
"""
