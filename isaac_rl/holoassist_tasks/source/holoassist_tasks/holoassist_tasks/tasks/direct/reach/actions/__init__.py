# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Action processing strategies for the reach task.

Each module in this package defines two functions:
    process(env, action) -> None    # called once per env step (in _pre_physics_step)
    apply(env) -> None              # called once per physics step (in _apply_action)

The env class imports one strategy per training run (aliased as `action_strategy`).
To A/B test a new action space (e.g., absolute joint targets, operational-space
control, end-effector velocity), add a new module and swap the import in
`reach_env.py`.
"""
