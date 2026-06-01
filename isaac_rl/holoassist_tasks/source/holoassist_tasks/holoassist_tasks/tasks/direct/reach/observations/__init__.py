# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Observation strategies for the reach task.

Each module in this package defines a single function `build(env) -> dict`
that returns the policy observation tensor. The env class imports one
strategy per training run (aliased as `obs_strategy`). To A/B test a new
observation, add a new module and swap the import in `reach_env.py`.
"""
