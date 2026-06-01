# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward strategies for the reach task.

Each module in this package defines a single function `compute(env) -> Tensor`
that returns the per-env scalar reward. The env class imports one strategy
per training run (aliased as `reward_strategy`). To A/B test a new reward
shape, add a new module and swap the import in `reach_env.py`.
"""
