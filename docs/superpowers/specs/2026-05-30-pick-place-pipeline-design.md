# Pick-and-Place Pipeline Design

**Date:** 2026-05-30
**Goal:** Train two small PPO models (reach + transport) plus scripted grasp/release, deployable to MuJoCo sim and the real UR3e via ROS.

---

## Overview

Replace the failing single-model approach with a 3-stage pipeline:

```
REACH (Model 1) → GRASP (script) → TRANSPORT (Model 2) → RELEASE (script)
```

A coordinator state machine drives the pipeline. The same coordinator code runs in MuJoCo and on the real robot — only the backend (physics engine vs ROS topics) differs.

---

## Stage 1 — Reach (Model 1)

**Task:** Move TCP to within 5 cm of the nearest cube. Gripper always points down.

**Obs (6D, float32, normalised -1 to 1):**
| idx | value |
|-----|-------|
| 0-2 | EE position (x, y, z) normalised to workspace bounds |
| 3-5 | Target cube position (x, y, z) normalised to workspace bounds |

**Action (3D, float32, -1 to 1):** Cartesian delta (dx, dy, dz) in world frame. Scale: ±2 cm per step.

**IK layer:** At each step, convert (dx, dy, dz) to joint targets via Jacobian damped-least-squares with an orientation penalty that keeps the wrist pointing down. Applied before sending to MuJoCo actuators.

**Reward:** `-dist(EE, cube)` per step. Pure dense. No grasping term.

**Episode done:** dist(EE, cube) < 5 cm (success) or 200 steps (timeout).

**Expected training:** < 100k steps.

---

## Stage 2 — Grasp (scripted)

**Trigger:** dist(EE, cube) < 5 cm (sim) / 8 cm (real robot, vision noise margin).

**In MuJoCo:** Activate MuJoCo equality weld constraint between `gripper_tcp` body and the nearest cube body. Cube becomes rigidly attached.

**On real robot:** Call `ros_interface.close_gripper()`, wait 0.5 s. Cube is considered held.

No model. No training.

---

## Stage 3 — Transport (Model 2)

**Task:** Move arm (with cube attached) to within 8 cm of bin_0, then release.

**Obs (7D, float32, normalised -1 to 1):**
| idx | value |
|-----|-------|
| 0-2 | EE position (x, y, z) normalised |
| 3-5 | Cube position (x, y, z) normalised (follows arm in sim due to weld) |
| 6   | dist(cube, bin) normalised |

**Action (3D):** Same Cartesian delta as Model 1.

**Reward:** `-dist(cube, bin)` per step. Pure dense.

**Episode done:** dist(cube, bin) < 8 cm (success) or 200 steps (timeout).

**Expected training:** < 100k steps.

---

## Stage 4 — Release (scripted)

**In MuJoCo:** Remove weld constraint. Cube drops under gravity.
**On real robot:** Call `ros_interface.open_gripper()`.

---

## Coordinator

A ~60-line state machine with states: `REACH → GRASP → TRANSPORT → RELEASE → DONE`.

One `PickPlaceCoordinator` class with two concrete subclasses:
- `MuJoCoCoordinator` — loads both models, runs physics, manages weld constraint
- `RosCoordinator` — subscribes to ROS topics, calls ROS gripper commands (future)

The coordinator is what gets tested end-to-end in the MuJoCo viewer.

---

## New Files

```
ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env/
  envs/
    reach_env.py          # Stage 1 Gymnasium env
    transport_env.py      # Stage 3 Gymnasium env
  ik.py                   # Jacobian IK: (dx,dy,dz) → joint targets, orientation locked
  coordinator.py          # State machine + MuJoCoCoordinator
  train_reach.py          # Train Model 1
  train_transport.py      # Train Model 2
watch_pipeline.py         # Watch full pipeline in viewer
```

---

## Immediate Milestone

Train `reach_env.py` and verify the arm visibly moves toward cubes in `watch_pipeline.py`. This is "C" from the user's request — watch it work in the viewer before grasping is added.

---

## Constraints

- EE orientation locked to pointing down throughout all stages
- Cube spawn: X(-0.20, 0.20), Y(-0.45, -0.10), Z=1.11 (matches existing env)
- Workspace bounds: same as `constants.py`
- Action scale: ±2 cm per step (tunable; start conservative for real robot safety)
- No changes to `reward.py`, `constants.py`, or existing envs
