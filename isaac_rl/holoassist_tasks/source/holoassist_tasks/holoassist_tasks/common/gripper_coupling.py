# Copyright (c) 2026, HoloAssist-AI contributors.
# SPDX-License-Identifier: BSD-3-Clause

"""Couple RG2 linkage joints via PhysX (mimic API preferred, fixed tendon fallback).

Replicates the URDF `<mimic>` chain that was stripped during Isaac's URDF
import (the importer drops the joint limits when mimic is present — see
scripts/prepare_urdf.py). Without this coupling, the 6 RG2 finger linkage
joints are driven independently. The grasp test shows the failure: under
contact (cube between fingers), the four-bar mechanism deforms because each
joint settles independently — visible geometry artifacts.

Two strategies, tried in order:

1. **PhysxMimicJointAPI (preferred)** — applied per follower joint, with
   a `referenceJoint` relationship pointing at the master joint, and a
   `gearing` scalar. Works for joints in PARALLEL branches (our RG2 case:
   6 joints all hanging off the gripper base). No topology constraint.

2. **PhysX articulation fixed tendon (fallback)** — applies a tendon root
   API to the master joint and tendon axis API to followers. Requires
   joints to lie along a SERIAL chain from articulation root. Our RG2
   doesn't satisfy this — PhysX returns "topology issue" warnings and
   rejects the constraint. Tried as fallback only.

Mechanism layout:
    master            : finger_joint               (the joint we command)
    followers (gear)  : left_inner_knuckle_joint   -1
                       left_inner_finger_joint    +1
                       right_outer_knuckle_joint  -1
                       right_inner_knuckle_joint  -1
                       right_inner_finger_joint   +1

Gear sign: +1 means follower moves SAME direction as master, -1 means
OPPOSITE. Matches the closed-pose values in UR_ONROBOT_CFG.init_state.

Usage (call BEFORE sim.reset() so PhysX initialises with the constraint):

    import omni.usd
    from holoassist_tasks.common.gripper_coupling import apply_rg2_mimic
    stage = omni.usd.get_context().get_stage()
    apply_rg2_mimic(stage, "/World/envs/env_0/Robot")
    sim.reset()
"""

from __future__ import annotations

from pxr import Sdf, Usd, UsdPhysics

try:
    from pxr import PhysxSchema
    _HAS_PHYSX_SCHEMA = True
except ImportError:
    _HAS_PHYSX_SCHEMA = False


# ------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------

MASTER_JOINT_NAME = "finger_joint"

# Gear ratio per follower joint. +1 = same direction as master, -1 = opposite.
# Signs match the closed-pose values in UR_ONROBOT_CFG.init_state.
FOLLOWER_GEARS = {
    "left_inner_knuckle_joint":  -1.0,
    "left_inner_finger_joint":   +1.0,
    "right_outer_knuckle_joint": -1.0,
    "right_inner_knuckle_joint": -1.0,
    "right_inner_finger_joint":  +1.0,
}

# Candidate schema class names — version-dependent. First match is used.
_MIMIC_API_CANDIDATES = (
    "PhysxMimicJointAPI",      # Isaac Sim 5.x naming
    "PhysxJointMimicAPI",      # alternative naming
)
_TENDON_ROOT_API_CANDIDATES = (
    "PhysxArticulationFixedTendonAPI",
    "PhysxFixedTendonAPI",
    "PhysxArticulationFixedTendonAxisAPI",
    "PhysxTendonAxisRootAPI",   # what was found in 5.1
)
_TENDON_AXIS_API_CANDIDATES = (
    "PhysxArticulationTendonAxisAPI",
    "PhysxTendonAxisAPI",
    "PhysxArticulationFixedTendonAxisAPI",
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _find_joint_prim(stage, root_path: str, joint_name: str):
    """Recursively find a joint prim by name under an articulation root."""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdPhysics.Joint) and prim.GetName() == joint_name:
            return prim
    return None


def _enumerate_coupling_schemas() -> None:
    """Print all joint-coupling-related schemas in pxr.PhysxSchema and pxr.UsdPhysics.

    Used at startup so we can see what's actually available in this build
    and update the candidate lists if needed.
    """
    print("[gripper_coupling] Schema discovery — joint coupling APIs available:", flush=True)
    for module_name, mod in (("pxr.PhysxSchema", PhysxSchema), ("pxr.UsdPhysics", UsdPhysics)):
        if mod is None:
            print(f"  {module_name}: <not imported>", flush=True)
            continue
        matches = sorted(
            name for name in dir(mod)
            if any(kw in name for kw in ("Tendon", "Mimic", "Coupl", "Constrain", "FixedJoint"))
            and not name.startswith("_")
        )
        if matches:
            print(f"  Found in {module_name}:", flush=True)
            for name in matches:
                print(f"    - {name}", flush=True)
        else:
            print(f"  None in {module_name}", flush=True)


def _resolve_first_available(module, candidates: tuple[str, ...]):
    """Return (name, cls) of the first candidate that exists on module, or (None, None)."""
    for name in candidates:
        if hasattr(module, name):
            return name, getattr(module, name)
    return None, None


# ------------------------------------------------------------------
# Strategy 1: per-joint mimic API (preferred — no topology constraints)
# ------------------------------------------------------------------

def _apply_mimic_api(stage, root_path: str, mimic_cls,
                     instance_name: str = "rg2") -> int:
    """Apply PhysxMimicJointAPI (multi-apply schema) to each follower joint.

    PhysxMimicJointAPI is multi-apply — Apply() requires (prim, instance_name)
    where instance_name is a TfToken that namespaces the schema's attributes.
    Using the same instance name across all followers keeps the USD layer
    consistent.

    Returns the number of followers successfully constrained.

    This API is per-joint and doesn't require a serial kinematic chain —
    perfect for our RG2's six parallel joints branching off the gripper base.
    """
    from pxr import Sdf  # for path types if needed

    master_prim = _find_joint_prim(stage, root_path, MASTER_JOINT_NAME)
    if master_prim is None:
        raise RuntimeError(f"Master joint '{MASTER_JOINT_NAME}' not found under {root_path}")
    master_path = master_prim.GetPath()

    n_applied = 0
    n_failed = 0
    for joint_name, gear in FOLLOWER_GEARS.items():
        follower_prim = _find_joint_prim(stage, root_path, joint_name)
        if follower_prim is None:
            print(f"  WARN: follower joint not found: {joint_name}", flush=True)
            continue

        try:
            # Multi-apply schema: Apply(prim, instance_name) per the C++ signature
            # reported by Boost.Python in the v0.1 run.
            mimic_api = mimic_cls.Apply(follower_prim, instance_name)

            # Reference joint relationship — points at the master joint.
            ref_rel = mimic_api.CreateReferenceJointRel()
            ref_rel.SetTargets([master_path])

            # Gear ratio: follower_pos = gearing * master_pos + offset
            if hasattr(mimic_api, "CreateGearingAttr"):
                mimic_api.CreateGearingAttr(gear)

            # Optional attributes — set if the schema exposes them.
            if hasattr(mimic_api, "CreateOffsetAttr"):
                mimic_api.CreateOffsetAttr(0.0)
            # Drive parameters for the mimic constraint itself. Higher
            # naturalFrequency = stiffer coupling; damping=1 = critical (no overshoot).
            # Bump well above expected disturbances so contact forces from the
            # cube don't deflect followers out of the constrained relationship.
            if hasattr(mimic_api, "CreateNaturalFrequencyAttr"):
                mimic_api.CreateNaturalFrequencyAttr(200.0)
            if hasattr(mimic_api, "CreateDampingRatioAttr"):
                mimic_api.CreateDampingRatioAttr(1.0)

            # Optional: specify which axis of the reference joint to mimic.
            # For our revolute joints, the typical axis is "rotZ" but
            # PhysX usually figures it out from the joint type. Set explicitly
            # if the attribute exists so we don't rely on defaults.
            if hasattr(mimic_api, "CreateReferenceJointAxisAttr"):
                mimic_api.CreateReferenceJointAxisAttr("rotZ")

            print(f"  - Applied to {joint_name} (gearing={gear:+.1f}, ref={MASTER_JOINT_NAME})", flush=True)
            n_applied += 1
        except Exception as e:
            print(f"  - FAILED on {joint_name}: {type(e).__name__}: {e}", flush=True)
            n_failed += 1

    if n_failed > 0:
        raise RuntimeError(f"Mimic API failed on {n_failed}/{len(FOLLOWER_GEARS)} followers")
    return n_applied


# ------------------------------------------------------------------
# Strategy 2: fixed tendon (fallback — known broken for our topology, but try)
# ------------------------------------------------------------------

def _apply_fixed_tendon(stage, root_path: str, root_cls, axis_cls,
                        stiffness: float, damping: float,
                        instance_name: str = "rg2_mimic") -> int:
    """Apply a PhysX articulation fixed tendon. Known to fail with topology
    error on RG2 (parallel-branch joints), kept as fallback for diagnostics."""
    master_prim = _find_joint_prim(stage, root_path, MASTER_JOINT_NAME)
    if master_prim is None:
        raise RuntimeError(f"Master joint '{MASTER_JOINT_NAME}' not found")

    # Apply the root tendon API to the master joint
    tendon_root = root_cls.Apply(master_prim, instance_name)
    if hasattr(tendon_root, "CreateStiffnessAttr"):
        tendon_root.CreateStiffnessAttr(stiffness)
    if hasattr(tendon_root, "CreateDampingAttr"):
        tendon_root.CreateDampingAttr(damping)
    if hasattr(tendon_root, "CreateRestLengthAttr"):
        tendon_root.CreateRestLengthAttr(0.0)
    if hasattr(tendon_root, "CreateGearingAttr"):
        tendon_root.CreateGearingAttr([1.0])

    n_applied = 0
    for joint_name, our_sign in FOLLOWER_GEARS.items():
        follower_prim = _find_joint_prim(stage, root_path, joint_name)
        if follower_prim is None:
            continue
        axis = axis_cls.Apply(follower_prim, instance_name)
        # Tendon constraint: sum(gear_i * pos_i) = restLength. With master
        # gear +1.0 and follower gear -our_sign, follower = our_sign * master.
        if hasattr(axis, "CreateGearingAttr"):
            axis.CreateGearingAttr([-our_sign])
        if hasattr(axis, "CreateForceCoefficientAttr"):
            axis.CreateForceCoefficientAttr([1.0])
        n_applied += 1
    return n_applied


# ------------------------------------------------------------------
# Main entry point
# ------------------------------------------------------------------

def apply_rg2_mimic(
    stage,
    articulation_root_path: str,
    tendon_stiffness: float = 50000.0,
    tendon_damping: float = 500.0,
) -> str:
    """Couple the RG2 linkage joints. Returns the strategy used:
       'mimic_api', 'fixed_tendon', or 'none'.

    Call BEFORE sim.reset() so PhysX initialises with the constraints in place.
    """
    if not _HAS_PHYSX_SCHEMA:
        raise RuntimeError("PhysxSchema unavailable — cannot apply RG2 coupling.")

    art_prim = stage.GetPrimAtPath(articulation_root_path)
    if not art_prim.IsValid():
        raise RuntimeError(f"Articulation prim not found at {articulation_root_path}")

    _enumerate_coupling_schemas()

    # ---- Strategy 1: PhysxMimicJointAPI (preferred) ----
    mimic_name, mimic_cls = _resolve_first_available(PhysxSchema, _MIMIC_API_CANDIDATES)
    if mimic_cls is not None:
        print(f"[gripper_coupling] Strategy 1: PhysxSchema.{mimic_name}", flush=True)
        try:
            n = _apply_mimic_api(stage, articulation_root_path, mimic_cls)
            print(f"[gripper_coupling] SUCCESS — mimic API applied to {n}/{len(FOLLOWER_GEARS)} followers", flush=True)
            return "mimic_api"
        except Exception as e:
            import traceback
            print(f"[gripper_coupling] Mimic API failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    else:
        print("[gripper_coupling] No PhysxMimicJointAPI available — falling back to fixed tendon", flush=True)

    # ---- Strategy 2: Fixed tendon (fallback — known to fail topology on RG2) ----
    root_name, root_cls = _resolve_first_available(PhysxSchema, _TENDON_ROOT_API_CANDIDATES)
    axis_name, axis_cls = _resolve_first_available(PhysxSchema, _TENDON_AXIS_API_CANDIDATES)
    if root_cls is None or axis_cls is None:
        print("[gripper_coupling] No tendon APIs available either.", flush=True)
        return "none"

    print(f"[gripper_coupling] Strategy 2: PhysxSchema.{root_name} + PhysxSchema.{axis_name}", flush=True)
    print("[gripper_coupling] NOTE: this may fail with topology errors on RG2's parallel-branch joints.", flush=True)
    try:
        n = _apply_fixed_tendon(stage, articulation_root_path, root_cls, axis_cls,
                                 tendon_stiffness, tendon_damping)
        print(f"[gripper_coupling] Tendon schemas applied to master + {n}/{len(FOLLOWER_GEARS)} followers. "
              f"(PhysX may still reject at parse time — watch for topology warnings.)", flush=True)
        return "fixed_tendon"
    except Exception as e:
        import traceback
        print(f"[gripper_coupling] Fixed tendon also failed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return "none"


# Back-compat alias for any caller that imports the old name
def apply_rg2_mimic_tendon(stage, articulation_root_path: str, **kwargs) -> bool:
    """Deprecated name — kept for back-compat with grasp_test_v0.py. Use apply_rg2_mimic."""
    strategy = apply_rg2_mimic(stage, articulation_root_path, **kwargs)
    return strategy != "none"
