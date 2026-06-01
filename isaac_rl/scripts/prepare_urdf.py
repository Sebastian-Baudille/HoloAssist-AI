"""Prepare a flattened URDF for Isaac Sim 5.1 URDF Importer.

Input:  isaac_rl/assets/urdf/ur_onrobot.urdf            (raw xacro flatten)
Output: isaac_rl/assets/urdf/ur_onrobot_prepared.urdf   (Isaac Sim-ready)

Two transforms applied:

1. **Rewrite mesh URIs.** The flattened URDF references meshes via ROS package
   URIs (`package://ur_description/meshes/...`). Isaac Sim's URDF importer can't
   resolve these. We rewrite to paths relative to the URDF file, pointing at
   the mesh tree we mirrored into isaac_rl/assets/meshes/.

2. **Strip <mimic> tags.** The RG2 gripper xacro uses a two-level mimic chain
   (`finger_width - finger_joint - 5 others`). Isaac Sim 5.1's URDF importer
   has known issues with multi-level mimics: it creates the joints as mimic
   typed but loses finite limits during conversion, then PhysX refuses to drive
   them ("needs a finite limit set to be used by the mimic joint feature"). The
   workaround is to remove `<mimic>` tags entirely — joints become plain
   revolute, we drive them independently in Python, replicating the gripper's
   parallel-jaw kinematics via a unified open/close helper.

Run from any cwd:
    python isaac_rl/scripts/prepare_urdf.py
"""

import re
import sys
from pathlib import Path

ISAAC_RL = Path(__file__).resolve().parents[1]
ASSETS = ISAAC_RL / "assets"
INPUT = ASSETS / "urdf" / "ur_onrobot.urdf"
OUTPUT = ASSETS / "urdf" / "ur_onrobot_prepared.urdf"
MESHES = ASSETS / "meshes"

REWRITES = {
    "package://ur_description/meshes/": "../meshes/ur_description/",
    "package://onrobot_description/meshes/": "../meshes/onrobot_description/",
}

# Match <mimic ... /> (self-closing) or <mimic ...></mimic>, possibly multiline.
MIMIC_PATTERN = re.compile(r"\s*<mimic\b[^/>]*?(?:/>|></mimic>)", re.DOTALL)


def main() -> int:
    if not INPUT.is_file():
        print(f"ERROR: input URDF not found: {INPUT}", file=sys.stderr)
        print("Run xacro flatten first (Phase 3 Step A).", file=sys.stderr)
        return 1

    text = INPUT.read_text()

    # --- Transform 1: rewrite package:// URIs -------------------------------
    counts = {old: text.count(old) for old in REWRITES}
    for old, new in REWRITES.items():
        text = text.replace(old, new)

    remaining = sorted(set(re.findall(r"package://[^\"']+", text)))
    if remaining:
        print(f"WARNING: {len(remaining)} unrewritten package:// URIs remain:", file=sys.stderr)
        for u in remaining:
            print(f"  {u}", file=sys.stderr)
        print("Add a new entry to REWRITES if these are expected.", file=sys.stderr)

    # --- Transform 2: strip <mimic> tags ------------------------------------
    mimic_matches = MIMIC_PATTERN.findall(text)
    text = MIMIC_PATTERN.sub("", text)

    OUTPUT.write_text(text)
    print(f"Wrote {OUTPUT.relative_to(ISAAC_RL.parent)}")
    for old, n in counts.items():
        print(f"  {n:>3} URI replacements: {old!r} -> {REWRITES[old]!r}")
    print(f"  {len(mimic_matches):>3} <mimic> tags removed")

    # --- Verify every relative mesh path resolves on disk -------------------
    rel_uris = sorted(set(re.findall(r'filename="(\.\./meshes/[^"]+)"', text)))
    missing = []
    for rel in rel_uris:
        full = (OUTPUT.parent / rel).resolve()
        if not full.is_file():
            missing.append((rel, full))
    print(f"  {len(rel_uris)} unique relative mesh paths checked")
    if missing:
        print(f"ERROR: {len(missing)} mesh files missing:", file=sys.stderr)
        for rel, full in missing:
            print(f"  {rel}  ->  {full}", file=sys.stderr)
        return 2

    print("OK — all meshes resolve, mimics stripped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
