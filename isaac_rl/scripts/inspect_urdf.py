"""Print a structural summary of the prepared URDF.

Useful both as a sanity check before Isaac Sim import and as a reference when
writing the Isaac Lab `ArticulationCfg` (joint names, limits, link names).

Run:
    python isaac_rl/scripts/inspect_urdf.py
    python isaac_rl/scripts/inspect_urdf.py path/to/other.urdf   # override
"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ISAAC_RL = Path(__file__).resolve().parents[1]
DEFAULT_URDF = ISAAC_RL / "assets" / "urdf" / "ur_onrobot_prepared.urdf"


def main(urdf_path: Path) -> int:
    if not urdf_path.is_file():
        print(f"ERROR: URDF not found: {urdf_path}", file=sys.stderr)
        return 1

    root = ET.parse(urdf_path).getroot()
    robot = root.attrib.get("name", "?")
    print(f"=== Robot: {robot} ({urdf_path.name}) ===\n")

    # Links
    links = root.findall("link")
    print(f"=== Links ({len(links)}) ===")
    print(f"{'name':<28} {'mass(kg)':>10}  {'visual':>6} {'colli':>5} notes")
    missing_inertia = []
    zero_mass = []
    for link in links:
        name = link.attrib["name"]
        inertial = link.find("inertial")
        if inertial is None:
            mass_str = "(none)"
            missing_inertia.append(name)
        else:
            m_el = inertial.find("mass")
            if m_el is None:
                mass_str = "(no mass tag)"
            else:
                mass = float(m_el.attrib.get("value", "0"))
                mass_str = f"{mass:.4f}"
                if mass == 0:
                    zero_mass.append(name)
        visual_n = len(link.findall("visual"))
        coll_n = len(link.findall("collision"))
        notes = []
        if any(k in name for k in ("tcp", "tip", "frame", "mock")):
            notes.append("frame-only")
        notes_str = ",".join(notes)
        print(f"{name:<28} {mass_str:>10}  {visual_n:>6} {coll_n:>5} {notes_str}")

    print()
    # Joints
    joints = root.findall("joint")
    print(f"=== Joints ({len(joints)}) ===")
    type_counts: dict[str, int] = {}
    print(f"{'name':<32} {'type':<10} {'parent':<22} {'child':<22} limits/axis")
    for j in joints:
        name = j.attrib["name"]
        jtype = j.attrib["type"]
        type_counts[jtype] = type_counts.get(jtype, 0) + 1
        parent_el = j.find("parent")
        child_el = j.find("child")
        parent = parent_el.attrib["link"] if parent_el is not None else "?"
        child = child_el.attrib["link"] if child_el is not None else "?"
        extra_parts = []
        if jtype in ("revolute", "prismatic", "continuous"):
            axis = j.find("axis")
            ax = axis.attrib.get("xyz", "?") if axis is not None else "?"
            limit = j.find("limit")
            if limit is not None:
                lo = limit.attrib.get("lower", "?")
                hi = limit.attrib.get("upper", "?")
                eff = limit.attrib.get("effort", "?")
                vel = limit.attrib.get("velocity", "?")
                extra_parts.append(f"[{lo}, {hi}] axis={ax} eff={eff} vel={vel}")
            else:
                extra_parts.append(f"NO LIMIT axis={ax}")
        elif jtype == "fixed":
            extra_parts.append("(fixed)")
        mimic = j.find("mimic")
        if mimic is not None:
            extra_parts.append(f"MIMICS {mimic.attrib.get('joint', '?')}")
        print(f"{name:<32} {jtype:<10} {parent:<22} {child:<22} {' '.join(extra_parts)}")

    print()
    print("Joint type counts:", type_counts)
    mimics = [j for j in joints if j.find("mimic") is not None]
    if mimics:
        print(f"\nMimic joints ({len(mimics)}):")
        for m in mimics:
            mi = m.find("mimic")
            print(
                f"  {m.attrib['name']:<28} mimics {mi.attrib.get('joint'):<24}"
                f" multiplier={mi.attrib.get('multiplier', 1)}"
                f" offset={mi.attrib.get('offset', 0)}"
            )

    print()
    print("=== Sanity flags ===")
    print(f"Links missing <inertial>: {len(missing_inertia)} {missing_inertia or ''}")
    print(f"Links with zero mass:     {len(zero_mass)} {zero_mass or ''}")
    ros2c = root.findall("ros2_control")
    gazebo = root.findall("gazebo")
    print(f"<ros2_control> blocks (Isaac ignores): {len(ros2c)}")
    print(f"<gazebo>        blocks (Isaac ignores): {len(gazebo)}")

    return 0


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_URDF
    sys.exit(main(target))
