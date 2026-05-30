#!/usr/bin/env python3
"""Convert URDF to MuJoCo MJCF using MjSpec with fusestatic=False."""
import mujoco

ASSETS = "/home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco"
urdf_path = f"{ASSETS}/ur3e_rg2.urdf"
mjcf_path = f"{ASSETS}/robot.xml"

spec = mujoco.MjSpec.from_file(urdf_path)
spec.compiler.fusestatic = False

xml_str = spec.to_xml()
with open(mjcf_path, 'w') as f:
    f.write(xml_str)

# Verify round-trip
model = mujoco.MjModel.from_xml_path(mjcf_path)
print(f"✓ Converted to MJCF: {model.nq} DoF, {model.nbody} bodies")
print(f"  Written to: {mjcf_path}")

print(f"\nBodies ({model.nbody}):")
for i in range(model.nbody):
    print(f"  {model.body(i).name!r}")

print(f"\nJoints ({model.njnt}):")
for i in range(model.njnt):
    j = model.joint(i)
    print(f"  {j.name!r:45} type={j.type}")
