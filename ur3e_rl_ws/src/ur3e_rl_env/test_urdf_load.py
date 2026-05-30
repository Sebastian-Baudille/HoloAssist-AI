#!/usr/bin/env python3
import mujoco

ASSETS = "/home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco"
urdf_path = f"{ASSETS}/ur3e_rg2.urdf"

model = mujoco.MjModel.from_xml_path(urdf_path)
print(f"✓ URDF loaded: {model.nq} DoF, {model.nbody} bodies, {model.ngeom} geoms")
print(f"\nJoints ({model.njnt}):")
for i in range(model.njnt):
    j = model.joint(i)
    print(f"  [{i}] {j.name!r:45} type={j.type}")
print(f"\nBodies ({model.nbody}):")
for i in range(model.nbody):
    b = model.body(i)
    print(f"  [{i}] {b.name!r}")
print(f"\nSites ({model.nsite}):")
for i in range(model.nsite):
    print(f"  [{i}] {model.site(i).name!r}")
