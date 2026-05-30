#!/usr/bin/env python3
import mujoco

ASSETS = "/home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco"
scene_path = f"{ASSETS}/scene.xml"

model = mujoco.MjModel.from_xml_path(scene_path)
data  = mujoco.MjData(model)

for _ in range(100):
    mujoco.mj_step(model, data)

print(f"✓ Scene stepped 100 times. DoF={model.nq}, bodies={model.nbody}")

print(f"\nActuators ({model.nu}):")
for i in range(model.nu):
    print(f"  [{i}] {model.actuator(i).name!r}")

print(f"\nSites ({model.nsite}):")
for i in range(model.nsite):
    print(f"  [{i}] {model.site(i).name!r}  pos={data.site_xpos[i]}")

print(f"\nBodies with 'cube' in name:")
for i in range(model.nbody):
    name = model.body(i).name
    if 'cube' in name:
        print(f"  [{i}] {name!r}  pos={data.xpos[i]}")

print(f"\nBodies with 'tcp' or 'gripper' in name:")
for i in range(model.nbody):
    name = model.body(i).name
    if 'tcp' in name.lower() or 'gripper' in name.lower():
        print(f"  [{i}] {name!r}  pos={data.xpos[i]}")
