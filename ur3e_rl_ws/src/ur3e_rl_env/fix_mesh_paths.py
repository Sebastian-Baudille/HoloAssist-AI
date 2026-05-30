#!/usr/bin/env python3
"""Fix mesh paths in URDF to absolute paths that MuJoCo can read."""
import re

REPO = "/home/john/git/HoloAssist-AI"
ASSETS = f"{REPO}/ur3e_rl_ws/src/ur3e_rl_env/assets/mujoco"
URDF_PATH = f"{ASSETS}/ur3e_rg2.urdf"

with open(URDF_PATH) as f:
    content = f.read()

# xacro already expands package:// to file:// absolute URIs.
# MuJoCo needs plain absolute paths, not file:// URIs.
content = re.sub(r'filename="file://', 'filename="', content)

# Also catch any remaining package:// (shouldn't exist but handle anyway)
content = re.sub(
    r'package://ur3e_gazebo_sim/meshes/',
    f'{ASSETS}/meshes/',
    content
)

remaining = re.findall(r'package://[^"]+', content)
if remaining:
    print(f"WARNING: Unresolved package:// paths: {remaining[:5]}")

remaining_file = re.findall(r'filename="file://', content)
if remaining_file:
    print(f"WARNING: Still has file:// paths: {len(remaining_file)}")

with open(URDF_PATH, 'w') as f:
    f.write(content)

# Report what paths look like now
sample = re.findall(r'filename="[^"]*"', content)[:6]
print("Sample paths after fix:")
for s in sample:
    print(f"  {s}")
print(f"\nTotal filename references: {len(re.findall(r'filename=', content))}")
