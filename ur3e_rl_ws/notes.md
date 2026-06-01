Terminal 1 — Launch the sim
￼
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select ur3e_rl_env ur3e_gazebo_sim --symlink-install
source install/setup.bash
ros2 launch ur3e_gazebo_sim ur3e_pick_place_world.launch.py
Wait until Gazebo is fully up and the arm is visible.

Terminal 2 — RViz (optional but recommended)
￼
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
rviz2 -d scripts/training_monitor.rviz
In RViz, add a Marker display subscribed to /ik_target_marker — yellow sphere = approach target, red = cube centre.

Terminal 3 — The IK visualiser
￼
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
# 6 preset positions covering the workspace:
python3 scripts/visualize_ik.py

# OR specific cube:
python3 scripts/visualize_ik.py --cube 0.0 -0.3 1.11

# OR N random positions:
python3 scripts/visualize_ik.py --random 8




new 

# Terminal 1 — rebuild (only kinematics.py changed, symlink-install is fast)
cd ~/git/HoloAssist-AI/ur3e_rl_ws
source /opt/ros/humble/setup.bash && source install/setup.bash
colcon build --packages-select ur3e_rl_env --symlink-install

# Terminal 3 — re-run the visualizer (sim must still be running)
source /opt/ros/humble/setup.bash && source install/setup.bash
python3 scripts/visualize_ik.py

