cmake_minimum_required(VERSION 3.8)
project(ur3e_gazebo_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds models urdf meshes config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/gazebo_pose_bridge.py
    scripts/send_home_trajectory.py
    scripts/setup_controllers.py
  DESTINATION lib/${PROJECT_NAME}
)

install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
