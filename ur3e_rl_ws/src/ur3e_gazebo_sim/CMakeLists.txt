cmake_minimum_required(VERSION 3.8)
project(ur3e_gazebo_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds models urdf meshes
  DESTINATION share/${PROJECT_NAME}
)

install(
  FILES README.md
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
