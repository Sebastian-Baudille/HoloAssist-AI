cmake_minimum_required(VERSION 3.8)
project(holoassist_sim)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch worlds config
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/render_world.py
    scripts/save_pointcloud.py
    scripts/scene_controller.py
    scripts/dataset_capture.py
    scripts/randomize_scene.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
