from glob import glob
import os

from setuptools import find_packages, setup

package_name = "cube_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="john",
    maintainer_email="176361326+John-A-Chen@users.noreply.github.com",
    description="Cube detection and tracking from RealSense point clouds for HoloAssist-AI.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "perception_node = cube_perception.perception_node:main",
            "perception_benchmark = cube_perception.benchmark:main",
        ],
    },
)
