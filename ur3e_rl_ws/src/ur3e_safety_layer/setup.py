from setuptools import find_packages, setup

package_name = "ur3e_safety_layer"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Ollie",
    maintainer_email="ollie@example.com",
    description="Safety layer for UR3e reinforcement learning.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "safety_node = ur3e_safety_layer.safety_node:main",
            "moveit_collision_checker = ur3e_safety_layer.moveit_collision_checker:main",
        ],
    },
)

