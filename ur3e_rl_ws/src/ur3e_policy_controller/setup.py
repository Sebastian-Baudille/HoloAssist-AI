from setuptools import find_packages, setup

package_name = "ur3e_policy_controller"

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
    description="Runtime controller node for trained UR3e RL policies.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "rl_policy_node = ur3e_policy_controller.rl_policy_node:main",
        ],
    },
)

