from setuptools import find_packages, setup

package_name = "ur3e_rl_env"

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
    description="Gymnasium PPO environment for UR3e Gazebo reinforcement learning.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "train_ppo = ur3e_rl_env.train_ppo:main",
            "train_ppo_parallel = ur3e_rl_env.train_ppo_parallel:main",
            "evaluate_policy = ur3e_rl_env.evaluate_policy:main",
            "smoke_test_joint_command = ur3e_rl_env.smoke_test_joint_command:main",
            "record_demo = ur3e_rl_env.record_demo:main",
            "pretrain_from_demos = ur3e_rl_env.pretrain_from_demos:main",
            "keyboard_teleop = ur3e_rl_env.keyboard_teleop:main",
            "train_ppo_mujoco = ur3e_rl_env.train_ppo_mujoco:main",
        ],
    },
)
