# RL Training Progress

## MuJoCo Backend (2026-05-30)

- [x] xacro expanded to URDF (base_z=1.10, base_yaw=π)
- [x] Mesh paths fixed, URDF loads in MuJoCo
- [x] URDF converted to MJCF via MjSpec (fusestatic=False → gripper_tcp body preserved)
- [x] scene.xml built and steps without error (DoF=34, 26 bodies)
- [x] ur3e_mujoco_env.py implemented
- [x] Weight transfer test passed (all 5 assertions)
- [x] Sanity training run passed (1531 steps/s on 4 envs)
- [ ] Full training run started (500k steps)
- [ ] ep_rew_mean at 100k steps: ___
- [ ] ep_rew_mean at 500k steps: ___
- [ ] Compared to Gazebo PPO baseline: ___

### Start full run

```bash
cd /home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env/ur3e_rl_env
PYTHONPATH=/home/john/git/HoloAssist-AI/ur3e_rl_ws/src/ur3e_rl_env \
python3 train_ppo_mujoco.py --timesteps 500000 --envs 16
```

### Monitor

```bash
tensorboard --logdir /home/john/git/HoloAssist-AI/ur3e_rl_ws/mujoco_tb_logs
```

### Key files

| File | Purpose |
|---|---|
| `src/ur3e_rl_env/assets/mujoco/scene.xml` | MuJoCo scene (robot + table + cubes + bins) |
| `src/ur3e_rl_env/ur3e_rl_env/envs/ur3e_mujoco_env.py` | Gymnasium env |
| `src/ur3e_rl_env/ur3e_rl_env/train_ppo_mujoco.py` | PPO training script |
| `src/ur3e_rl_env/weight_transfer_test.py` | Compatibility verification |
| `rl_models/mujoco_checkpoints/` | Saved checkpoints |
| `rl_models/mujoco_best/best_model.zip` | Best eval checkpoint |
| `mujoco_tb_logs/` | TensorBoard logs |

### Notes

- Robot base at z=1.10 (matches Gazebo launch, NOT xacro default 0.82)
- RG2 gripper all-fixed joints → virtual _gripper_closed bool, no physics actuation
- Obs normalization: exact copy of build_observation() in ros_interface.py
- Cube spawn: X(-0.20,0.20), Y(-0.45,-0.10), Z=1.11 — matches Gazebo env
- PHYSICS_STEPS_PER_RL_STEP=50 (0.1s/step) vs Gazebo 0.2-0.3s; shorter = faster training
- If SubprocVecEnv fork hangs: change start_method="fork" to "spawn" in train_ppo_mujoco.py
