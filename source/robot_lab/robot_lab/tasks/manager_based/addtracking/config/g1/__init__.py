# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="RobotLab-Isaac-AddTracking-Flat-Unitree-G1-v0",
    # entry_point="isaaclab.envs:ManagerBasedRLEnv",
    entry_point="robot_lab.envs:AddG1Env",
    disable_env_checker=True,   # 关闭安检
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1BeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1AddTrackingFlatPPORunnerCfg",
    },
)
'''
entry_point: 选取哪种环境

环境配置 (物理世界、奖励、观测)：
    类名: UnitreeG1BeyondMimicFlatEnvCfg
    文件名: flat_env_cfg.py
    位置: 和你刚才打开的 __init__.py 在同一个文件夹下。

训练配置 (PPO算法、网络架构)：
    类名: UnitreeG1BeyondMimicFlatPPORunnerCfg
    文件名: rsl_rl_ppo_cfg.py
    位置: 在同一个文件夹下的 agents 子文件夹里。
'''