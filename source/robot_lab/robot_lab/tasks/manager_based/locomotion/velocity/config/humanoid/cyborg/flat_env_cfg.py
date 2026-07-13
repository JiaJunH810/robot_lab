# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import CyborgHPRoughEnvCfg


@configclass
class CyborgHPFlatEnvCfg(CyborgHPRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        # no command curriculum
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # Rewards
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.02
        self.rewards.joint_acc_l2.weight = 0
        self.rewards.joint_torques_l2.weight = -2.0e-6
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*"]
        self.rewards.feet_air_time.weight = 10.0
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_height.weight = -1.5
        # joint_pos_penalty inherits -1.0 from rough_env_cfg
        self.rewards.feet_slide.weight = -0.1

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CyborgHPFlatEnvCfg":
            self.disable_zero_weight_rewards()
