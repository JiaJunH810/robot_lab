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


        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CyborgHPFlatEnvCfg":
            self.disable_zero_weight_rewards()
