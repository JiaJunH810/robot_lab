# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.cyborg import CYBORG_BIPED_ACTION_SCALE, CYBORG_BIPED_CFG
from robot_lab.tasks.manager_based.beyondamp.tracking_env_cfg import CyborgEnvCfg
import robot_lab.tasks.manager_based.beyondamp.obs_groups as amp_groups


@configclass
class CyborgBeyondAMPFlatEnvCfg(CyborgEnvCfg):
    """Cyborg beyondAMP 环境配置 —— 带 AMP 判别器观测组。

    amp 观测组已通过 tracking_env_cfg.py 的 ObservationsCfg 继承，
    body 数据过滤由 MotionCommand.body_indexes 自动完成。
    """

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CYBORG_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CYBORG_BIPED_ACTION_SCALE
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/"
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "hip_l_roll_link", "knee_l_pitch_link", "ankle_l_roll_link",
            "hip_r_roll_link", "knee_r_pitch_link", "ankle_r_roll_link",
            "waist_yaw_link",
            "arm_l_02_link", "arm_l_04_link", "arm_l_07_link",
            "arm_r_02_link", "arm_r_04_link", "arm_r_07_link",
        ]
        self.commands.motion.amp_obs_terms = amp_groups.AMPObsHardTrackTerms
        self.episode_length_s = 30.0
