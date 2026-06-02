# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.cyborg import CYBORG_BIPED_ACTION_SCALE, CYBORG_BIPED_CFG
from robot_lab.tasks.manager_based.beyondamp.tracking_env_cfg import CyborgEnvCfg
import robot_lab.tasks.manager_based.beyondamp.obs_groups as amp_groups
import robot_lab.tasks.manager_based.beyondamp.config.cyborg.cyborg_params as params


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
        self.commands.motion.motion_file = params.CYBORG_MOTION_FILE
        self.commands.motion.anchor_body_name = params.CYBORG_ANCHOR_NAME
        self.commands.motion.body_names = params.CYBORG_KEY_BODY_NAMES
        self.commands.motion.amp_obs_terms = amp_groups.AMPObsHardTrackTerms
        self.episode_length_s = 30.0
