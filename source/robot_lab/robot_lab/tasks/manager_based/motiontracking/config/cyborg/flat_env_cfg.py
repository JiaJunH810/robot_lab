# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.cyborg import CYBORG_BIPED_ACTION_SCALE, CYBORG_BIPED_CFG
from robot_lab.tasks.manager_based.motiontracking.tracking_env_cfg import CyborgEnvCfg


@configclass
class CyborgBeyondMimicFlatEnvCfg(CyborgEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = CYBORG_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = CYBORG_BIPED_ACTION_SCALE
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/B1_-_stand_to_walk_stageii.npz"
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "hip_l_roll_link",
            "knee_l_pitch_link",
            "ankle_l_roll_link",
            "hip_r_roll_link",
            "knee_r_pitch_link",
            "ankle_r_roll_link",
            "waist_yaw_link",
            "arm_l_02_link",
            "arm_l_04_link",
            "arm_l_07_link",
            "arm_r_02_link",
            "arm_r_04_link",
            "arm_r_07_link",
        ]

        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None

        self.episode_length_s = 30.0
