# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import os

from isaaclab.utils import configclass

from robot_lab.assets.unitree import UNITREE_G1_29DOF_ACTION_SCALE, UNITREE_G1_29DOF_CFG
from robot_lab.tasks.manager_based.beyondamp.tracking_env_cfg import CyborgEnvCfg
import robot_lab.tasks.manager_based.beyondamp.obs_groups as amp_groups


@configclass
class G1BeyondAMPFlatEnvCfg(CyborgEnvCfg):
    """Unitree G1 beyondAMP 环境配置。

    amp 观测组已通过 tracking_env_cfg.py 的 ObservationsCfg 继承，
    body 数据过滤由 MotionCommand.body_indexes 自动完成。
    """

    def __post_init__(self):
        super().__post_init__()

        # Robot
        self.scene.robot = UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = UNITREE_G1_29DOF_ACTION_SCALE

        # Motion (anchor is pelvis to match pre-processed motion data)
        self.commands.motion.motion_file = f"{os.path.dirname(__file__)}/motion/"
        self.commands.motion.anchor_body_name = "pelvis"
        self.commands.motion.body_names = [
            "pelvis",
            "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
            "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
            "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
        ]
        self.commands.motion.amp_obs_terms = amp_groups.AMPObsHardTrackTerms
        self.episode_length_s = 30.0

        # Override Cyborg-specific event body names for G1
        self.events.randomize_com_positions.params["asset_cfg"].body_names = ("pelvis",)

        # Override feet_slide body names
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [
            "left_ankle_roll_link", "right_ankle_roll_link",
        ]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [
            "left_ankle_roll_link", "right_ankle_roll_link",
        ]

        # Override self_collisions body names (use torso_link for upper body collisions)
        self.rewards.self_collisions.params["sensor_cfg"].body_names = [
            "pelvis",
            "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
            "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
        ]

        # G1 termination: lower minimum height (pelvis at 0.76m crouch)
        self.terminations.bad_base_height.params["minimum_height"] = 0.55

        # G1 track_anchor velocity/anchor config (pelvis as anchor)
        self.rewards.track_anchor_linear_velocity.params["anchor_cfg"].body_names = ["pelvis"]
        self.rewards.track_anchor_angular_velocity.params["anchor_cfg"].body_names = ["pelvis"]
        self.rewards.body_ang_vel_xy_l2.params["body_cfg"].body_names = ["pelvis"]

        # Disable track_head_height (Cyborg-specific)
        self.rewards.track_head_height.weight = 0.0
