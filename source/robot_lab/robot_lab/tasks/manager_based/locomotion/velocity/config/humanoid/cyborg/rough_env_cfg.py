# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

import robot_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from robot_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
from robot_lab.assets.cyborg_hp import CYBORG_HALF_PED_ACTION_SCALE, CYBORG_HALF_PED_CFG  # isort: skip


@configclass
class CyborgHPRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base_link"
    foot_link_name = "ankle_.*_roll_link"

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Scene------------------------------
        self.scene.robot = CYBORG_HALF_PED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.history_length = 15
        self.observations.critic.history_length = 15
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None

        # ------------------------------Actions------------------------------
        self.actions.joint_pos = mdp.DelayedJointPositionActionCfg(
            asset_name="robot", 
            joint_names=[".*"], 
            scale=CYBORG_HALF_PED_ACTION_SCALE,
            use_default_offset=True, 
            clip={".*": (-100.0, 100.0)}, 
            preserve_order=True,
            delay_steps=(1, 20),
            joint_obs_delay_steps=(1, 10),
            imu_obs_delay_steps=(1, 10)
        )

        # ------------------------------Events------------------------------
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = -200.0

        # Root penalties
        self.rewards.vel_mismatch_exp.weight = 0.1
        self.rewards.base_acc.weight = 0.2
        self.rewards.flat_orientation_l2.func = mdp.orientation_exp
        self.rewards.flat_orientation_l2.weight = 0.3
        self.rewards.flat_orientation_l2.params["tolerance"] = 0.069756
        # base_height_l2 -10.0（防塌锚定，替代 stand_still 软化后的防塌角色）：
        # 塌 5cm 罚 0.025/步、塌 10cm 罚 0.10/步；柔顺变形 1-2cm 几乎不罚
        self.rewards.base_height_l2.weight = -25.0
        self.rewards.base_height_l2.params["target_height"] = 0.89
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-6
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*", "J_ankle_.*",]
        # joint_vel_l2 -1e-05（08-18 柔顺措施）：抑制关节速度，动作更柔和
        self.rewards.joint_vel_l2.weight = -1e-5
        self.rewards.joint_acc_l2.weight = -1e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*", "J_ankle_.*",]
        # 只保留髋关节偏移惩罚（Cyborg HP 无臂无腰）
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -1.25, ["J_hip_.*_yaw", "J_hip_.*_roll"])
        self.rewards.joint_pos_limits.weight = -0.5
        # stand_still -1.0 → -0.3（软化）：被压时关节允许变形（柔顺），防塌由 base_height_l2 接管
        self.rewards.stand_still.weight = -0.1
        self.rewards.stand_still.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*"]

        # Action penalties
        self.rewards.action_smoothness.weight = -0.003

        # Contact sensor
        self.rewards.contact_forces.weight = -5.0e-4
        self.rewards.contact_forces.params["threshold"] = 2000
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_x_exp.weight = 0.3
        self.rewards.track_lin_vel_y_exp.weight = 0.3
        self.rewards.track_ang_vel_z_exp.weight = 0.3
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.1
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp
        self.rewards.phase_ref_joint_pos.weight = 1.0
        self.rewards.periodic_contact_mismatch.weight = -0.3
        self.rewards.periodic_contact_mismatch.params["sensor_cfg"].body_names = ["ankle_l_roll_link", "ankle_r_roll_link"]
        self.rewards.periodic_contact_mismatch.params["sensor_cfg"].preserve_order = True

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_orientation_l2.weight = -2.0
        self.rewards.feet_orientation_l2.params["asset_cfg"].body_names = [self.foot_link_name]
        # 步宽约束（防交叉脚）：期望两脚 y 距离 0.31 m（default 位姿步宽），body 顺序须为 [左, 右]
        self.rewards.feet_distance_y_exp.weight = 0.5
        self.rewards.feet_distance_y_exp.params["std"] = 0.15
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.31
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_y_exp.params["asset_cfg"].preserve_order = True

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "CyborgHPRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        self.terminations.illegal_contact = None
        self.terminations.bad_base_height.params["minimum_height"] = 0.65

        # ------------------------------Curriculums------------------------------
        # Enabled: start at 10% velocity range, scale up to 100% as tracking improves

        # ------------------------------Commands------------------------------
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.5
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ------------------------------Episode------------------------------
        self.episode_length_s = 30.0
