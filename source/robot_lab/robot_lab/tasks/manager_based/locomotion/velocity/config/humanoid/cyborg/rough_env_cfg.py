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
            delay_steps=(1, 10),
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
        self.rewards.lin_vel_z_l2.weight = 0
        self.rewards.ang_vel_xy_l2.weight = -0.1
        self.rewards.flat_orientation_l2.weight = -0.2
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -1.5e-7
        self.rewards.joint_torques_l2.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*", "J_ankle_.*"]
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -1.25e-7
        self.rewards.joint_acc_l2.params["asset_cfg"].joint_names = ["J_hip_.*", "J_knee_.*"]
        # 只保留髋关节偏移惩罚（Cyborg HP 无臂无腰）
        self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -1.0, ["J_hip_.*_yaw", "J_hip_.*_roll"])
        self.rewards.joint_pos_limits.weight = -0.5
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = 0
        self.rewards.stand_still.weight = -1.0
        self.rewards.stand_still.params["recovery_tilt_threshold"] = 0.17
        # joint_pos_penalty 已关闭（weight 0）：运动时与 phase_ref_joint_pos 对抗，静止时与 stand_still 重复
        self.rewards.joint_mirror.weight = 0
        self.rewards.joint_mirror.params["mirror_joints"] = [["J_.*_l_.*", "J_.*_r_.*"]]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.action_mirror.weight = 0
        self.rewards.action_mirror.params["mirror_joints"] = [["J_.*_l_.*", "J_.*_r_.*"]]

        # Contact sensor
        self.rewards.undesired_contacts.weight = 0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = 0
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_ang_vel_z_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_world_exp

        # Others
        self.rewards.feet_air_time.weight = 0.25
        self.rewards.feet_air_time.func = mdp.feet_air_time_positive_biped
        self.rewards.feet_air_time.params["threshold"] = 0.4
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -10.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        # feet_slide 权重对齐 EngineAI（-0.1），实现已对齐（世界系 + √ 核 + 垂直力>5N）
        self.rewards.feet_slide.weight = -0.1
        # 触地瞬间垂直速度惩罚（触地沿帧罚 (|vz|-0.1).clip(0)）：
        # 奖励景观里唯一的"触地瞬间"维度，防止后期收敛到重砸局部最优。
        # 0.5 m/s 触地罚 0.4/脚 ×2.0；ENCOS 基准 vz 0.03-0.06 < 阈值不罚
        self.rewards.feet_landing_velocity.weight = -2.0
        self.rewards.feet_landing_velocity.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_landing_velocity.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 步宽约束（防交叉脚）：期望两脚 y 距离 0.31 m（default 位姿步宽），body 顺序须为 [左, 右]
        self.rewards.feet_distance_y_exp.weight = 0.5
        self.rewards.feet_distance_y_exp.params["std"] = 0.15
        self.rewards.feet_distance_y_exp.params["stance_width"] = 0.31
        self.rewards.feet_distance_y_exp.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_distance_y_exp.params["asset_cfg"].preserve_order = True
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height.params["target_height"] = 0.12
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = 0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.upward.weight = 1.0
        # EngineAI 式双向（匹配 +1 / 不匹配 -0.3），weight 对齐 EN feet_contact_number=1.4；
        # recovery 倾斜已去掉（仅 moving 激活）：与 phase_ref 一致，消除"站立倾斜切换
        # 相位接触要求"的 hack 杠杆；0 命令时恒定要求双脚着地
        self.rewards.periodic_contact_mismatch.weight = 1.4
        self.rewards.periodic_contact_mismatch.params["sensor_cfg"].body_names = ["ankle_l_roll_link", "ankle_r_roll_link"]
        self.rewards.periodic_contact_mismatch.params["sensor_cfg"].preserve_order = True
        # phase_feet_height 已关闭（weight 0）：与 phase_ref_joint_pos 同一摆动窗口双重约束，冗余
        self.rewards.phase_ref_joint_pos.weight = 2.0
        self.rewards.phase_ref_joint_pos.params["recovery_tilt_threshold"] = 0.17
        # 膝摆幅 0.45（16-26-39 实测抬脚 67mm 档位）+ 踝 pitch 0.15 背屈参与
        # （对齐 ENCOS 踝实摆 0.13，落地缓冲）；踝 roll 仍锁 0（额状面稳）
        self.rewards.phase_ref_joint_pos.params["knee_scale"] = 0.45
        self.rewards.phase_ref_joint_pos.params["ankle_scale"] = 0.15
        self.rewards.phase_ref_joint_pos.params["ankle_roll_scale"] = 0.0

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
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)

        # ------------------------------Episode------------------------------
        self.episode_length_s = 30.0
