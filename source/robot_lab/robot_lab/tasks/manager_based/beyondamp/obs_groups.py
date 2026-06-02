from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
import robot_lab.tasks.manager_based.beyondamp.mdp as mdp

# -------------------- Terms和Cfg中字段顺序保持一致 -------------------- #

# ---- 判别器观测项键名 ----
# 供 sample_expert_transition() 使用，指定拼接顺序和字段
AMPObsHardTrackTerms = [
    "joint_pos", "joint_vel",
    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w",
]

@configclass
class AMPObsBodyHardTrackCfg(ObsGroup):
    joint_pos = ObsTerm(func=mdp.joint_pos_rel)
    joint_vel = ObsTerm(func=mdp.joint_vel_rel)
    body_pos_w = ObsTerm(func=mdp.robot_body_pos_w, params={"command_name": "motion"})
    body_quat_w = ObsTerm(func=mdp.robot_body_quat_w, params={"command_name": "motion"})
    body_lin_vel_w = ObsTerm(func=mdp.robot_body_lin_vel_w, params={"command_name": "motion"})
    body_ang_vel_w = ObsTerm(func=mdp.robot_body_ang_vel_w, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
