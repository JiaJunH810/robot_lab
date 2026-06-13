from __future__ import annotations

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass
import robot_lab.tasks.manager_based.beyondamp.mdp as mdp

# -------------------- Terms和Cfg中字段顺序保持一致 -------------------- #

# ---- 判别器观测项键名 ----
# 供 sample_expert_transition() 使用，指定拼接顺序和字段
AMPObsHardTrackTerms = [
    "body_pos_b", "body_ori_b", "body_lin_vel_b", "body_ang_vel_b",
    "body_pos_w_rel_z",
]

@configclass
class AMPObsBodyHardTrackCfg(ObsGroup):
    body_pos_b = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
    body_ori_b = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
    body_lin_vel_b = ObsTerm(func=mdp.robot_body_lin_vel_b, params={"command_name": "motion"})
    body_ang_vel_b = ObsTerm(func=mdp.robot_body_ang_vel_b, params={"command_name": "motion"})
    body_pos_w_rel_z = ObsTerm(func=mdp.robot_body_pos_w_rel_z, params={"command_name": "motion"})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
