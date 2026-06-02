import os
# ---- Cyborg 机器人常量（AMP runner 配置复用）----

CYBORG_KEY_BODY_NAMES = [
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
CYBORG_ANCHOR_NAME = "base_link"
CYBORG_MOTION_FILE = os.path.join(os.path.dirname(__file__), "motion", "B1_-_stand_to_walk_stageii.npz")