import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


# =============================================================================
# Cyborg Half-Ped (下半身 only, 12 DoF): 髋/膝/踝
# Armature: 等效转动惯量 (kg·m²)
# 电机型号对应关节：
#   10020 (ratio 24): 髋、膝
#   6416  (ratio 25): 踝
# =============================================================================
ARMATURE_10020_24 = 0.02773762228   # 10020, ratio 24
ARMATURE_6416     = 0.06581459643   # 6416,  ratio 25

# =============================================================================
# PD 增益: stiffness = armature * ω², damping = 2ζ * armature * ω
# ω = 10 Hz * 2π, ζ = 2.0 (过阻尼)
# =============================================================================
NATURAL_FREQ = 10 * 2.0 * 3.141592653589793
DAMPING_RATIO = 2.0

STIFFNESS_10020_24 = ARMATURE_10020_24 * NATURAL_FREQ ** 2
STIFFNESS_6416     = ARMATURE_6416     * NATURAL_FREQ ** 2

DAMPING_10020_24 = 2.0 * DAMPING_RATIO * ARMATURE_10020_24 * NATURAL_FREQ
DAMPING_6416     = 2.0 * DAMPING_RATIO * ARMATURE_6416     * NATURAL_FREQ


CYBORG_HALF_PED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/cyborg/biped_temp_1_0/urdf/biped_temp_1_0_fixed.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.94),
        joint_pos={
            "J_hip_.*_roll": 0.0,
            "J_hip_.*_yaw": 0.0,
            "J_hip_l_pitch": 0.4,
            "J_knee_l_pitch": 0.7,
            "J_ankle_l_pitch": 0.3,
            "J_ankle_l_roll": 0.0,
            "J_hip_r_pitch": -0.4,
            "J_knee_r_pitch": -0.7,
            "J_ankle_r_pitch": -0.3,
            "J_ankle_r_roll": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        # 髋 + 膝: EC-A10020-P2-24
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_hip_.*_roll",
                "J_hip_.*_yaw",
                "J_hip_.*_pitch",
                "J_knee_.*_pitch",
            ],
            effort_limit_sim=330.0,
            velocity_limit_sim=12.043,
            stiffness=STIFFNESS_10020_24,
            damping=DAMPING_10020_24,
            armature=ARMATURE_10020_24,
        ),
        # 踝: EC-A6416-P2-25 (ratio 25)
        "feet": ImplicitActuatorCfg(
            joint_names_expr=["J_ankle_.*_pitch", "J_ankle_.*_roll"],
            effort_limit_sim=120.0,
            velocity_limit_sim=11.205,
            stiffness=STIFFNESS_6416,
            damping=DAMPING_6416,
            armature=ARMATURE_6416,
        ),
    },
)

CYBORG_HALF_PED_ACTION_SCALE = {}
for a in CYBORG_HALF_PED_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            CYBORG_HALF_PED_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
