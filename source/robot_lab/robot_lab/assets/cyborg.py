import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from robot_lab.assets import ISAACLAB_ASSETS_DATA_DIR


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

CYBORG_BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/cyborg/biped_temp_1_0/urdf/biped_temp_1_0.urdf",
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
            ".*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_hip_.*_roll",
                "J_hip_.*_yaw",
                "J_hip_.*_pitch",
                "J_knee_.*_pitch",
            ],
            effort_limit_sim={
                "J_hip_.*_roll": 330.0,
                "J_hip_.*_yaw": 330.0,
                "J_hip_.*_pitch": 330.0,
                "J_knee_.*_pitch": 330.0,
            },
            velocity_limit_sim={
                "J_hip_.*_roll": 12.04,
                "J_hip_.*_yaw": 12.04,
                "J_hip_.*_pitch": 12.04,
                "J_knee_.*_pitch": 12.04,
            },
            stiffness={
                "J_hip_.*_roll": 250.0,
                "J_hip_.*_yaw": 120.0,
                "J_hip_.*_pitch": 300.0,
                "J_knee_.*_pitch": 300.0,
            },
            damping={
                "J_hip_.*_roll": 10.0,
                "J_hip_.*_yaw": 10.0,
                "J_hip_.*_pitch": 10.0,
                "J_knee_.*_pitch": 10.0,
            },
            armature={
                ".*": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=120.0,
            velocity_limit_sim=11.21,
            joint_names_expr=["J_ankle_.*_pitch", "J_ankle_.*_roll"],
            stiffness=80.0,
            damping=3.0,
            armature=0.01,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J_waist_yaw", "J_waist_pitch"],
            effort_limit_sim={
                "J_waist_yaw": 144.0,
                "J_waist_pitch": 165.0,
            },
            velocity_limit_sim={
                "J_waist_yaw": 3.14,
                "J_waist_pitch": 2.41,
            },
            stiffness=100.0,
            damping=5.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                "J_arm_.*_01",
                "J_arm_.*_02",
                "J_arm_.*_03",
                "J_arm_.*_04",
                "J_arm_.*_05",
                "J_arm_.*_06",
                "J_arm_.*_07",
            ],
            effort_limit_sim={
                "J_arm_.*_01": 144.0,
                "J_arm_.*_02": 89.0,
                "J_arm_.*_03": 60.0,
                "J_arm_.*_04": 60.0,
                "J_arm_.*_05": 31.0,
                "J_arm_.*_06": 11.0,
                "J_arm_.*_07": 11.0,
            },
            velocity_limit_sim={
                "J_arm_.*_01": 3.14,
                "J_arm_.*_02": 3.14,
                "J_arm_.*_03": 3.04,
                "J_arm_.*_04": 3.04,
                "J_arm_.*_05": 3.45,
                "J_arm_.*_06": 3.45,
                "J_arm_.*_07": 3.45,
            },
            stiffness=50.0,
            damping=2.0,
            armature=0.005,
        ),
    },
)

CYBORG_BIPED_ACTION_SCALE = {}
for a in CYBORG_BIPED_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            CYBORG_BIPED_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
