import numpy as np
import torch
import yaml
import mujoco
import mujoco_viewer
from collections import deque
from tqdm import tqdm

# Joint order the policy expects (matches MuJoCo actuator order 0-11)
LEG_JOINT_NAMES = [
    "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch",
    "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll",
    "J_hip_r_roll", "J_hip_r_yaw", "J_hip_r_pitch",
    "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll",
]


def quaternion_to_euler(w, x, y, z):
    """Quaternion (w,x,y,z) -> roll, pitch, yaw (same as C++ version)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if np.abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float32)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


class EncosSim2Sim:
    def __init__(self, xml_path, policy_path, config_yaml_path):
        # ── Load YAML config ──
        with open(config_yaml_path, "r") as f:
            self.cfg = yaml.safe_load(f)["CyborgWalkCfg"]

        # ── MuJoCo ──
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)

        # Joint / actuator IDs for the 12 leg joints
        self.leg_joint_ids = np.array([
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in LEG_JOINT_NAMES
        ])
        self.leg_qpos_adr = np.array([
            self.m.jnt_qposadr[jid] for jid in self.leg_joint_ids
        ])
        self.leg_dof_adr = np.array([
            self.m.jnt_dofadr[jid] for jid in self.leg_joint_ids
        ])

        # Base link body id for IMU / orientation
        self.base_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "base_link")

        # ── Default joint angles ──
        yaml_default_angles = self.cfg["init_state"]["default_joint_angle"]
        self.default_angle = np.array([
            yaml_default_angles[yml_name]
            for yml_name in [
                "l_hip_roll_joint", "l_hip_yaw_joint", "l_hip_pitch_joint",
                "l_knee_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
                "r_hip_roll_joint", "r_hip_yaw_joint", "r_hip_pitch_joint",
                "r_knee_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
            ]
        ], dtype=np.float32)

        # ── KP / KD (YAML keys are like "l_hip_roll_joint") ──
        stiffness = self.cfg["control"]["stiffness"]
        damping = self.cfg["control"]["damping"]
        self.kp = np.array([stiffness[k] for k in [
            "l_hip_roll_joint", "l_hip_yaw_joint", "l_hip_pitch_joint",
            "l_knee_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
            "r_hip_roll_joint", "r_hip_yaw_joint", "r_hip_pitch_joint",
            "r_knee_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
        ]], dtype=np.float32)
        self.kd = np.array([damping[k] for k in [
            "l_hip_roll_joint", "l_hip_yaw_joint", "l_hip_pitch_joint",
            "l_knee_joint", "l_ankle_pitch_joint", "l_ankle_roll_joint",
            "r_hip_roll_joint", "r_hip_yaw_joint", "r_hip_pitch_joint",
            "r_knee_joint", "r_ankle_pitch_joint", "r_ankle_roll_joint",
        ]], dtype=np.float32)

        # ── Action scale ──
        self.action_scale = np.array(self.cfg["control"]["action_scale"], dtype=np.float32)

        # ── Observation config ──
        self.obs_names = self.cfg["observations"]  # list like ["sin_cos", "commands", ...]
        self.num_single_obs = self.cfg["size"]["num_single_obs"]  # 47
        self.frame_stack = self.cfg["size"]["frame_stack"]  # 15
        self.num_actions = self.cfg["size"]["actions_size"]  # 12

        # ── Normalization scales ──
        osc = self.cfg["normalization"]["obs_scales"]
        self.lin_vel_scale = osc["lin_vel"]  # 2.0
        self.ang_vel_scale = osc["ang_vel"]  # 1.0
        self.dof_pos_scale = osc["dof_pos"]  # 1.0
        self.dof_vel_scale = osc["dof_vel"]  # 0.05
        csc = self.cfg["normalization"]["clip_scales"]
        self.clip_obs = csc["clip_observations"]  # 50.0
        self.clip_actions = csc["clip_actions"]  # 5.0

        # ── Control timing ──
        self.cycle_time = self.cfg["control"]["cycle_time"]  # 0.8
        self.dt = 0.01  # 100Hz
        self.decimation = self.cfg["control"]["decimation"]  # 10 (sim dt=0.001, ctrl dt=0.01)

        # ── Load policy ──
        self.device = torch.device("cpu")
        self.policy = torch.jit.load(policy_path).eval()
        print(f"[EncosSim2Sim] Policy loaded from {policy_path}")

        # ── State ──
        self.lowlevel_cnt = 0
        self.current_phase = 0.0
        self.hist_obs = deque(maxlen=self.frame_stack)
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.target_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)  # default: walk forward

        # ── Reset & viewer ──
        self._reset()
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -15

    def _reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        for i, qpos_adr in enumerate(self.leg_qpos_adr):
            self.d.qpos[qpos_adr] = self.default_angle[i]
        # ── Auto-adjust pelvis height so feet touch ground ──
        mujoco.mj_forward(self.m, self.d)
        ankle_body_names = ["ankle_l_roll_link", "ankle_r_roll_link"]
        foot_z = min(
            self.d.xpos[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, name)][2]
            for name in ankle_body_names
        )
        self.d.qpos[2] -= foot_z - 0.005  # 5mm above ground to avoid interpenetration
        mujoco.mj_forward(self.m, self.d)
        self.hist_obs.clear()
        for _ in range(self.frame_stack):
            self.hist_obs.append(np.zeros(self.num_single_obs, dtype=np.float32))
        self.lowlevel_cnt = 0
        self.current_phase = 0.0
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

    def get_joint_state(self):
        """Return leg joint positions and velocities."""
        q = np.array([self.d.qpos[adr] for adr in self.leg_qpos_adr], dtype=np.float32)
        dq_arr = np.array([self.d.qvel[adr] for adr in self.leg_dof_adr], dtype=np.float32)
        return q, dq_arr

    def get_imu_data(self):
        """Return base quaternion (w,x,y,z) and body-frame angular velocity."""
        # Base quaternion is stored as (w,x,y,z) in xquat
        quat = self.d.xquat[self.base_body_id].copy()  # (w, x, y, z)
        # cvel gives [lin_vel_world(3), ang_vel_world(3)]
        ang_vel_world = self.d.cvel[self.base_body_id][3:6].copy()
        # Rotate angular velocity to body frame
        R = self.d.xmat[self.base_body_id].reshape(3, 3)
        ang_vel_body = R.T @ ang_vel_world
        return quat, ang_vel_body.astype(np.float32)

    def compute_obs(self):
        self.lowlevel_cnt += 1

        # Phase (same as C++ ComputeObs)
        phase = 2.0 * np.pi * self.lowlevel_cnt * self.dt / self.cycle_time
        gait_freq = 0.8 + 0.15 * abs(self.target_cmd[0])
        self.current_phase = np.fmod(self.current_phase + self.dt * gait_freq, 1.0)
        gait_phase = 2.0 * np.pi * self.current_phase

        out_sin = np.sin(phase)
        out_cos = np.cos(phase)
        gait_sin = np.sin(gait_phase)
        gait_cos = np.cos(gait_phase)

        # Scaled commands
        cmd_scaled = np.array([
            self.target_cmd[0] * self.lin_vel_scale,
            self.target_cmd[1] * self.lin_vel_scale,
            self.target_cmd[2] * self.ang_vel_scale,
        ], dtype=np.float32)

        # Joint state
        q, dq_arr = self.get_joint_state()
        quat, ang_vel_body = self.get_imu_data()

        # Euler angles from quaternion
        eu_ang = quaternion_to_euler(quat[0], quat[1], quat[2], quat[3])

        # Build observation dict (same logic as C++ ComputeObs)
        obs_list = []
        for obs_name in self.obs_names:
            if obs_name == "ang_vel":
                obs_list.append(ang_vel_body * self.ang_vel_scale)
            elif obs_name == "commands":
                obs_list.append(cmd_scaled)
            elif obs_name == "dof_pos":
                dof_pos_rel = (q - self.default_angle) * self.dof_pos_scale
                obs_list.append(dof_pos_rel)
            elif obs_name == "dof_vel":
                obs_list.append(dq_arr * self.dof_vel_scale)
            elif obs_name == "actions":
                obs_list.append(self.last_action.copy())
            elif obs_name == "sin_cos":
                obs_list.append(np.array([out_sin, out_cos], dtype=np.float32))
            elif obs_name == "eu_ang":
                obs_list.append(eu_ang)
            elif obs_name == "gait_obs":
                gait_offset = 0.5
                gait_height = 0.07
                obs_list.append(np.array(
                    [gait_freq, gait_offset, gait_height, gait_sin, gait_cos],
                    dtype=np.float32,
                ))

        obs = np.concatenate(obs_list).astype(np.float32)
        obs = np.clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def build_policy_input(self, obs):
        self.hist_obs.append(obs)
        # Flatten frame_stack * num_single_obs
        policy_input = []
        for h in self.hist_obs:
            policy_input.extend(h)
        return np.array(policy_input, dtype=np.float32)

    def run_inference(self, policy_input):
        tensor = torch.from_numpy(policy_input).unsqueeze(0).float()
        with torch.no_grad():
            output = self.policy.forward(tensor)
            if isinstance(output, tuple):
                action_tensor = output[0]
            else:
                action_tensor = output
        action = action_tensor.squeeze(0).numpy().astype(np.float32)
        action = np.clip(action, -self.clip_actions, self.clip_actions)
        return action

    def policy_step(self):
        obs = self.compute_obs()
        policy_input = self.build_policy_input(obs)
        action = self.run_inference(policy_input)
        self.last_action = action

        # action -> target joint position
        target_q = action * self.action_scale + self.default_angle
        return target_q

    def _hold_non_leg_joints(self):
        """PD-hold waist + arms at their default positions."""
        for j, act_id in enumerate(self._non_leg_actuator_ids):
            jid = self._non_leg_joint_ids[j]
            q_adr = self.m.jnt_qposadr[jid]
            dq_adr = self.m.jnt_dofadr[jid]
            pos_err = self._non_leg_default[j] - self.d.qpos[q_adr]
            vel_err = 0.0 - self.d.qvel[dq_adr]
            self.d.ctrl[act_id] = pos_err * 200.0 + vel_err * 10.0

    def _apply_leg_torques(self, target_q):
        """PD control for the 12 leg joints."""
        q, dq_arr = self.get_joint_state()
        torque = pd_control(target_q, q, self.kp,
                            np.zeros(self.num_actions), dq_arr, self.kd)
        for j, t in enumerate(torque):
            self.d.ctrl[j] = t

    def _settle(self, duration=0.5):
        """Drop robot onto ground with PD holding default pose, no policy."""
        steps = int(duration / self.m.opt.timestep)
        target_q = self.default_angle.copy()
        for _ in range(steps):
            self._apply_leg_torques(target_q)
            self._hold_non_leg_joints()
            mujoco.mj_step(self.m, self.d)

    def _warmup(self, duration=2.0):
        """Fill observation history with standing data. No policy yet — just
        PD-hold the default pose with zero command so the robot stabilises
        and the frame-stack buffer fills with consistent standing obs."""
        n_steps = int(duration / self.dt)
        saved_cmd = self.target_cmd.copy()
        self.target_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.lowlevel_cnt = 0
        self.current_phase = 0.0
        self.hist_obs.clear()
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)

        target_q = self.default_angle.copy()
        for _ in tqdm(range(n_steps), desc="Warming up"):
            obs = self.compute_obs()
            self.hist_obs.append(obs)
            self._apply_leg_torques(target_q)
            self._hold_non_leg_joints()
            mujoco.mj_step(self.m, self.d)

        self.target_cmd = saved_cmd

    def run(self, sim_duration=60.0):
        sim_dt = self.m.opt.timestep  # 0.001
        sim_steps = int(sim_duration / sim_dt)
        sim_decimation = self.decimation  # 10

        # ── Identify non-leg joints once ──
        all_actuator_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.m.nu)
        ]
        self._non_leg_actuator_ids = [
            i for i, name in enumerate(all_actuator_names)
            if name not in set(LEG_JOINT_NAMES)
        ]
        self._non_leg_joint_ids = [
            mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT,
                              all_actuator_names[i])
            for i in self._non_leg_actuator_ids
        ]
        non_leg_joint_names = [all_actuator_names[i] for i in self._non_leg_actuator_ids]
        upper_body_pose = {
            "J_arm_l_02": -1.4, "J_arm_l_04": -2.09, "J_arm_l_06": 0.9,
            "J_arm_r_02":  1.4, "J_arm_r_04":  2.09, "J_arm_r_06": -0.9,
        }
        self._non_leg_default = np.array([
            upper_body_pose.get(name, 0.0) for name in non_leg_joint_names
        ])
        for j, jid in enumerate(self._non_leg_joint_ids):
            self.d.qpos[self.m.jnt_qposadr[jid]] = self._non_leg_default[j]

        # ── Settle + warmup ──
        print("[EncosSim2Sim] Settling (PD holding default pose)...")
        self._settle(0.5)
        print("[EncosSim2Sim] Warming up (filling observation history)...")
        self._warmup(2.0)

        # ── Main loop ──
        target_q = self.default_angle.copy()
        for i in tqdm(range(sim_steps), desc="Encos sim2sim"):
            if i % sim_decimation == 0:
                target_q = self.policy_step()
                self.viewer.cam.lookat[:] = self.d.qpos[0:3].copy()
                self.viewer.render()

            self._apply_leg_torques(target_q)
            self._hold_non_leg_joints()
            mujoco.mj_step(self.m, self.d)

        self.viewer.close()


# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--xml",
                        default="/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/temp/biped_temp_1_0.xml")
    parser.add_argument("--policy",
                        default="/home/cyborg/Desktop/projects/sim2real_encos/src/deploy_real/policies/c10000.pt")
    parser.add_argument("--config",
                        default="/home/cyborg/Desktop/projects/sim2real_encos/src/deploy_real/config/Cyborg_Encos_Config.yaml")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="Simulation duration in seconds")
    args = parser.parse_args()

    sim = EncosSim2Sim(args.xml, args.policy, args.config)
    sim.target_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)  # vx, vy, vyaw
    sim.run(sim_duration=args.duration)
