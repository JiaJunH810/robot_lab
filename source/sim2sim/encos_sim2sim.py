import time
import math
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch


def quaternion_to_euler(q_wxyz):
    """q_wxyz: [w, x, y, z] (MuJoCo convention). Returns [roll, pitch, yaw]."""
    w, x, y, z = q_wxyz
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return np.array([roll, pitch, yaw])


class EncosSim2Sim:
    def __init__(self, xml_path, policy_path, cmd_vx=0.1, cmd_vy=0.0, cmd_vyaw=0.0):
        # ---- MuJoCo ----
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        self.num_leg = 12  # policy only controls leg joints (actuator indices 0-11)
        self.num_all_actuators = self.m.nu

        # ---- 上半身固定位姿（actuator 索引 12-27） ----
        # 12:J_waist_yaw  13:J_waist_pitch
        # 14:J_arm_l_01  15:J_arm_l_02  16:J_arm_l_03  17:J_arm_l_04
        # 18:J_arm_l_05  19:J_arm_l_06  20:J_arm_l_07
        # 21:J_arm_r_01  22:J_arm_r_02  23:J_arm_r_03  24:J_arm_r_04
        # 25:J_arm_r_05  26:J_arm_r_06  27:J_arm_r_07
        self.upper_body_pose = np.zeros(self.num_all_actuators - self.num_leg)
        self.upper_body_pose[3]  = -1.4   # J_arm_l_02 (act idx 15)
        self.upper_body_pose[5]  = -2.09  # J_arm_l_04 (act idx 17)
        self.upper_body_pose[7]  =  0.9   # J_arm_l_06 (act idx 19)
        self.upper_body_pose[10] =  1.4   # J_arm_r_02 (act idx 22)
        self.upper_body_pose[12] =  2.09  # J_arm_r_04 (act idx 24)
        self.upper_body_pose[14] = -0.9   # J_arm_r_06 (act idx 26)

        # ---- 获取 XML 中腿关节名（前 12 个 actuator） ----
        self.leg_joint_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.num_leg)
        ]

        # ---- Policy 参数（与 Cyborg_Encos_Config.yaml 对齐） ----
        self.default_angle = np.array([
            0.0, 0.0,  0.4,  0.7,  0.3, 0.0,   # left
            0.0, 0.0, -0.4, -0.7, -0.3, 0.0,   # right
        ])
        self.action_scale = np.array([
            0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
            0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
        ])
        self.kp = np.array([
            250, 120, 300, 300, 80, 80,
            250, 120, 300, 300, 80, 80,
        ])
        self.kd = np.array([
            10, 10, 10, 10, 3, 3,
            10, 10, 10, 10, 3, 3,
        ])
        self.cycle_time = 0.8
        self.control_dt = 0.01          # 100 Hz
        self.sim_decimation = 10        # 0.001 * 10 = 0.01
        self.clip_obs = 50.0
        self.clip_act = 5.0
        self.frame_stack = 15
        self.num_single_obs = 47
        self.num_observations = self.frame_stack * self.num_single_obs  # 705

        # ---- 指令速度 ----
        self.cmd_vx = cmd_vx
        self.cmd_vy = cmd_vy
        self.cmd_vyaw = cmd_vyaw

        # ---- 加载策略 ----
        self.policy = torch.jit.load(policy_path).eval()
        print(f"[EncosSim2Sim] Policy loaded from {policy_path}")

        # ---- 初始化 MuJoCo ----
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        self._set_default_pose()
        mujoco.mj_step(self.m, self.d)

        # ---- Viewer ----
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        self.viewer.cam.distance = 2.0
        self.viewer.cam.lookat = [0, 0, 0.5]

        # ---- 观测历史 ----
        self.hist_obs = deque(
            [np.zeros(self.num_single_obs) for _ in range(self.frame_stack)],
            maxlen=self.frame_stack,
        )
        self.last_action = np.zeros(self.num_leg)
        self.lowlevel_cnt = 0

    def _set_default_pose(self):
        """设置初始站立位姿。qpos 布局: [px,py,pz, qw,qx,qy,qz,  joint0, ..., joint27]"""
        qpos = self.d.qpos.copy()
        qpos[2] = 0.94  # base height
        qpos[3:7] = [1, 0, 0, 0]  # identity quat
        # 腿
        for i in range(self.num_leg):
            qpos[7 + i] = self.default_angle[i]
        # 上半身（腰+臂）：设为固定位姿
        for i in range(len(self.upper_body_pose)):
            qpos[7 + self.num_leg + i] = self.upper_body_pose[i]
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.0

    def _clip(self, val, lo, hi):
        return np.clip(val, lo, hi)

    def _compute_obs(self):
        """与 C++ ComputeObs() 对齐: sin_cos, commands, dof_pos, dof_vel, actions, ang_vel, eu_ang"""
        phase = 2.0 * math.pi * self.lowlevel_cnt * self.control_dt / self.cycle_time

        # -- sin_cos --
        sin_cos = np.array([math.sin(phase), math.cos(phase)])

        # -- commands (C++: cmd.command * obs_scales) --
        commands = np.array([
            self.cmd_vx * 2.0,
            self.cmd_vy * 2.0,
            self.cmd_vyaw * 1.0,
        ])

        # -- joint state (只取腿关节, qpos[7:7+12]) --
        q = self.d.qpos[7:7 + self.num_leg].copy()
        dq = self.d.qvel[6:6 + self.num_leg].copy()  # qvel[0:3]=lin vel, [3:6]=ang vel
        dof_pos = q - self.default_angle       # * 1.0
        dof_vel = dq * 0.05

        # -- actions --
        actions = self.last_action.copy()

        # -- ang_vel (body frame) --
        ang_vel = self.d.qvel[3:6].copy()       # * 1.0

        # -- eu_ang --
        base_quat = self.d.qpos[3:7].copy()     # [qw, qx, qy, qz]
        eu_ang = quaternion_to_euler(base_quat)
        eu_ang = np.where(eu_ang > math.pi, eu_ang - 2 * math.pi, eu_ang)

        obs = np.concatenate([
            sin_cos,      # 2
            commands,     # 3
            dof_pos,      # 12
            dof_vel,      # 12
            actions,      # 12
            ang_vel,      # 3
            eu_ang,       # 3
        ])  # 47

        obs = self._clip(obs, -self.clip_obs, self.clip_obs)
        return obs

    def _build_policy_input(self):
        """帧堆叠: [f0_47 | f1_47 | ... | f14_47], 最老在前"""
        policy_input = np.zeros(self.num_observations, dtype=np.float32)
        for i, obs in enumerate(self.hist_obs):
            policy_input[i * self.num_single_obs:(i + 1) * self.num_single_obs] = obs
        return policy_input.reshape(1, -1)

    def _forward(self, policy_input):
        """TorchScript 推理，返回 12 维 action 并 clip"""
        with torch.no_grad():
            t = torch.from_numpy(policy_input)
            output = self.policy.forward(t)
            if isinstance(output, tuple):
                action_tensor = output[0]
            else:
                action_tensor = output
            action = action_tensor.detach().numpy().squeeze()
        action = self._clip(action, -self.clip_act, self.clip_act)
        return action

    def step(self):
        """单次控制步: 取观测 → 推策略 → PD → 发扭矩"""
        # ---- 计算观测 & 帧堆叠 ----
        obs = self._compute_obs()
        self.hist_obs.append(obs)
        policy_input = self._build_policy_input()

        # ---- 推理 ----
        self.last_action = self._forward(policy_input)

        # ---- action → target_q ----
        target_q = self.last_action * self.action_scale + self.default_angle

        # ---- PD 控制（仅腿关节） ----
        q = self.d.qpos[7:7 + self.num_leg]
        dq = self.d.qvel[6:6 + self.num_leg]
        leg_torque = (target_q - q) * self.kp + (0.0 - dq) * self.kd

        # ---- 填充 actuator 扭矩：只控腿，上半身不给力矩 ----
        torque = np.zeros(self.num_all_actuators)
        torque[:self.num_leg] = leg_torque
        self.d.ctrl = torque

        # ---- 上半身完全固定：直接覆写 qpos/qvel，锁死在初始位姿 ----
        self.d.qpos[7 + self.num_leg:7 + self.num_all_actuators] = self.upper_body_pose.copy()
        self.d.qvel[6 + self.num_leg:6 + self.num_all_actuators] = 0.0

        # ---- 物理步进（decimation=10） ----
        for _ in range(self.sim_decimation):
            mujoco.mj_step(self.m, self.d)

        self.lowlevel_cnt += 1

    def run(self, duration=60.0):
        """主循环"""
        sim_dt = self.m.opt.timestep
        steps = int(duration / (sim_dt * self.sim_decimation))
        for _ in range(steps):
            self.step()
            self.viewer.sync()
            time.sleep(sim_dt * self.sim_decimation)
        self.viewer.close()


if __name__ == "__main__":
    xml_path = "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/temp/biped_temp_1_0.xml"
    policy_path = "/home/cyborg/Desktop/projects/sim2real_encos/src/deploy_real/policies/c10000.pt"

    sim = EncosSim2Sim(xml_path, policy_path, cmd_vx=0.3, cmd_vy=0.0, cmd_vyaw=0.0)
    sim.run(duration=60.0)
