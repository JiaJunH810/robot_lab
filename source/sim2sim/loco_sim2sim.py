"""Locomotion policy sim2sim: velocity-command biped walking in MuJoCo.

Matches Isaac Lab RobotLab-Isaac-Velocity-Flat-Cyborg-HP-v0 exactly.
Supports joystick (e.g. GameSir) for manual velocity control.
If no joystick detected, falls back to uniform random commands.
"""

import math
import os
import struct

import numpy as np
import onnx
import onnxruntime
import mujoco
import mujoco_viewer
from tqdm import tqdm


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_to_rot_matrix(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx*qx - 2*qz*qz,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx*qx - 2*qy*qy],
    ])


def projected_gravity(quat_wxyz):
    R = quat_to_rot_matrix(quat_wxyz)
    return R.T @ np.array([0.0, 0.0, -1.0])


class LocoSim2Sim:
    def __init__(self, xml_path, policy_path, history_length=15,
                 cmd_max=(0.5, 0.5, 0.5), cycle_time=0.9,
                 command_threshold=0.1, recovery_tilt_threshold=0.17,
                 max_acceleration=(0.2, 0.2, 0.2)):
        self.history_length = history_length
        self.cmd_max = cmd_max  # (vx_max, vy_max, wz_max)
        self.cycle_time = cycle_time
        self.command_threshold = command_threshold
        self.recovery_tilt_threshold = recovery_tilt_threshold
        self.max_acceleration = np.asarray(max_acceleration, dtype=np.float32)

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_step(self.m, self.d)

        # 脚踝连杆 body ID，用于打印高度
        self.ankle_l_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_roll_link")
        self.ankle_r_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_roll_link")

        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -20.0

        onnx_model = onnx.load(policy_path)
        self.policy = onnxruntime.InferenceSession(policy_path)
        self._load_metadata(onnx_model)

        self.hist_phase = np.zeros((history_length, 2), dtype=np.float32)
        self.hist_cmd  = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_ang  = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_grav = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_pos  = np.zeros((history_length, self.num_action), dtype=np.float32)
        self.hist_vel  = np.zeros((history_length, self.num_action), dtype=np.float32)
        self.hist_act  = np.zeros((history_length, self.num_action), dtype=np.float32)

        # Actor observation per frame:
        # phase(2) + ang_vel(3) + gravity(3) + command(3)
        # + joint_pos(12) + joint_vel(12) + last_action(12) = 47.
        self.expected_obs_dim = history_length * (11 + 3 * self.num_action)
        policy_obs_dim = self.policy.get_inputs()[0].shape[-1]
        if isinstance(policy_obs_dim, int) and policy_obs_dim != self.expected_obs_dim:
            raise ValueError(
                f"Policy expects {policy_obs_dim} observations, but sim2sim builds {self.expected_obs_dim}. "
                "Check phase observation, history length, and joint count."
            )

        self.action_buffer = np.zeros(self.num_action, dtype=np.float32)
        self.vel_cmd = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.target_vel_cmd = np.zeros(3, dtype=np.float32)
        self.cmd_resample_timer = 0

        # ---- Joystick ----
        self._js_fd = self._open_js_device()
        self._js_axes = {}
        self._js_buttons = set()
        self._js_mapped = False
        # GameSir default axis mapping (adjust if needed)
        self._js_axis_vx = 1   # left stick Y
        self._js_axis_vy = 0   # left stick X
        self._js_axis_wz = 3   # right stick X (try 2 or 4 if no response)

    @staticmethod
    def _push_history(buf, new_val):
        buf[:-1] = buf[1:]
        buf[-1] = new_val

    def _load_metadata(self, model):
        self.xml_order = []
        for i in range(self.m.nu):
            self.xml_order.append(
                mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i))
        self.num_action = len(self.xml_order)
        print(" -------------------- xml order -------------------- ")
        print(self.xml_order)

        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.lab_order = [x for x in prop.value.split(",")]
            elif prop.key == "default_joint_pos":
                self.lab_default_joint_pos = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "joint_stiffness":
                self.lab_joint_stiffness = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "joint_damping":
                self.lab_joint_damping = np.array([float(x) for x in prop.value.split(",")])
            elif prop.key == "action_scale":
                self.lab_action_scale = np.array([float(x) for x in prop.value.split(",")])
        print(" -------------------- lab order -------------------- ")
        print(self.lab_order)
        print(self.lab_default_joint_pos)
        print(self.lab_joint_stiffness)
        print(self.lab_joint_damping)
        print(self.lab_action_scale)

        self.xml_to_lab = [self.xml_order.index(j) for j in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(j) for j in self.xml_order]

    # =========================================================================
    # Joystick
    # =========================================================================
    def _open_js_device(self):
        for dev in ["/dev/input/js0", "/dev/input/js1", "/dev/input/js2"]:
            try:
                fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
                print(f"[Joystick] Opened {dev}")
                return fd
            except OSError:
                continue
        print("[Joystick] Not found, using uniform random commands")
        return None

    def _read_js_state(self):
        if self._js_fd is None:
            return {}, set()
        while True:
            try:
                data = os.read(self._js_fd, 8)
                if len(data) < 8:
                    break
                _, value, etype, num = struct.unpack('<IhBB', data)
                if etype == 0x02:
                    self._js_axes[num] = value / 32767.0
                elif etype == 0x01:
                    if value:
                        self._js_buttons.add(num)
                    else:
                        self._js_buttons.discard(num)
            except BlockingIOError:
                break
            except OSError:
                break
        return self._js_axes, self._js_buttons

    def _get_joystick_cmd(self):
        axes, buttons = self._read_js_state()
        if buttons and not self._js_mapped:
            print(f"[Joystick] axes: {dict(sorted(axes.items()))}  buttons: {buttons}")
            self._js_mapped = True

        def dead(val, dz=0.1):
            return 0.0 if abs(val) < dz else val

        vx = -dead(axes.get(self._js_axis_vx, 0.0)) * self.cmd_max[0]
        vy = -dead(axes.get(self._js_axis_vy, 0.0)) * self.cmd_max[1]
        wz = -dead(axes.get(self._js_axis_wz, 0.0)) * self.cmd_max[2]
        # A button = emergency stop
        if 0 in buttons:
            vx = vy = wz = 0.0
        return np.array([vx, vy, wz], dtype=np.float32)

    # =========================================================================
    def run(self, sim_duration=120.0):
        sim_dt = 0.001
        sim_decimation = 10
        total_steps = int(sim_duration / sim_dt)
        policy_dt = sim_dt * sim_decimation
        self.cmd_resample_interval = int(5.0 / policy_dt)

        self.d.qpos[-self.num_action:] = self.lab_default_joint_pos[self.lab_to_xml]
        self.d.qvel[-self.num_action:] = 0.0
        self.d.qpos[:3] = [0.0, 0.0, 0.895]
        self.d.qpos[3:7] = [0.9990482, 0.0, -0.0436194, 0.0]
        # self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for _ in range(10):
            qj = self.d.qpos.astype(np.float32)[-self.num_action:]
            qvj = self.d.qvel.astype(np.float32)[-self.num_action:]
            tgt = self.lab_default_joint_pos[self.lab_to_xml]
            trq = pd_control(
                tgt, qj, self.lab_joint_stiffness[self.lab_to_xml],
                np.zeros(self.num_action), qvj, self.lab_joint_damping[self.lab_to_xml],
            )
            self.d.ctrl = trq
            mujoco.mj_step(self.m, self.d)

        pd_target = self.lab_default_joint_pos[self.lab_to_xml].copy()
        self.ctrl_step = 0

        for i in tqdm(range(total_steps), desc="Running..."):
            if i % sim_decimation == 0:
                phase_step = self.ctrl_step
                self.ctrl_step += 1

                # Match mdp.phase(): episode_length_buf * step_dt / cycle_time.
                # phase_step starts at zero, matching the initial Isaac Lab observation.
                policy_dt = sim_dt * sim_decimation
                phase = (phase_step * policy_dt / self.cycle_time) % 1.0
                phase_obs = np.array([
                    math.sin(2.0 * math.pi * phase),
                    math.cos(2.0 * math.pi * phase),
                ], dtype=np.float32)

                # ---- Velocity command: joystick or uniform random ----
                if self._js_fd is not None:
                    self.target_vel_cmd = self._get_joystick_cmd()
                    if np.linalg.norm(self.target_vel_cmd[:2]) < 0.1:
                        self.target_vel_cmd[:2] = 0.0
                else:
                    self.cmd_resample_timer += 1
                    if self.cmd_resample_timer >= self.cmd_resample_interval:
                        self.target_vel_cmd = np.array([
                            np.random.uniform(-self.cmd_max[0], self.cmd_max[0]),
                            np.random.uniform(-self.cmd_max[1], self.cmd_max[1]),
                            np.random.uniform(-self.cmd_max[2], self.cmd_max[2]),
                        ], dtype=np.float32)
                        if np.linalg.norm(self.target_vel_cmd[:2]) < 0.1:
                            self.target_vel_cmd[:2] = 0.0
                        self.cmd_resample_timer = 0

                max_delta = self.max_acceleration * policy_dt
                self.vel_cmd += np.clip(
                    self.target_vel_cmd - self.vel_cmd,
                    -max_delta,
                    max_delta,
                )

                xml_joint_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
                xml_joint_vel = self.d.qvel.astype(np.float32)[-self.num_action:]
                root_quat = self.d.sensor("orientation").data.astype(np.float32)

                # MuJoCo qvel[3:6] is already body-frame angular velocity
                base_ang_vel_body = self.d.sensor("angular-velocity").data.astype(np.float32)

                proj_grav = projected_gravity(root_quat)
                # 相位门控仅 moving（对齐训练：观测 phase 无 recovery_tilt_threshold）
                moving = (
                    np.linalg.norm(self.vel_cmd[:2]) > self.command_threshold
                    or abs(self.vel_cmd[2]) > self.command_threshold
                )
                if not moving:
                    phase_obs.fill(0.0)

                joint_pos_rel = xml_joint_pos[self.xml_to_lab] - self.lab_default_joint_pos
                joint_vel_rel = xml_joint_vel[self.xml_to_lab]

                self._push_history(self.hist_phase, phase_obs)
                self._push_history(self.hist_cmd,  self.vel_cmd)
                self._push_history(self.hist_ang,  base_ang_vel_body * 0.25)
                self._push_history(self.hist_grav, proj_grav)
                self._push_history(self.hist_pos,  joint_pos_rel)
                self._push_history(self.hist_vel,  joint_vel_rel * 0.05)
                self._push_history(self.hist_act,  self.action_buffer)

                obs = np.concatenate([
                    self.hist_phase.flatten(),
                    self.hist_ang.flatten(),
                    self.hist_grav.flatten(),
                    self.hist_cmd.flatten(),
                    self.hist_pos.flatten(),
                    self.hist_vel.flatten(),
                    self.hist_act.flatten(),
                ]).astype(np.float32).reshape(1, -1)

                lab_actions = self.policy.run(["actions"], {"obs": obs})[0].squeeze()
                self.action_buffer = lab_actions.copy()

                scale_actions = lab_actions * self.lab_action_scale
                pd_target = (scale_actions[self.lab_to_xml]
                             + self.lab_default_joint_pos[self.lab_to_xml])

                self.viewer.cam.lookat = self.d.qpos.astype(np.float32)[:3]
                self.viewer.render()

                if self.ctrl_step % 10 == 0:
                    t = i * sim_dt
                    bz = self.d.qpos[2]
                    qw, qx, qy, qz = self.d.qpos[3:7]
                    pitch = math.asin(max(-1, min(1, 2*(qw*qy - qz*qx))))
                    roll  = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
                    xml_joint_pos_diag = self.d.qpos.astype(np.float32)[-self.num_action:]
                    xml_joint_vel_diag = self.d.qvel.astype(np.float32)[-self.num_action:]
                    err = pd_target - xml_joint_pos_diag
                    trq = (err * self.lab_joint_stiffness[self.lab_to_xml]
                           + (0 - xml_joint_vel_diag) * self.lab_joint_damping[self.lab_to_xml])
                    ankle_l_z = self.d.xpos[self.ankle_l_id][2]
                    ankle_r_z = self.d.xpos[self.ankle_r_id][2]
                    knee_pitch_l = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_knee_l_pitch")]]
                    knee_pitch_r = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_knee_r_pitch")]]
                    hip_pitch_l = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_hip_l_pitch")]]
                    hip_pitch_r = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_hip_r_pitch")]]
                    print(f"[{self.ctrl_step:5d} t={t:5.1f}s] BaseZ={bz:.3f} "
                          f"Pitch={math.degrees(pitch):5.1f} Roll={math.degrees(roll):5.1f} | "
                          f"AnkleZ L={ankle_l_z:.3f} R={ankle_r_z:.3f} | "
                          f"ActMean={np.mean(lab_actions):.3f} MaxErr={np.max(np.abs(err)):.3f} "
                          f"MaxTrq={np.max(np.abs(trq)):.0f} | "
                          f"Cmd:vx={self.vel_cmd[0]:.2f} vy={self.vel_cmd[1]:.2f} wz={self.vel_cmd[2]:.2f}" 
                          f"BaseHeight={self.d.qpos[2]} " 
                          f"BaseAngVel={self.d.qvel[3:5]} "
                          f"KneePitchAngel L={knee_pitch_l:.3f} R={knee_pitch_r:.3f} "
                          f"HipPitchAngel L={hip_pitch_l:.3f} R={hip_pitch_r:.3f}")
                    if bz < 0.5:
                        print(f"  *** FALLEN ***")

            xml_joint_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
            xml_joint_vel = self.d.qvel.astype(np.float32)[-self.num_action:]
            torque = pd_control(
                pd_target, xml_joint_pos,
                self.lab_joint_stiffness[self.lab_to_xml],
                np.zeros(self.num_action),
                xml_joint_vel,
                self.lab_joint_damping[self.lab_to_xml],
            )
            self.d.ctrl = torque
            mujoco.mj_step(self.m, self.d)

        if self._js_fd is not None:
            os.close(self._js_fd)
        self.viewer.close()


if __name__ == "__main__":
    xml_path = "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/temp/biped_temp_1_0_fixed.xml"
    policy_path = "/home/cyborg/Desktop/projects/robot_lab/logs/rsl_rl/cyborg_hp_flat/2026-09-02_11-19-54/exported/policy.onnx"

    sim = LocoSim2Sim(
        xml_path,
        policy_path,
        history_length=15,
        cmd_max=(0.5, 0.3, 0.5),
    )
    sim.run(sim_duration=120.0)
