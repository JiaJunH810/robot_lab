"""Locomotion policy sim2sim: velocity-command biped walking in MuJoCo.

Matches Isaac Lab RobotLab-Isaac-Velocity-Flat-Cyborg-HP-v0 exactly.
"""

import math
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
    def __init__(self, xml_path, policy_path, history_length=15):
        self.history_length = history_length

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_step(self.m, self.d)

        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -20.0

        onnx_model = onnx.load(policy_path)
        self.policy = onnxruntime.InferenceSession(policy_path)
        self._load_metadata(onnx_model)

        self.hist_cmd  = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_ang  = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_grav = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_pos  = np.zeros((history_length, self.num_action), dtype=np.float32)
        self.hist_vel  = np.zeros((history_length, self.num_action), dtype=np.float32)
        self.hist_act  = np.zeros((history_length, self.num_action), dtype=np.float32)

        self.action_buffer = np.zeros(self.num_action, dtype=np.float32)
        self.vel_cmd = np.array([0.3, 0.0, 0.0], dtype=np.float32)
        self.cmd_resample_timer = 0
        self.cmd_resample_interval = int(10.0 / 0.02)

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

        self.xml_to_lab = [self.xml_order.index(j) for j in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(j) for j in self.xml_order]

    def run(self, sim_duration=120.0):
        sim_dt = 0.001
        sim_decimation = 20
        total_steps = int(sim_duration / sim_dt)

        self.d.qpos[-self.num_action:] = self.lab_default_joint_pos[self.lab_to_xml]
        self.d.qvel[-self.num_action:] = 0.0
        self.d.qpos[:3] = [0.0, 0.0, 0.94]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for _ in range(10):
            qj = self.d.qpos.astype(np.float32)[-self.num_action:]
            qvj = self.d.qvel.astype(np.float32)[-self.num_action:]
            tgt = self.lab_default_joint_pos[self.lab_to_xml]
            trq = pd_control(tgt, qj, self.lab_joint_stiffness[self.lab_to_xml],
                             np.zeros(self.num_action), qvj, self.lab_joint_damping[self.lab_to_xml])
            self.d.ctrl = trq
            mujoco.mj_step(self.m, self.d)

        pd_target = self.lab_default_joint_pos[self.lab_to_xml].copy()
        self.ctrl_step = 0

        for i in tqdm(range(total_steps), desc="Running..."):
            if i % sim_decimation == 0:
                self.ctrl_step += 1
                self.cmd_resample_timer += 1
                if self.cmd_resample_timer >= self.cmd_resample_interval:
                    self.vel_cmd = np.array([
                        np.random.uniform(-0.0, 0.0),
                        np.random.uniform(-0.0, 0.0),
                        np.random.uniform(-1.0, 1.0),
                    ], dtype=np.float32)
                    if np.linalg.norm(self.vel_cmd[:2]) < 0.2:
                        self.vel_cmd[:2] = 0.0
                    self.cmd_resample_timer = 0

                xml_joint_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
                xml_joint_vel = self.d.qvel.astype(np.float32)[-self.num_action:]
                root_quat = self.d.qpos.astype(np.float32)[3:7].copy()

                ang_vel_world = self.d.qvel.astype(np.float32)[3:6]
                R_wb = quat_to_rot_matrix(root_quat)
                base_ang_vel_body = (R_wb.T @ ang_vel_world).astype(np.float32)

                proj_grav = projected_gravity(root_quat)
                joint_pos_rel = xml_joint_pos[self.xml_to_lab] - self.lab_default_joint_pos
                joint_vel_rel = xml_joint_vel[self.xml_to_lab]

                self._push_history(self.hist_cmd,  self.vel_cmd)
                self._push_history(self.hist_ang,  base_ang_vel_body * 0.25)
                self._push_history(self.hist_grav, proj_grav)
                self._push_history(self.hist_pos,  joint_pos_rel)
                self._push_history(self.hist_vel,  joint_vel_rel * 0.05)
                self._push_history(self.hist_act,  self.action_buffer)

                obs = np.concatenate([
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

                if self.ctrl_step % 50 == 0:
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
                    print(f"[{self.ctrl_step:5d} t={t:5.1f}s] BaseZ={bz:.3f} "
                          f"Pitch={math.degrees(pitch):5.1f} Roll={math.degrees(roll):5.1f} | "
                          f"ActMean={np.mean(lab_actions):.3f} MaxErr={np.max(np.abs(err)):.3f} "
                          f"MaxTrq={np.max(np.abs(trq)):.0f} | "
                          f"Cmd:vx={self.vel_cmd[0]:.2f} vy={self.vel_cmd[1]:.2f} wz={self.vel_cmd[2]:.2f}")
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

        self.viewer.close()


if __name__ == "__main__":
    xml_path = "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/temp/biped_temp_1_0_fixed.xml"
    policy_path = "/home/cyborg/Desktop/projects/robot_lab/logs/rsl_rl/cyborg_hp_flat/2026-07-08_11-48-30/exported/policy.onnx"

    sim = LocoSim2Sim(xml_path, policy_path, history_length=15)
    sim.run(sim_duration=120.0)
