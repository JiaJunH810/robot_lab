import mujoco
import mujoco_viewer
import numpy as np
from tqdm import tqdm
import torch

default_q = np.array([
        0.0, 0.0, 0.4, 0.7, 0.3, 0.0,
        0.0, 0.0, -0.4, -0.7, -0.3, 0.0,
    ], dtype=np.float32)
kp = np.array([
        250, 120, 300, 300, 80, 80,
        250, 120, 300, 300, 80, 80,
    ], dtype=np.float32)
kd = np.array([
        10, 10, 10, 10, 3, 3,
        10, 10, 10, 10, 3, 3,
    ], dtype=np.float32)
action_scale = np.array([
        0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
        0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
    ], dtype=np.float32)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd

def quat_to_euler(q):
    w, x, y, z = q
    return np.array([
        np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
        np.arcsin(np.clip(2 * (w*y - z*x), -1.0, 1.0)),
        np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z)),
    ], dtype=np.float32)

class Locomotion:
    def __init__(self, xml_path, policy_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetData(self.m, self.d)
        mujoco.mj_step(self.m, self.d)
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -20.0

        self.policy = torch.jit.load(policy_path, map_location="cpu")
        self.policy.eval()

        
        self.history_obs = np.zeros((15, 47), dtype=np.float32)
        self.last_action = np.zeros(12, dtype=np.float32)
        self.policy_step = 0

    def get_policy_obs(self, command):
        self.policy_step += 1
        phase = 2.0 * np.pi * self.policy_step * 0.01 / 0.8
        phase_obs = np.array([np.sin(phase), np.cos(phase)], dtype=np.float32,)
        if np.sum(np.abs(command)) <= 0.1:
            phase_obs[:] = 0.0
        q = self.d.qpos[-12:].astype(np.float32)
        dq = self.d.qvel[-12:].astype(np.float32)
        quat = self.d.sensor("orientation").data.astype(np.float32)
        omega = self.d.sensor("angular-velocity").data.astype(np.float32)

        obs = np.concatenate([
            phase_obs,
            command * np.array([2.0, 2.0, 1.0], dtype=np.float32),
            q - default_q,
            dq * 0.05,
            self.last_action,
            omega,
            quat_to_euler(quat),
        ]).astype(np.float32)

        obs = np.clip(obs, -50.0, 50.0)
        self.history_obs[:-1] = self.history_obs[1:]
        self.history_obs[-1] = obs

        return self.history_obs.reshape(1, 705)

    def run(self):
        sim_duration = 120.0
        decimation = 10
        total_steps = int(sim_duration / self.m.opt.timestep)

        self.d.qpos[:3] = [0.0, 0.0, 0.895]
        self.d.qpos[3:7] = [0.9990482, 0.0, -0.0436194, 0.0]
        # self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.d.qpos[-12:] = default_q
        self.d.qvel[:] = 0.0
        mujoco.mj_forward(self.m, self.d)

        target_q = default_q.copy()
        target_dq = np.zeros(12, dtype=np.float32)

        command = np.array([0.3, 0.0, 0.0], dtype=np.float32)

        for i in tqdm(range(total_steps)):
            if i % decimation == 0:
                obs = self.get_policy_obs(command)
                with torch.inference_mode():
                    action = self.policy(torch.from_numpy(obs)).cpu().numpy().squeeze(0)
                action = np.clip(action, -5.0, 5.0)
                target_q = default_q + action * action_scale
                self.last_action = action.copy()
                self.viewer.cam.lookat = self.d.qpos[:3]
                self.viewer.render()
            q = self.d.qpos[-12:].astype(np.float32)
            dq = self.d.qvel[-12:].astype(np.float32)
            torque = pd_control(target_q, q, kp, target_dq, dq, kd,)

            knee_l = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_knee_l_pitch")]]
            knee_r = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_knee_r_pitch")]]
            hip_pitch_l = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_hip_l_pitch")]]
            hip_pitch_r = self.d.qpos[self.m.jnt_qposadr[mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "J_hip_r_pitch")]]
            print(f"BaseAngVel={self.d.qvel[3:5]} "
                  f"KneeAngel L={knee_l:.3f} R={knee_r:.3f} "
                  f"HipPitchAngel L={hip_pitch_l:.3f} R={hip_pitch_r:.3f}")
            
            self.d.ctrl[:] = torque
            mujoco.mj_step(self.m, self.d)

        self.viewer.close()

if __name__ == "__main__":
    xml_path = "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/temp/biped_temp_1_0_fixed.xml"
    policy_path = "/home/cyborg/Desktop/projects/sim2real_robotlab/src/deploy_real/policy/c12100.pt"

    sim = Locomotion(xml_path, policy_path)
    sim.run()
    
