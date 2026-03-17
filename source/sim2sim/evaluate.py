import time
import mujoco, mujoco_viewer
import numpy as np
import torch
import onnx
import yaml
from tqdm import tqdm
import onnxruntime
import joblib
import os
import glob

@torch.jit.script
def quat_unique(q: torch.Tensor) -> torch.Tensor:
    """Convert a unit quaternion to a standard form where the real part is non-negative.

    Quaternion representations have a singularity since ``q`` and ``-q`` represent the same
    rotation. This function ensures the real part of the quaternion is non-negative.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        Standardized quaternions. Shape is (..., 4).
    """
    return torch.where(q[..., 0:1] < 0, -q, q)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply an inverse quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Computes the conjugate of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (..., 4).

    Returns:
        The conjugate quaternion in (w, x, y, z). Shape is (..., 4).
    """
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Computes the inverse of a quaternion.

    Args:
        q: The quaternion orientation in (w, x, y, z). Shape is (N, 4).
        eps: A small value to avoid division by zero. Defaults to 1e-9.

    Returns:
        The inverse quaternion in (w, x, y, z). Shape is (N, 4).
    """
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions together.

    Args:
        q1: The first quaternion in (w, x, y, z). Shape is (..., 4).
        q2: The second quaternion in (w, x, y, z). Shape is (..., 4).

    Returns:
        The product of the two quaternions in (w, x, y, z). Shape is (..., 4).

    Raises:
        ValueError: Input shapes of ``q1`` and ``q2`` are not matching.
    """
    # check input is correct
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    # reshape to (N, 4) for multiplication
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    # extract components from quaternions
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    # perform multiplication
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    return torch.stack([w, x, y, z], dim=-1).view(shape)

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply a quaternion rotation to a vector.

    Args:
        quat: The quaternion in (w, x, y, z). Shape is (..., 4).
        vec: The vector in (x, y, z). Shape is (..., 3).

    Returns:
        The rotated vector in (x, y, z). Shape is (..., 3).
    """
    # store shape
    shape = vec.shape
    # reshape to (N, 3) for multiplication
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
    # extract components from quaternions
    xyz = quat[:, 1:]
    t = xyz.cross(vec, dim=-1) * 2
    return (vec + quat[:, 0:1] * t + xyz.cross(t, dim=-1)).view(shape)

def subtract_frame_transforms(
    t01: torch.Tensor, q01: torch.Tensor, t02: torch.Tensor, q02: torch.Tensor
):
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = quat_mul(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def get_yaw_quat(quat: torch.Tensor) -> torch.Tensor:
        """从四元数中提取仅包含 Yaw 旋转的部分 (w, x, y, z)"""
        # 转换到欧拉角或直接提取
        # 这里使用简便方法：将 x, y 设为 0 并重新归一化
        yaw_quat = quat.clone()
        yaw_quat[..., 1:3] = 0.0
        return yaw_quat / torch.norm(yaw_quat, dim=-1, keepdim=True)

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class RobotLabSim2Sim:
    def __init__(self, xml_path, motion_folder, policy_path, code_path):

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_step(self.m, self.d)
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 5.0
        
        model = onnx.load(policy_path)
        self.load(model)

        self.policy = onnxruntime.InferenceSession(policy_path)

        self.evaluate(motion_folder, code_path)
    
    def evaluate(self, motion_folder, code_path):
        self.avg_sr = []
        self.avg_mpjpe = []
        self.avg_mpjve = []
        if os.path.isfile(motion_folder):
            motion_files = [motion_folder]
        else:
            motion_files = glob.glob(f"{motion_folder}/**/*.npz", recursive=True)
        for motion_file in tqdm(motion_files):
            motion = np.load(motion_file)
            self.motion_joint_pos = motion["joint_pos"]
            self.motion_joint_vel = motion["joint_vel"]
            self.motion_body_pos_w = motion["body_pos_w"]
            self.motion_body_quat_w = motion["body_quat_w"]
            self.motion_body_lin_vel_w = motion["body_lin_vel_w"]
            code_data = np.load(code_path, allow_pickle=True)
            self.codebook = code_data['codebook']
            self.code_idx = code_data[motion_file]

            results = self.run()
            self.avg_sr.append(float(results["success"]))

            if results["success"]:
                self.avg_mpjpe.append(results["mpjpe_mm"])
                self.avg_mpjve.append(results["mpjve_mm_s"])
                print(f"Success! MPJPE: {results['mpjpe_mm']:.2f} mm, MPJVE: {results['mpjve_mm_s']:.2f} mm/s", {motion_file})
            else:
                print(f"Failed: {motion_file}")
        
        # 打印汇总报表
        mean_sr = np.mean(self.avg_sr) * 100.0
        mean_mpjpe = np.mean(self.avg_mpjpe) if self.avg_mpjpe else 0.0
        mean_mpjve = np.mean(self.avg_mpjve) if self.avg_mpjve else 0.0
        
        print("\n" + "="*50)
        print(f"FINAL RESULTS")
        print("-"*50)
        print(f"Success Rate:      {mean_sr:.2f} %")
        print(f"Mean MPJPE (Local): {mean_mpjpe:.2f} mm")
        print(f"Mean MPJVE (Local): {mean_mpjve:.2f} mm/s")
        print("="*50 + "\n")
    
    def load(self, model):
        print ("========================== xml parameters ==========================")
        self.xml_order = []
        self.xml_body_names = []
        for i in range(self.m.nu):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            self.xml_order.append(name)
        for i in range(self.m.nbody):
            name = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, i)
            if name is None:
                name = 'world'
            self.xml_body_names.append(name)
        self.num_action = len(self.xml_order)
        print(f"xml_order: {self.xml_order}")
        print(f"num_action: {self.num_action}")
        print(f"body_names: {self.xml_body_names}")

        for prop in model.metadata_props:
            if prop.key == "joint_names":
                self.lab_order = [x for x in prop.value.split(',')]
            if prop.key == "default_joint_pos":
                self.lab_default_joint_pos = np.array([float(x) for x in prop.value.split(',')])
            if prop.key == "joint_stiffness":
                self.lab_joint_stiffness = np.array([float(x) for x in prop.value.split(',')])
            if prop.key == "joint_damping":
                self.lab_joint_damping = np.array([float(x) for x in prop.value.split(',')])
            if prop.key == "action_scale":
                self.lab_action_scale = np.array([float(x) for x in prop.value.split(',')])
            if prop.key == "body_names":
                self.lab_body_names = [x for x in prop.value.split(',')]
        print ("========================== lab parameters ==========================")
        print(f"lab_order: {self.lab_order}")
        print(f"default_joint_pos: {self.lab_default_joint_pos}")
        print(f"joint_stiffness: {self.lab_joint_stiffness}")
        print(f"joint_damping: {self.lab_joint_damping}")
        print(f"action_scale: {self.lab_action_scale}")
        print(f"body_names: {self.lab_body_names}")

        self.xml_to_lab = [self.xml_order.index(joint) for joint in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(joint) for joint in self.xml_order]
    
    def extract_data(self, anchor_name):
        dof_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
        dof_vel = self.d.qvel.astype(np.float32)[-self.num_action:]
        root_pos = self.d.qpos.astype(np.float32)[0:3]
        root_quat = self.d.qpos.astype(np.float32)[3:7]
        # root_pos = self.d.xpos[self.xml_body_names.index(anchor_name)]
        # root_quat = self.d.xquat[self.xml_body_names.index(anchor_name)]
        # ang_vel = self.d.sensor('imu-pelvis-angular-velocity').data.astype(np.float32)
        ang_vel = self.d.qvel.astype(np.float32)[3:6]
        return (dof_pos, dof_vel, root_pos, root_quat, ang_vel)

    def calc_motion_anchor_ori_b(self, robot_anchor_pos, robot_anchor_quat, motion_anchor_pos, motion_anchor_quat):
        _, ori = subtract_frame_transforms(
            robot_anchor_pos, robot_anchor_quat,
            motion_anchor_pos, motion_anchor_quat
        )
        mat = matrix_from_quat(ori)

        return mat[:, :2].flatten()
    
    def projected_gravity(self, anchor_quat):
        gravity = torch.tensor([0.0, 0.0, -1.0], dtype=anchor_quat.dtype)
        return quat_apply_inverse(anchor_quat, gravity)

    
    def future_motion(self, timestep, horizon, anchor_name, extra_body_name):
        future_idx = np.arange(timestep, timestep + horizon * 5, 5)
        future_idx = np.minimum(future_idx, self.motion_joint_pos.shape[0] - 1)
        anchor_idx = self.lab_body_names.index(anchor_name)
        future_anchor_pos = torch.tensor(self.motion_body_pos_w[future_idx, anchor_idx])
        future_anchor_quat = torch.tensor(self.motion_body_quat_w[future_idx, anchor_idx])
        future_anchor_lin_vel = torch.tensor(self.motion_body_lin_vel_w[future_idx, anchor_idx])
        current_anchor_pos = torch.tensor(self.motion_body_pos_w[timestep, anchor_idx]).unsqueeze(0).repeat(horizon, 1)
        current_anchor_quat = torch.tensor(self.motion_body_quat_w[timestep, anchor_idx]).unsqueeze(0).repeat(horizon, 1)
        pos, ori = subtract_frame_transforms(
            current_anchor_pos,
            current_anchor_quat,
            future_anchor_pos,
            future_anchor_quat,
        )
        ori = quat_unique(ori)
        vel = quat_apply_inverse(current_anchor_quat, future_anchor_lin_vel)
        height = future_anchor_pos[..., -1:]

        return torch.cat([pos, ori, vel, height], dim=-1).flatten().numpy()
    
    def vqvae_code(self, timestep):
        idx = self.code_idx[timestep]
        code_vector = self.codebook[idx]
        return code_vector.reshape(1, -1)

    
    def run(self):
        sr = True
        count = 0
        mpjpe_list = []
        mpjve_list = []

        print("帧数：", self.motion_joint_pos.shape[0])
        sim_duration = self.motion_joint_pos.shape[0]
        sim_dt = 0.001
        sim_decimation = 20
        timestep = 0
        anchor_name = "pelvis"
        anchor_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, anchor_name)
        extra_body_name = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_rubber_hand", "right_wrist_roll_rubber_hand",]
        action_buffer = np.zeros((self.num_action, ), dtype=np.float32)
        print(f"帧数: {self.motion_joint_pos.shape[0]}")

        # 初始化状态
        # self.d.qpos[-self.num_action:] = self.lab_default_joint_pos[self.lab_to_xml]
        self.d.qpos[-self.num_action:] = self.motion_joint_pos[0, self.lab_to_xml]
        self.d.qvel[-self.num_action:] = self.motion_joint_vel[0, self.lab_to_xml]
        self.d.qpos[:3] = self.motion_body_pos_w[0, self.lab_body_names.index(anchor_name), :]
        self.d.qpos[3:7] = self.motion_body_quat_w[0, self.lab_body_names.index(anchor_name), :]
        
        for i in tqdm(range(int(sim_duration * sim_decimation)), desc="Running simulation..."):
            xml_joint_pos, xml_joint_vel, root_pos, root_quat, ang_vel = self.extract_data(anchor_name)
           
            curr_root_quat_torch = torch.tensor(root_quat)
            ref_root_quat_torch = torch.tensor(self.motion_body_quat_w[timestep, self.lab_body_names.index(anchor_name), :])
            curr_yaw_quat = get_yaw_quat(curr_root_quat_torch)
            ref_yaw_quat = get_yaw_quat(ref_root_quat_torch)
            delta_yaw_quat = quat_mul(curr_yaw_quat.unsqueeze(0), quat_inv(ref_yaw_quat.unsqueeze(0))).squeeze(0)

            if abs(self.motion_body_pos_w[timestep, self.lab_body_names.index(anchor_name), -1] - root_pos[-1]) > 0.2:
                count += 1
                if count > sim_decimation * 5:
                    sr = False
                    break
            current_frame_pe = []
            current_frame_ve = []
            curr_body_pos = self.d.xpos.copy()
            curr_body_vel = self.d.cvel[:, 3:6].copy()

            root_pos_sim = root_pos
            root_vel_sim = self.d.cvel[anchor_id, 3:6]

            ref_anchor_idx = self.lab_body_names.index(anchor_name)
            ref_root_pos = self.motion_body_pos_w[timestep, ref_anchor_idx, :]
            ref_root_vel = self.motion_body_lin_vel_w[timestep, ref_anchor_idx, :]

            for b_idx_lab, b_name in enumerate(self.lab_body_names):
                m_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, b_name)
                if m_body_id == -1: continue
                
                # --- 位置误差 (PE) 对齐计算 ---
                # A. 参考动作中肢体相对于根部的偏移 (World frame)
                ref_rel_pos = self.motion_body_pos_w[timestep, b_idx_lab, :] - ref_root_pos
                
                # B. 将该偏移旋转 delta_yaw，使其与机器人朝向一致 (对应 IsaacLab 的 quat_apply(delta_ori_w, ...))
                ref_rel_pos_aligned = quat_apply(delta_yaw_quat, torch.tensor(ref_rel_pos)).numpy()
                
                # C. 机器人实际的相对偏移
                sim_rel_pos = curr_body_pos[m_body_id] - root_pos_sim
                
                # D. 计算误差：两者之差的范数
                pe = np.linalg.norm(sim_rel_pos - ref_rel_pos_aligned)
                current_frame_pe.append(pe)
                
                # --- 速度误差 (VE) 对齐计算 ---
                # A. 参考动作中肢体相对于根部的相对速度
                ref_rel_vel = self.motion_body_lin_vel_w[timestep, b_idx_lab, :] - ref_root_vel
                
                # B. 将参考速度偏移也进行 Yaw 旋转对齐
                ref_rel_vel_aligned = quat_apply(delta_yaw_quat, torch.tensor(ref_rel_vel)).numpy()
                
                # C. 机器人实际的相对速度
                sim_rel_vel = curr_body_vel[m_body_id] - root_vel_sim
                
                # D. 计算误差
                ve = np.linalg.norm(sim_rel_vel - ref_rel_vel_aligned)
                current_frame_ve.append(ve)
            
            mpjpe_list.append(np.mean(current_frame_pe) * 1000.0) # mm
            mpjve_list.append(np.mean(current_frame_ve) * 1000.0) # mm/s (因为 cvel 本身是 m/s)

            target_h = self.motion_body_pos_w[timestep, self.lab_body_names.index(anchor_name), -1]
            if abs(target_h - root_pos[-1]) > 0.25:
                count += 1
                if count > sim_decimation * 10: # 持续约 0.1s 的显著误差判定为失败
                    sr = False
                    break
            else:
                count = 0

            if i % sim_decimation == 0:
                # timestep = min(timestep, self.motion_joint_pos.shape[0] - 1)
                command = np.concatenate((self.motion_joint_pos[timestep, :], self.motion_joint_vel[timestep, :]), axis=-1)
                motion_anchor_ori_b = self.calc_motion_anchor_ori_b(
                    torch.tensor(root_pos), torch.tensor(root_quat),
                    torch.tensor(self.motion_body_pos_w[timestep, self.lab_body_names.index(anchor_name), :]),
                    torch.tensor(self.motion_body_quat_w[timestep, self.lab_body_names.index(anchor_name), :])
                    ).numpy()
                projected_gravity = self.projected_gravity(torch.tensor(root_quat))
                base_ang_vel = ang_vel
                joint_pos = xml_joint_pos[self.xml_to_lab] - self.lab_default_joint_pos
                joint_vel = xml_joint_vel[self.xml_to_lab]
                last_actions = action_buffer
                # future = self.future_motion(timestep, horizon=10, anchor_name=anchor_name, extra_body_name=extra_body_name)

                obs = np.concatenate([
                    command,
                    motion_anchor_ori_b,
                    projected_gravity,
                    base_ang_vel,
                    joint_pos,
                    joint_vel,
                    last_actions,
                    # future,
                ]).astype(np.float32).reshape(1, -1)
                code = self.vqvae_code(timestep)

                lab_actions = self.policy.run(['actions'], {'obs': obs, 'code': code})[0].squeeze()
                action_buffer = lab_actions.copy()
                scale_actions = lab_actions * self.lab_action_scale
                
                pd_target = scale_actions[self.lab_to_xml] + self.lab_default_joint_pos[self.lab_to_xml]

                self.viewer.cam.lookat = self.d.qpos.astype(np.float32)[:3]
                self.viewer.render()
                
                # timestep = (timestep + 1) % self.motion_joint_pos.shape[0]
                timestep = min(timestep + 1, self.motion_joint_pos.shape[0] - 1)
                

            torque = pd_control(pd_target, xml_joint_pos, self.lab_joint_stiffness[self.lab_to_xml], np.zeros_like(self.lab_joint_damping), xml_joint_vel, self.lab_joint_damping[self.lab_to_xml])

            self.d.ctrl = torque
            mujoco.mj_step(self.m, self.d)

        return {
            "success": sr,
            "mpjpe_mm": np.mean(mpjpe_list) if mpjpe_list else 0,
            "mpjve_mm_s": np.mean(mpjve_list) if mpjve_list else 0
        }
# ================= 主程序 =================
if __name__ == "__main__":
    # 路径配置
    xml_path = "/home/ubuntu/projects/hjj-robot_lab/source/sim2sim/assets/g1_23dof_rev_1_0.xml"
    code_path = "/home/ubuntu/projects/hjj-robot_lab/source/vq-vae/logs/2026-02-18_14-06-53/codebook_interp.npz"
    motion_folder = "/home/ubuntu/projects/hjj-robot_lab/source/motion/motions_npz/cycle/113_08_stageii_and_09_09_stageii.npz"
    policy_path = "/home/ubuntu/projects/hjj-robot_lab/logs/rsl_rl/unitree_g1_vaemimic_flat/2026-02-21_14-12-57/exported/policy.onnx"
    # policy_path = "/home/ubuntu/projects/hjj-robot_lab/logs/rsl_rl/unitree_g1_vaemimic_flat/2026-02-19_23-44-38/exported/policy.onnx"
    # policy_path = "/home/ubuntu/projects/hjj-robot_lab/logs/rsl_rl/unitree_g1_vaemimic_flat/2026-02-19_09-37-32/exported/policy.onnx"

    
    r = RobotLabSim2Sim(xml_path, motion_folder, policy_path, code_path)
    # r.run()