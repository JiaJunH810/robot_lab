import time
import mujoco, mujoco_viewer
import numpy as np
import torch
import onnx
import yaml
from tqdm import tqdm
import onnxruntime
from isaaclab.utils.math import quat_error_magnitude

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

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd


class RobotLabSim2Sim:
    def __init__(self, xml_path, motion_file, policy_path):
        motion = np.load(motion_file)
        self.motion_joint_pos = motion["joint_pos"]
        self.motion_joint_vel = motion["joint_vel"]
        self.motion_body_pos_w = motion["body_pos_w"]
        self.motion_body_quat_w = motion["body_quat_w"]
        self.motion_body_ang_vel_w = motion["body_ang_vel_w"]

        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_step(self.m, self.d)
        self.viewer = mujoco_viewer.MujocoViewer(self.m, self.d)
        self.viewer.cam.distance = 5.0
        
        model = onnx.load(policy_path)
        self.load(model)

        self.policy = onnxruntime.InferenceSession(policy_path)
    
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
        # root_pos = self.d.qpos.astype(np.float32)[0:3]
        # root_quat = self.d.qpos.astype(np.float32)[3:7]
        root_pos = self.d.xpos[self.xml_body_names.index(anchor_name)]
        root_quat = self.d.xquat[self.xml_body_names.index(anchor_name)]
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
    
    def calc_robot_elastic(self, anchor_name, timesteps):
        robot_dof_pos, _, _, robot_quat, robot_ang_vel = self.extract_data(anchor_name)
        motion_dof_pos = self.motion_joint_pos[timesteps, self.lab_to_xml]
        motion_quat = self.motion_body_quat_w[timesteps, self.lab_body_names.index(anchor_name), :]
        motion_ang_vel = self.motion_body_ang_vel_w[timesteps, self.lab_body_names.index(anchor_name), :]

        r_dof_pos = torch.exp(-torch.sum(torch.square(torch.tensor(robot_dof_pos) - torch.tensor(motion_dof_pos)), dim=-1))
        r_anchor_quat_w = torch.exp(-(quat_error_magnitude(torch.tensor(motion_quat), torch.tensor(robot_quat))** 2) / 0.4**2)
        r_anchor_ang_vel_w = torch.exp(-torch.sum(torch.square(torch.tensor(robot_ang_vel) - torch.tensor(motion_ang_vel)), dim=-1) / 3.14**2)
        beta = r_dof_pos * r_anchor_quat_w * r_anchor_ang_vel_w
        log_beta = torch.log10(beta + 1e-6)
        return torch.floor(-log_beta).long()

    def run(self):
        sim_duration = 60.0
        sim_dt = 0.005
        sim_decimation = 4
        timestep = 0
        anchor_name = "pelvis"
        action_buffer = np.zeros((self.num_action, ), dtype=np.float32)
        elastics = 0

        # 初始化状态
        self.d.qpos[-self.num_action:] = self.motion_joint_pos[0, self.lab_to_xml]
        self.d.qvel[-self.num_action:] = self.motion_joint_vel[0, self.lab_to_xml]
        self.d.qpos[:3] = self.motion_body_pos_w[0, self.lab_body_names.index(anchor_name), :]
        self.d.qpos[3:7] = self.motion_body_quat_w[0, self.lab_body_names.index(anchor_name), :]
        
        for i in tqdm(range(int(sim_duration / sim_dt)), desc="Running simulation..."):
            xml_joint_pos, xml_joint_vel, root_pos, root_quat, ang_vel = self.extract_data(anchor_name)
            
            if i % sim_decimation == 0:
                # timestep = min(timestep, self.motion_joint_pos.shape[0] - 1)
                command = np.concatenate((self.motion_joint_pos[timestep, :], self.motion_joint_vel[timestep, :]), axis=-1)
                motion_anchor_ori_b = self.calc_motion_anchor_ori_b(
                    torch.tensor(root_pos), torch.tensor(root_quat),
                    torch.tensor(self.motion_body_pos_w[timestep, self.lab_body_names.index(anchor_name), :]),
                    torch.tensor(self.motion_body_quat_w[timestep, self.lab_body_names.index(anchor_name), :])
                    ).numpy()
                base_ang_vel = ang_vel
                joint_pos = xml_joint_pos[self.xml_to_lab] - self.lab_default_joint_pos
                joint_vel = xml_joint_vel[self.xml_to_lab]
                last_actions = action_buffer

                obs = np.concatenate([
                    command,
                    motion_anchor_ori_b,
                    base_ang_vel,
                    joint_pos,
                    joint_vel,
                    last_actions
                ]).astype(np.float32).reshape(1, -1)

                lab_actions = self.policy.run(['actions'], {'obs': obs})[0].squeeze()
                action_buffer = lab_actions.copy()
                scale_actions = lab_actions * self.lab_action_scale

                pd_target = scale_actions[self.lab_to_xml] + self.lab_default_joint_pos[self.lab_to_xml]

                self.viewer.cam.lookat = self.d.qpos.astype(np.float32)[:3]
                self.viewer.render()
                frames = self.calc_robot_elastic(anchor_name, timestep).item()
                if elastics > 0:
                    elastics -= 1
                else:
                    elastics = frames
                    print(elastics)
                    timestep = (timestep + 1) % self.motion_joint_pos.shape[0]
            
            torque = pd_control(pd_target, xml_joint_pos, self.lab_joint_stiffness[self.lab_to_xml], np.zeros_like(self.lab_joint_damping), xml_joint_vel, self.lab_joint_damping[self.lab_to_xml])
            self.d.ctrl = torque
            mujoco.mj_step(self.m, self.d)

        self.viewer.close()

# ================= 主程序 =================
if __name__ == "__main__":
    # 路径配置
    xml_path = "/home/ubuntu/projects/RoboJuDo/assets/robots/g1/g1_23dof_rev_1_0.xml"
    motion_file = "/home/ubuntu/projects/hjj-robot_lab/source/robot_lab/robot_lab/tasks/manager_based/elastictracking/config/g1/motion/motions_npz/gvhmr_single_stand.npz"
    policy_path = "/home/ubuntu/projects/hjj-robot_lab/logs/rsl_rl/unitree_g1_elastictracking_flat/2026-01-13_15-19-22/exported/policy.onnx"
    
    r = RobotLabSim2Sim(xml_path, motion_file, policy_path)
    r.run()