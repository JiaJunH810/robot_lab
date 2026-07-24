import time
import os
import struct
import mujoco
import mujoco.viewer
import numpy as np
import torch
import onnx
from tqdm import tqdm
import onnxruntime

@torch.jit.script
def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    shape = q.shape
    q = q.reshape(-1, 4)
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1).view(shape)

@torch.jit.script
def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp(min=eps)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    if q1.shape != q2.shape:
        msg = f"Expected input quaternion shape mismatch: {q1.shape} != {q2.shape}."
        raise ValueError(msg)
    shape = q1.shape
    q1 = q1.reshape(-1, 4)
    q2 = q2.reshape(-1, 4)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
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
    shape = vec.shape
    quat = quat.reshape(-1, 4)
    vec = vec.reshape(-1, 3)
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
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
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


class RobotLabAMPSim2Sim:
    def __init__(self, xml_path, policy_path):
        self.m = mujoco.MjModel.from_xml_path(xml_path)
        self.m.opt.timestep = 0.001
        self.d = mujoco.MjData(self.m)
        mujoco.mj_resetDataKeyframe(self.m, self.d, 0)
        mujoco.mj_step(self.m, self.d)
        self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
        self.viewer.cam.distance = 5.0
        self.viewer.cam.lookat = [0, 0, 0.7]

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
        print(f"default_joint_pos: {', '.join(map(str, self.lab_default_joint_pos))}")
        print(f"joint_stiffness: {', '.join(map(str, self.lab_joint_stiffness))}")
        print(f"joint_damping: {', '.join(map(str, self.lab_joint_damping))}")
        print(f"action_scale: {', '.join(map(str, self.lab_action_scale))}")
        print(f"body_names: {self.lab_body_names}")

        # G1 XML 执行器名无 _joint 后缀，lab_order 有 _joint 后缀，做兼容匹配
        _xml_set = set(self.xml_order)
        def _find_xml(lab_name):
            if lab_name in _xml_set: return lab_name
            s = lab_name.replace("_joint", "")
            if s in _xml_set: return s
            raise ValueError(f"Cannot match '{lab_name}' in XML order")
        _lab_set = set(self.lab_order)
        def _find_lab(xml_name):
            if xml_name in _lab_set: return xml_name
            s = xml_name + "_joint"
            if s in _lab_set: return s
            raise ValueError(f"Cannot match '{xml_name}' in lab order")

        self.xml_to_lab = [self.xml_order.index(_find_xml(lab)) for lab in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(_find_lab(xml)) for xml in self.xml_order]

    def extract_data(self, anchor_name):
        dof_pos = self.d.qpos.astype(np.float32)[-self.num_action:]
        dof_vel = self.d.qvel.astype(np.float32)[-self.num_action:]
        root_pos = self.d.xpos[self.xml_body_names.index(anchor_name)]
        root_quat = self.d.xquat[self.xml_body_names.index(anchor_name)]
        ang_vel = self.d.qvel.astype(np.float32)[3:6]
        return (dof_pos, dof_vel, root_pos, root_quat, ang_vel)

    def _open_js_device(self):
        """Try to open /dev/input/js0 (or js1, js2) directly via Linux joystick API."""
        for dev in ["/dev/input/js0", "/dev/input/js1", "/dev/input/js2"]:
            try:
                fd = os.open(dev, os.O_RDONLY | os.O_NONBLOCK)
                print(f"[Joystick] Opened {dev}")
                return fd
            except OSError:
                continue
        return None

    def _read_js_state(self):
        """Read all pending joystick events from the device, update axis/button state.
        Returns (axes, buttons) where axes is a dict {idx: value} and buttons is a set of pressed button indices.
        """
        if self._js_fd is None:
            return {}, set()
        # drain all pending events
        while True:
            try:
                data = os.read(self._js_fd, 8)
                if len(data) < 8:
                    break
                t_ms, value, etype, num = struct.unpack('<IhBB', data)
                # axis event: etype=0x02, value in [-32767, 32767]
                if etype == 0x02:
                    self._js_axes[num] = value / 32767.0
                # button event: etype=0x01, value 1=pressed 0=released
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

    def read_joystick_command(self):
        """Read velocity command from joystick (/dev/input/js0) with keyboard fallback via GLFW.

        Joystick:  Left stick=线速度(vx/vy)  Right stick=角速度(wz)  A=急停
        Keyboard:  W/S=vx  A/D=vy  Q/E=wz  Space=急停
        """
        vx, vy, wz = 0.0, 0.0, 0.0

        # ---- joystick via Linux js device (no library needed) ----
        if self._js_fd is not None:
            try:
                axes, buttons = self._read_js_state()

                # debug: print axis mapping on button press (help find correct mapping)
                if buttons and not self._js_mapped:
                    print(f"[Joystick] axes: {dict(sorted(axes.items()))}  buttons: {buttons}")
                    self._js_mapped = True

                def dead(val, dz=0.1):
                    return 0.0 if abs(val) < dz else val
                # Common Xbox mapping: axis 0=LX, 1=LY, 2=RX, 3=RY  (may differ per controller)
                # Adjust these indices if your controller uses different mapping
                vx = -dead(axes.get(self._js_axis_vx, 0.0)) * self._cmd_scale[0]
                vy = -dead(axes.get(self._js_axis_vy, 0.0)) * self._cmd_scale[1]
                wz = -dead(axes.get(self._js_axis_wz, 0.0)) * self._cmd_scale[2]
                if 0 in buttons:  # button 0 = A (Xbox) / X (PS)
                    vx = vy = wz = 0.0
            except Exception:
                pass

        # ---- keyboard fallback (GLFW polling) ----
        if self._glfw_window is not None:
            try:
                import glfw
                win = self._glfw_window
                if glfw.get_key(win, glfw.KEY_W) == glfw.PRESS: vx += self._cmd_scale[0]
                if glfw.get_key(win, glfw.KEY_S) == glfw.PRESS: vx -= self._cmd_scale[0]
                if glfw.get_key(win, glfw.KEY_A) == glfw.PRESS: vy += self._cmd_scale[1]
                if glfw.get_key(win, glfw.KEY_D) == glfw.PRESS: vy -= self._cmd_scale[1]
                if glfw.get_key(win, glfw.KEY_Q) == glfw.PRESS: wz += self._cmd_scale[2]
                if glfw.get_key(win, glfw.KEY_E) == glfw.PRESS: wz -= self._cmd_scale[2]
                if glfw.get_key(win, glfw.KEY_SPACE) == glfw.PRESS:
                    vx = vy = wz = 0.0
            except Exception:
                pass

        return np.array([vx, vy, wz], dtype=np.float32)

    def check_quit(self):
        if self._glfw_window is not None:
            try:
                import glfw
                return glfw.get_key(self._glfw_window, glfw.KEY_ESCAPE) == glfw.PRESS
            except Exception:
                pass
        return False

    def run(self):
        # ---- init joystick & keyboard ----
        self._js_fd = self._open_js_device()
        self._js_axes = {}     # axis index → normalized value [-1, 1]
        self._js_buttons = set()  # set of pressed button indices
        self._js_mapped = False
        # axis indices: change these if your controller uses different mapping
        # (press any button to see actual axis values in console)
        self._js_axis_vx = 1   # left stick Y → forward velocity
        self._js_axis_vy = 0   # left stick X → lateral velocity
        self._js_axis_wz = 2   # right stick X → yaw rate
        self._glfw_window = None  # launch_passive 无 .window，键盘控速暂不可用
        self._cmd_scale = [1.0, 1.5, 0.0]  # vx_max, vy_max, wz_max

        # ---- sim params ----
        sim_duration = 120.0
        sim_dt = 0.001
        sim_decimation = 20
        anchor_name = "base_link"
        history_length = 4
        action_buffer = np.zeros((self.num_action, ), dtype=np.float32)

        # IsaacLab history: per-term, oldest-first. Each term has its own history buffer.
        # 6 terms: cmd(3), ang_vel(3), grav(3), joint_pos(N), joint_vel(N), actions(N)
        hist_cmd = np.zeros((history_length, 3), dtype=np.float32)
        hist_ang = np.zeros((history_length, 3), dtype=np.float32)
        hist_grav = np.zeros((history_length, 3), dtype=np.float32)
        hist_pos = np.zeros((history_length, self.num_action), dtype=np.float32)
        hist_vel = np.zeros((history_length, self.num_action), dtype=np.float32)
        hist_act = np.zeros((history_length, self.num_action), dtype=np.float32)

        def push_history(buf, new_val):
            """Push new value, oldest at buf[0], newest at buf[-1]."""
            buf[:-1] = buf[1:]
            buf[-1] = new_val

        # ---- init state: default standing pose ----
        self.d.qpos[-self.num_action:] = self.lab_default_joint_pos[self.lab_to_xml]
        self.d.qvel[-self.num_action:] = 0.0
        self.d.qpos[:3] = [0.0, 0.0, 0.94]
        self.d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        print("手柄: 左摇杆=线速度(vx/vy)  右摇杆=角速度(wz)  A=急停")
        print("键盘: W/S=前后 A/D=左右 Q/E=转向 Space=急停 Esc=退出")

        for i in tqdm(range(int(sim_duration / sim_dt)), desc="Running AMP sim2sim..."):
            xml_joint_pos, xml_joint_vel, root_pos, root_quat, ang_vel = self.extract_data(anchor_name)

            if i % sim_decimation == 0:
                if self.check_quit():
                    break

                vel_cmd = self.read_joystick_command()
                # vel_cmd[0] = -2.
                # vel_cmd[1] = 0.
                print(vel_cmd)
                # projected gravity: world [0,0,-1] rotated to body frame
                gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
                proj_grav = quat_apply(quat_inv(torch.tensor(root_quat, dtype=torch.float32)), gravity_w).numpy()

                joint_pos = xml_joint_pos[self.xml_to_lab] - self.lab_default_joint_pos
                joint_vel = xml_joint_vel[self.xml_to_lab]
                last_actions = action_buffer

                # per-term history: oldest-first (matching IsaacLab CircularBuffer)
                push_history(hist_cmd, vel_cmd)
                push_history(hist_ang, ang_vel)
                push_history(hist_grav, proj_grav)
                push_history(hist_pos, joint_pos)
                push_history(hist_vel, joint_vel)
                push_history(hist_act, last_actions)

                # concat: oldest first per term (matching flatten_history_dim=True)
                obs = np.concatenate([
                    hist_cmd.flatten(),   # 12: cmd_t-3, cmd_t-2, cmd_t-1, cmd_t
                    hist_ang.flatten(),   # 12: ang_vel_t-3, ..., ang_vel_t
                    hist_grav.flatten(),  # 12: grav_t-3, ..., grav_t
                    hist_pos.flatten(),   # 112: pos_t-3, ..., pos_t
                    hist_vel.flatten(),   # 112: vel_t-3, ..., vel_t
                    hist_act.flatten(),   # 112: act_t-3, ..., act_t
                ]).astype(np.float32).reshape(1, -1)

                lab_actions = self.policy.run(['actions'], {'obs': obs})[0].squeeze()
                action_buffer = lab_actions.copy()
                scale_actions = lab_actions * self.lab_action_scale

                pd_target = scale_actions[self.lab_to_xml] + self.lab_default_joint_pos[self.lab_to_xml]

                self.viewer.cam.lookat = self.d.qpos.astype(np.float32)[:3]
                self.viewer.sync()

            torque = pd_control(pd_target, xml_joint_pos, self.lab_joint_stiffness[self.lab_to_xml], np.zeros_like(self.lab_joint_damping), xml_joint_vel, self.lab_joint_damping[self.lab_to_xml])
            self.d.ctrl = torque
            mujoco.mj_step(self.m, self.d)
            time.sleep(self.m.opt.timestep)

        if self._js_fd is not None:
            os.close(self._js_fd)
        self.viewer.close()

# ================= 主程序 =================
if __name__ == "__main__":
    xml_path = "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/assets/ENX/biped_ENX_1_1.xml"
    policy_path = "/home/cyborg/Desktop/projects/robot_lab/logs/rsl_rl/cyborg_beyondamp/2026-07-09_10-19-53/exported/policy.onnx"

    r = RobotLabAMPSim2Sim(xml_path, policy_path)
    r.run()
