"""TorchScript locomotion-policy sim2sim runner.

This is the ``policy.pt`` counterpart of ``loco_sim2sim.py``.  The
observation layout, history, joint order, joystick mapping and MuJoCo PD
controller are kept identical to the ONNX runner.
"""

import math
import os
import struct

import mujoco
import mujoco_viewer
import numpy as np
import torch
from tqdm import tqdm


POLICY_JOINT_NAMES = (
    "J_hip_l_roll",
    "J_hip_r_roll",
    "J_hip_l_yaw",
    "J_hip_r_yaw",
    "J_hip_l_pitch",
    "J_hip_r_pitch",
    "J_knee_l_pitch",
    "J_knee_r_pitch",
    "J_ankle_l_pitch",
    "J_ankle_r_pitch",
    "J_ankle_l_roll",
    "J_ankle_r_roll",
)

DEFAULT_JOINT_POS = np.array(
    [0.0, 0.0, 0.0, 0.0, 0.4, -0.4, 0.7, -0.7, 0.3, -0.3, 0.0, 0.0],
    dtype=np.float32,
)

JOINT_STIFFNESS = np.array(
    [109.504] * 8 + [259.826] * 4,
    dtype=np.float32,
)

JOINT_DAMPING = np.array(
    [6.971] * 8 + [16.541] * 4,
    dtype=np.float32,
)

ACTION_SCALE = np.array(
    [0.753] * 8 + [0.115] * 4,
    dtype=np.float32,
)


def pd_control(target_q, q, kp, target_dq, dq, kd):
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_to_rot_matrix(quat_wxyz):
    qw, qx, qy, qz = quat_wxyz
    return np.array(
        [
            [
                1 - 2 * qy * qy - 2 * qz * qz,
                2 * qx * qy - 2 * qz * qw,
                2 * qx * qz + 2 * qy * qw,
            ],
            [
                2 * qx * qy + 2 * qz * qw,
                1 - 2 * qx * qx - 2 * qz * qz,
                2 * qy * qz - 2 * qx * qw,
            ],
            [
                2 * qx * qz - 2 * qy * qw,
                2 * qy * qz + 2 * qx * qw,
                1 - 2 * qx * qx - 2 * qy * qy,
            ],
        ],
        dtype=np.float32,
    )


def projected_gravity(quat_wxyz):
    rotation = quat_to_rot_matrix(quat_wxyz)
    return rotation.T @ np.array([0.0, 0.0, -1.0], dtype=np.float32)


class LocoSim2SimPT:
    def __init__(
        self,
        xml_path,
        policy_path,
        history_length=15,
        cmd_max=(0.5, 0.5, 0.5),
        cycle_time=0.9,
    ):
        self.history_length = history_length
        self.cmd_max = np.asarray(cmd_max, dtype=np.float32)
        self.cycle_time = cycle_time
        self.num_action = len(POLICY_JOINT_NAMES)
        # phase(2) + ang_vel(3) + gravity(3) + command(3)
        # + joint_pos(12) + joint_vel(12) + last_action(12) = 47.
        self.single_observation_size = 11 + 3 * self.num_action
        self.policy_input_size = history_length * self.single_observation_size

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = 0.001
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_step(self.model, self.data)

        self.xml_order = [
            mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )
            for actuator_id in range(self.model.nu)
        ]
        self.lab_order = list(POLICY_JOINT_NAMES)

        missing = [name for name in self.lab_order if name not in self.xml_order]
        if missing:
            raise ValueError(f"MuJoCo XML is missing policy actuators: {missing}")
        if self.model.nu != self.num_action:
            raise ValueError(
                f"Expected {self.num_action} actuators, XML contains {self.model.nu}"
            )

        self.xml_to_lab = [self.xml_order.index(name) for name in self.lab_order]
        self.lab_to_xml = [self.lab_order.index(name) for name in self.xml_order]

        self.lab_default_joint_pos = DEFAULT_JOINT_POS.copy()
        self.lab_joint_stiffness = JOINT_STIFFNESS.copy()
        self.lab_joint_damping = JOINT_DAMPING.copy()
        self.lab_action_scale = ACTION_SCALE.copy()

        print(" -------------------- xml order -------------------- ")
        print(self.xml_order)
        print(" -------------------- lab order -------------------- ")
        print(self.lab_order)
        print(self.lab_default_joint_pos)
        print(self.lab_joint_stiffness)
        print(self.lab_joint_damping)
        print(self.lab_action_scale)

        torch.set_num_threads(1)
        self.policy = torch.jit.load(policy_path, map_location="cpu").eval()
        self._validate_policy()

        self.ankle_l_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "ankle_l_roll_link",
        )
        self.ankle_r_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "ankle_r_roll_link",
        )

        self.viewer = mujoco_viewer.MujocoViewer(self.model, self.data)
        self.viewer.cam.distance = 3.0
        self.viewer.cam.azimuth = 90.0
        self.viewer.cam.elevation = -20.0

        self.hist_phase = np.zeros((history_length, 2), dtype=np.float32)
        self.hist_cmd = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_ang = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_grav = np.zeros((history_length, 3), dtype=np.float32)
        self.hist_pos = np.zeros(
            (history_length, self.num_action),
            dtype=np.float32,
        )
        self.hist_vel = np.zeros(
            (history_length, self.num_action),
            dtype=np.float32,
        )
        self.hist_act = np.zeros(
            (history_length, self.num_action),
            dtype=np.float32,
        )

        self.action_buffer = np.zeros(self.num_action, dtype=np.float32)
        self.vel_cmd = np.zeros(3, dtype=np.float32)
        self.cmd_resample_timer = 0
        self.cmd_resample_interval = int(10.0 / 0.02)

        self.joystick_fd = self._open_js_device()
        self.joystick_axes = {}
        self.joystick_buttons = set()
        self.joystick_mapped = False
        self.joystick_axis_vx = 1
        self.joystick_axis_vy = 0
        self.joystick_axis_wz = 3

    def _validate_policy(self):
        sample = torch.zeros(
            (1, self.policy_input_size),
            dtype=torch.float32,
        )
        with torch.inference_mode():
            output = self.policy(sample)
            if isinstance(output, (tuple, list)):
                output = output[0]
        if tuple(output.shape) != (1, self.num_action):
            raise ValueError(
                f"Expected policy output [1, {self.num_action}], "
                f"got {tuple(output.shape)}"
            )
        print(
            f"[TorchScript] input=[1, {self.policy_input_size}] "
            f"output={list(output.shape)}"
        )

    @staticmethod
    def _push_history(buffer, new_value):
        buffer[:-1] = buffer[1:]
        buffer[-1] = new_value

    @staticmethod
    def _open_js_device():
        for device in ("/dev/input/js0", "/dev/input/js1", "/dev/input/js2"):
            try:
                fd = os.open(device, os.O_RDONLY | os.O_NONBLOCK)
                print(f"[Joystick] Opened {device}")
                return fd
            except OSError:
                pass
        print("[Joystick] Not found, using uniform random commands")
        return None

    def _read_js_state(self):
        if self.joystick_fd is None:
            return {}, set()
        while True:
            try:
                data = os.read(self.joystick_fd, 8)
                if len(data) < 8:
                    break
                _, value, event_type, number = struct.unpack("<IhBB", data)
                event_type &= 0x7F
                if event_type == 0x02:
                    self.joystick_axes[number] = value / 32767.0
                elif event_type == 0x01:
                    if value:
                        self.joystick_buttons.add(number)
                    else:
                        self.joystick_buttons.discard(number)
            except BlockingIOError:
                break
            except OSError:
                break
        return self.joystick_axes, self.joystick_buttons

    def _get_joystick_cmd(self):
        axes, buttons = self._read_js_state()
        if buttons and not self.joystick_mapped:
            print(
                f"[Joystick] axes: {dict(sorted(axes.items()))} "
                f"buttons: {buttons}"
            )
            self.joystick_mapped = True

        def dead(value, dead_zone=0.1):
            return 0.0 if abs(value) < dead_zone else value

        vx = -dead(axes.get(self.joystick_axis_vx, 0.0)) * self.cmd_max[0]
        vy = -dead(axes.get(self.joystick_axis_vy, 0.0)) * self.cmd_max[1]
        wz = -dead(axes.get(self.joystick_axis_wz, 0.0)) * self.cmd_max[2]
        if 0 in buttons:
            vx = vy = wz = 0.0
        return np.array([vx, vy, wz], dtype=np.float32)

    def _forward_policy(self, observation):
        observation_tensor = torch.from_numpy(observation)
        with torch.inference_mode():
            output = self.policy(observation_tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
        actions = output.squeeze(0).cpu().numpy().astype(np.float32)
        if actions.shape != (self.num_action,):
            raise RuntimeError(f"Unexpected action shape: {actions.shape}")
        return actions

    def run(self, sim_duration=120.0):
        sim_dt = 0.001
        sim_decimation = 20
        total_steps = int(sim_duration / sim_dt)

        self.data.qpos[-self.num_action:] = self.lab_default_joint_pos[
            self.lab_to_xml
        ]
        self.data.qvel[-self.num_action:] = 0.0
        self.data.qpos[:3] = [0.0, 0.0, 0.94]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]

        for _ in range(10):
            q = self.data.qpos.astype(np.float32)[-self.num_action:]
            dq = self.data.qvel.astype(np.float32)[-self.num_action:]
            target = self.lab_default_joint_pos[self.lab_to_xml]
            torque = pd_control(
                target,
                q,
                self.lab_joint_stiffness[self.lab_to_xml],
                np.zeros(self.num_action, dtype=np.float32),
                dq,
                self.lab_joint_damping[self.lab_to_xml],
            )
            self.data.ctrl = torque
            mujoco.mj_step(self.model, self.data)

        pd_target = self.lab_default_joint_pos[self.lab_to_xml].copy()
        control_step = 0

        try:
            for simulation_step in range(total_steps):
                if simulation_step % sim_decimation == 0:
                    phase_step = control_step
                    control_step += 1

                    # Match mdp.phase(): episode_length_buf * step_dt / cycle_time.
                    # phase_step starts at zero, matching the initial observation.
                    policy_dt = sim_dt * sim_decimation
                    phase = (phase_step * policy_dt / self.cycle_time) % 1.0
                    phase_observation = np.array(
                        [
                            math.sin(2.0 * math.pi * phase),
                            math.cos(2.0 * math.pi * phase),
                        ],
                        dtype=np.float32,
                    )

                    if self.joystick_fd is not None:
                        self.vel_cmd = self._get_joystick_cmd()
                        if np.linalg.norm(self.vel_cmd[:2]) < 0.1:
                            self.vel_cmd[:2] = 0.0
                    else:
                        self.cmd_resample_timer += 1
                        if (
                            self.cmd_resample_timer
                            >= self.cmd_resample_interval
                        ):
                            self.vel_cmd = np.array(
                                [
                                    np.random.uniform(
                                        -self.cmd_max[0],
                                        self.cmd_max[0],
                                    ),
                                    np.random.uniform(
                                        -self.cmd_max[1],
                                        self.cmd_max[1],
                                    ),
                                    np.random.uniform(
                                        -self.cmd_max[2],
                                        self.cmd_max[2],
                                    ),
                                ],
                                dtype=np.float32,
                            )
                            if np.linalg.norm(self.vel_cmd[:2]) < 0.2:
                                self.vel_cmd[:2] = 0.0
                            self.cmd_resample_timer = 0

                    joint_pos_xml = self.data.qpos.astype(np.float32)[
                        -self.num_action:
                    ]
                    joint_vel_xml = self.data.qvel.astype(np.float32)[
                        -self.num_action:
                    ]
                    root_quat = self.data.qpos.astype(np.float32)[3:7].copy()
                    base_ang_vel_body = self.data.qvel.astype(np.float32)[3:6]

                    gravity = projected_gravity(root_quat)
                    joint_pos_rel = (
                        joint_pos_xml[self.xml_to_lab]
                        - self.lab_default_joint_pos
                    )
                    joint_vel_lab = joint_vel_xml[self.xml_to_lab]

                    self._push_history(self.hist_phase, phase_observation)
                    self._push_history(self.hist_cmd, self.vel_cmd)
                    self._push_history(
                        self.hist_ang,
                        base_ang_vel_body * 0.25,
                    )
                    self._push_history(self.hist_grav, gravity)
                    self._push_history(self.hist_pos, joint_pos_rel)
                    self._push_history(
                        self.hist_vel,
                        joint_vel_lab * 0.05,
                    )
                    self._push_history(
                        self.hist_act,
                        self.action_buffer,
                    )

                    observation = np.concatenate(
                        [
                            self.hist_phase.flatten(),
                            self.hist_ang.flatten(),
                            self.hist_grav.flatten(),
                            self.hist_cmd.flatten(),
                            self.hist_pos.flatten(),
                            self.hist_vel.flatten(),
                            self.hist_act.flatten(),
                        ]
                    ).astype(np.float32).reshape(1, -1)

                    if observation.shape != (1, self.policy_input_size):
                        raise RuntimeError(
                            f"Unexpected observation shape: "
                            f"{observation.shape}"
                        )

                    actions = self._forward_policy(observation)
                    self.action_buffer = actions.copy()

                    scaled_actions = actions * self.lab_action_scale
                    pd_target = (
                        scaled_actions[self.lab_to_xml]
                        + self.lab_default_joint_pos[self.lab_to_xml]
                    )

                    self.viewer.cam.lookat = self.data.qpos.astype(
                        np.float32
                    )[:3]
                    self.viewer.render()

                    if control_step % 5 == 0:
                        time_seconds = simulation_step * sim_dt
                        base_height = self.data.qpos[2]
                        qw, qx, qy, qz = self.data.qpos[3:7]
                        pitch = math.asin(
                            max(
                                -1.0,
                                min(1.0, 2 * (qw * qy - qz * qx)),
                            )
                        )
                        roll = math.atan2(
                            2 * (qw * qx + qy * qz),
                            1 - 2 * (qx * qx + qy * qy),
                        )
                        position_error = pd_target - joint_pos_xml
                        diagnostic_torque = (
                            position_error
                            * self.lab_joint_stiffness[self.lab_to_xml]
                            - joint_vel_xml
                            * self.lab_joint_damping[self.lab_to_xml]
                        )
                        ankle_l_z = self.data.xpos[self.ankle_l_id][2]
                        ankle_r_z = self.data.xpos[self.ankle_r_id][2]
                        print(
                            f"[{control_step:5d} "
                            f"t={time_seconds:5.1f}s] "
                            f"BaseZ={base_height:.3f} "
                            f"Pitch={math.degrees(pitch):5.1f} "
                            f"Roll={math.degrees(roll):5.1f} | "
                            f"AnkleZ L={ankle_l_z:.3f} "
                            f"R={ankle_r_z:.3f} | "
                            f"ActMean={np.mean(actions):.3f} "
                            f"MaxErr={np.max(np.abs(position_error)):.3f} "
                            f"MaxTrq={np.max(np.abs(diagnostic_torque)):.0f} | "
                            f"Cmd:vx={self.vel_cmd[0]:.2f} "
                            f"vy={self.vel_cmd[1]:.2f} "
                            f"wz={self.vel_cmd[2]:.2f}"
                        )
                        max_error_index = int(np.argmax(np.abs(position_error)))
                        print(
                            f"MaxErrJoint={self.xml_order[max_error_index]} "
                            f"Target={pd_target[max_error_index]:.4f} "
                            f"Position={joint_pos_xml[max_error_index]:.4f} "
                            f"Velocity={joint_vel_xml[max_error_index]:.4f} "
                            f"Torque={diagnostic_torque[max_error_index]:.2f}"
                        )
                        if base_height < 0.5:
                            print("  *** FALLEN ***")

                joint_pos_xml = self.data.qpos.astype(np.float32)[
                    -self.num_action:
                ]
                joint_vel_xml = self.data.qvel.astype(np.float32)[
                    -self.num_action:
                ]
                torque = pd_control(
                    pd_target,
                    joint_pos_xml,
                    self.lab_joint_stiffness[self.lab_to_xml],
                    np.zeros(self.num_action, dtype=np.float32),
                    joint_vel_xml,
                    self.lab_joint_damping[self.lab_to_xml],
                )
                self.data.ctrl = torque
                mujoco.mj_step(self.model, self.data)
        finally:
            if self.joystick_fd is not None:
                os.close(self.joystick_fd)
            self.viewer.close()


if __name__ == "__main__":
    XML_PATH = (
        "/home/cyborg/Desktop/projects/robot_lab/source/sim2sim/"
        "assets/temp/biped_temp_1_0_fixed.xml"
    )
    POLICY_PATH = (
        "/home/cyborg/Desktop/projects/sim2real_robotlab/src/deploy_real/policy/policy.pt"
    )

    simulation = LocoSim2SimPT(
        XML_PATH,
        POLICY_PATH,
        history_length=15,
        cmd_max=(0.6, 0.4, 0.6),
    )
    simulation.run(sim_duration=120.0)
