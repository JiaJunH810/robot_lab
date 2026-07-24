"""MuJoCo sim2sim runner for the policy used by ``sim2real_encos``.

The observation, history, command filter and PD controller mirror
``sim2real_Encos.cpp`` together with ``Cyborg_Encos_Config.yaml``.  The policy
is a TorchScript model with a [1, 705] input (15 frames x 47 values) and a
12-dimensional action output.
"""

import argparse
import math
import os
import struct
import time
from collections import deque

import mujoco
import mujoco.viewer
import numpy as np
import torch


LEG_JOINT_NAMES = (
    "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch",
    "J_knee_l_pitch", "J_ankle_l_pitch", "J_ankle_l_roll",
    "J_hip_r_roll", "J_hip_r_yaw", "J_hip_r_pitch",
    "J_knee_r_pitch", "J_ankle_r_pitch", "J_ankle_r_roll",
)


def quaternion_to_euler(quat_wxyz):
    """Convert a MuJoCo [w, x, y, z] quaternion to roll, pitch and yaw."""
    w, x, y, z = quat_wxyz
    roll = math.atan2(2.0 * (w * x + y * z),
                      1.0 - 2.0 * (x * x + y * y))
    sin_pitch = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(sin_pitch))
    yaw = math.atan2(2.0 * (w * z + x * y),
                     1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=np.float32)


class EncosSim2Sim:
    """Run the Encos real-deployment policy against a MuJoCo robot model."""

    def __init__(self, xml_path, policy_path, cmd=(0.0, 0.0, 0.0),
                 use_joystick=True):
        self.sim_dt = 0.001
        self.control_dt = 0.01
        self.decimation = int(round(self.control_dt / self.sim_dt))

        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.model.opt.timestep = self.sim_dt
        self.data = mujoco.MjData(self.model)

        # Active values in Cyborg_Encos_Config.yaml.
        self.default_q = np.array([
            0.0, 0.0, 0.4, 0.7, 0.3, 0.0,
            0.0, 0.0, -0.4, -0.7, -0.3, 0.0,
        ], dtype=np.float32)
        self.action_scale = np.array([
            0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
            0.4, 0.35, 0.35, 0.35, 0.35, 0.35,
        ], dtype=np.float32)
        self.kp = np.array([
            250.0, 120.0, 300.0, 300.0, 80.0, 80.0,
            250.0, 120.0, 300.0, 300.0, 80.0, 80.0,
        ], dtype=np.float32)
        self.kd = np.array([
            10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
            10.0, 10.0, 10.0, 10.0, 3.0, 3.0,
        ], dtype=np.float32)
        self.stance_kp = np.array([
            300.0, 300.0, 500.0, 500.0, 400.0, 400.0,
            300.0, 300.0, 500.0, 500.0, 400.0, 400.0,
        ], dtype=np.float32)
        self.stance_kd = np.array([
            15.0, 12.0, 19.0, 25.0, 20.0, 20.0,
            15.0, 12.0, 19.0, 25.0, 20.0, 20.0,
        ], dtype=np.float32)

        self.cycle_time = 0.8
        self.clip_obs = 50.0
        self.clip_action = 5.0
        self.frame_stack = 15
        self.single_obs_size = 47
        self.policy_input_size = self.frame_stack * self.single_obs_size

        self.leg_qpos_addr, self.leg_dof_addr, self.leg_actuator_ids = (
            self._resolve_leg_indices()
        )
        self.ankle_l_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ankle_l_roll_link")
        self.ankle_r_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "ankle_r_roll_link")
        self.upper_joint_ids = self._upper_joint_ids()
        self.upper_targets = self._make_upper_targets()

        self.policy = torch.jit.load(policy_path, map_location="cpu").eval()
        self._validate_policy()
        print(f"[EncosSim2Sim] XML: {xml_path}")
        print(f"[EncosSim2Sim] Policy: {policy_path}")
        print(f"[EncosSim2Sim] Leg order: {list(LEG_JOINT_NAMES)}")

        self.history = deque(
            (np.zeros(self.single_obs_size, dtype=np.float32)
             for _ in range(self.frame_stack)),
            maxlen=self.frame_stack,
        )
        self.action = np.zeros(12, dtype=np.float32)
        self.target_cmd = np.asarray(cmd, dtype=np.float32)
        if self.target_cmd.shape != (3,):
            raise ValueError("cmd must contain exactly (vx, vy, yaw_rate)")
        # The real callback rounds joystick commands to one decimal place.
        self.target_cmd = np.round(self.target_cmd * 10.0) / 10.0
        self.command = np.zeros(3, dtype=np.float32)
        self.lowlevel_count = 0
        self.walk_stand_count = 0
        self.desired_heading = 0.0
        self.heading_initialized = False

        self.use_joystick = use_joystick
        self.joystick_fd = self._open_joystick() if use_joystick else None
        self.joystick_axes = {}
        self.joystick_buttons = set()
        self.joystick_mapped = False
        # Match loco_sim2sim.py's GameSir mapping.  Stick-up/left values from
        # Linux joystick devices are negative, hence the minus signs below.
        self.joystick_axis_vx = 1   # left stick Y
        self.joystick_axis_vy = 0   # left stick X
        self.joystick_axis_yaw = 3  # right stick X
        # Match Cyborg_Encos_Config.yaml joy_forward/joy_side/joy_turn.
        self.joystick_cmd_max = np.array([0.44, 0.8, 0.6], dtype=np.float32)

        mujoco.mj_resetData(self.model, self.data)
        self._set_initial_pose()
        mujoco.mj_forward(self.model, self.data)

    def _resolve_leg_indices(self):
        qpos_addr = []
        dof_addr = []
        actuator_ids = []
        for name in LEG_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if joint_id < 0 or actuator_id < 0:
                raise ValueError(f"MuJoCo XML is missing leg joint/actuator {name!r}")
            qpos_addr.append(self.model.jnt_qposadr[joint_id])
            dof_addr.append(self.model.jnt_dofadr[joint_id])
            actuator_ids.append(actuator_id)
        return (np.asarray(qpos_addr), np.asarray(dof_addr),
                np.asarray(actuator_ids))

    def _upper_joint_ids(self):
        leg_ids = {
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in LEG_JOINT_NAMES
        }
        return [joint_id for joint_id in range(self.model.njnt)
                if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE
                and joint_id not in leg_ids]

    def _make_upper_targets(self):
        # Pose used by the previous Encos sim2sim setup. Missing joints are
        # simply ignored, so the lower-body-only XML is also supported.
        desired = {
            "J_waist_yaw": 0.0,
            "J_waist_pitch": 0.0,
            "J_arm_l_02": -1.4,
            "J_arm_l_04": -2.09,
            "J_arm_l_06": 0.9,
            "J_arm_r_02": 1.4,
            "J_arm_r_04": 2.09,
            "J_arm_r_06": -0.9,
        }
        targets = {}
        for joint_id in self.upper_joint_ids:
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            value = desired.get(name, 0.0)
            if self.model.jnt_limited[joint_id]:
                low, high = self.model.jnt_range[joint_id]
                value = float(np.clip(value, low, high))
            targets[joint_id] = value
        return targets

    def _validate_policy(self):
        sample = torch.zeros((1, self.policy_input_size), dtype=torch.float32)
        try:
            with torch.inference_mode():
                output = self.policy(sample)
                if isinstance(output, tuple):
                    output = output[0]
        except Exception as exc:
            raise ValueError(
                f"Policy does not accept [1, {self.policy_input_size}] input"
            ) from exc
        if tuple(output.shape) != (1, 12):
            raise ValueError(f"Expected policy output [1, 12], got {tuple(output.shape)}")

    def _set_initial_pose(self):
        self.data.qpos[0:3] = (0.0, 0.0, 0.94)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.leg_qpos_addr] = self.default_q
        for joint_id, target in self.upper_targets.items():
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = target
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

    @staticmethod
    def _rate_limit(current, target, max_step=0.002):
        return current + np.clip(target - current, -max_step, max_step)

    @staticmethod
    def _open_joystick():
        for path in ("/dev/input/js0", "/dev/input/js1", "/dev/input/js2"):
            try:
                fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
                print(f"[Joystick] Opened {path}; axes 1/0/3 -> vx/vy/yaw")
                return fd
            except OSError:
                pass
        print("[Joystick] Not found; using the command-line fixed command")
        return None

    def _update_joystick_target(self):
        if self.joystick_fd is None:
            return
        while True:
            try:
                event = os.read(self.joystick_fd, 8)
                if len(event) != 8:
                    break
                _, value, event_type, number = struct.unpack("<IhBB", event)
                event_type &= 0x7F  # Remove the Linux JS_EVENT_INIT flag.
                if event_type == 0x02:
                    self.joystick_axes[number] = float(value) / 32767.0
                elif event_type == 0x01:
                    if value:
                        self.joystick_buttons.add(number)
                    else:
                        self.joystick_buttons.discard(number)
            except BlockingIOError:
                break
            except OSError:
                return

        if self.joystick_buttons and not self.joystick_mapped:
            print(f"[Joystick] axes={dict(sorted(self.joystick_axes.items()))} "
                  f"buttons={sorted(self.joystick_buttons)}")
            self.joystick_mapped = True

        def deadzone(value, threshold=0.1):
            return 0.0 if abs(value) < threshold else value

        stick = np.array([
            -deadzone(self.joystick_axes.get(self.joystick_axis_vx, 0.0)),
            -deadzone(self.joystick_axes.get(self.joystick_axis_vy, 0.0)),
            -deadzone(self.joystick_axes.get(self.joystick_axis_yaw, 0.0)),
        ], dtype=np.float32)
        joystick_cmd = stick * self.joystick_cmd_max

        # Keep the real deployment's one-decimal command quantization.
        self.target_cmd = np.round(joystick_cmd * 10.0) / 10.0
        # GameSir A button: immediately zero all velocity directions.
        if 0 in self.joystick_buttons:
            self.target_cmd[:] = 0.0
            self.command[:] = 0.0

    def _scaled_command(self, euler, body_ang_vel):
        self._update_joystick_target()
        self.command = self._rate_limit(self.command, self.target_cmd)
        scaled = self.command * np.array([2.0, 2.0, 1.0], dtype=np.float32)

        walking = float(np.abs(self.command).sum()) > 0.1
        if walking and not self.heading_initialized:
            self.desired_heading = float(euler[2])
            self.heading_initialized = True

        if walking and abs(float(self.command[2])) < 0.05:
            yaw_error = math.atan2(
                math.sin(float(euler[2]) - self.desired_heading),
                math.cos(float(euler[2]) - self.desired_heading),
            )
            correction = -(1.5 * yaw_error + 0.3 * float(body_ang_vel[2]))
            scaled[2] = np.clip(correction, -0.5, 0.5)
        elif abs(float(self.command[2])) >= 0.05:
            self.desired_heading = float(euler[2])
        elif not walking:
            self.desired_heading = float(euler[2])
            self.heading_initialized = False
        return scaled, walking

    def _compute_observation(self):
        # C++ ComputeObs increments this before calculating the phase.
        self.lowlevel_count += 1
        phase = (2.0 * math.pi * self.lowlevel_count * self.control_dt
                 / self.cycle_time)
        sin_cos = np.array([math.sin(phase), math.cos(phase)], dtype=np.float32)

        q = self.data.qpos[self.leg_qpos_addr].astype(np.float32, copy=True)
        dq = self.data.qvel[self.leg_dof_addr].astype(np.float32, copy=True)
        # MuJoCo free-joint rotational qvel is expressed in the local body frame.
        body_ang_vel = self.data.qvel[3:6].astype(np.float32, copy=True)
        euler = quaternion_to_euler(self.data.qpos[3:7])
        commands, walking = self._scaled_command(euler, body_ang_vel)

        if walking:
            self.walk_stand_count = 0
        elif self.walk_stand_count <= 160:
            commands[:] = 0.0
            self.walk_stand_count += 1
        elif abs(float(euler[0])) >= 0.1 or abs(float(euler[1])) >= 0.1:
            self.walk_stand_count = 0
        else:
            sin_cos[:] = 0.0
            commands[:] = 0.0

        obs = np.concatenate((
            sin_cos,
            commands,
            q - self.default_q,
            dq * 0.05,
            self.action,
            body_ang_vel,
            euler,
        )).astype(np.float32)
        if obs.size != self.single_obs_size:
            raise RuntimeError(f"Expected 47 observations, built {obs.size}")
        return np.clip(obs, -self.clip_obs, self.clip_obs)

    def _infer(self, observation):
        self.history.append(observation)
        policy_input = np.concatenate(tuple(self.history)).reshape(1, -1)
        tensor = torch.from_numpy(policy_input)
        with torch.inference_mode():
            output = self.policy(tensor)
            if isinstance(output, tuple):
                output = output[0]
        action = output.detach().cpu().numpy().reshape(-1)
        return np.clip(action, -self.clip_action, self.clip_action).astype(np.float32)

    def _apply_pd(self, leg_target, kp=None, kd=None):
        if kp is None:
            kp = self.kp
        if kd is None:
            kd = self.kd
        q = self.data.qpos[self.leg_qpos_addr]
        dq = self.data.qvel[self.leg_dof_addr]
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self.leg_actuator_ids] = (leg_target - q) * kp - dq * kd

        # The real deployment only owns the legs; in the full-body simulation
        # the remaining joints need a separate pose holder to represent the
        # upper-body controllers. Lower-body-only XMLs skip this block.
        for joint_id, target in self.upper_targets.items():
            name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if actuator_id < 0:
                continue
            qpos_addr = self.model.jnt_qposadr[joint_id]
            dof_addr = self.model.jnt_dofadr[joint_id]
            self.data.ctrl[actuator_id] = (
                80.0 * (target - self.data.qpos[qpos_addr])
                - 5.0 * self.data.qvel[dof_addr]
            )

    def _settle(self, seconds):
        for _ in range(max(0, int(seconds / self.sim_dt))):
            self._apply_pd(self.default_q, self.stance_kp, self.stance_kd)
            mujoco.mj_step(self.model, self.data)

    def run(self, duration=60.0, settle_duration=1.0, realtime=True):
        """Open the viewer and execute the policy for ``duration`` seconds."""
        self._settle(settle_duration)
        control_steps = int(duration / self.control_dt)
        target_q = self.default_q.copy()

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            viewer.cam.distance = 3.0
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -20.0
            for step in range(control_steps):
                if not viewer.is_running():
                    break
                wall_start = time.monotonic()
                observation = self._compute_observation()
                self.action = self._infer(observation)
                target_q = self.default_q + self.action * self.action_scale

                for _ in range(self.decimation):
                    self._apply_pd(target_q)
                    mujoco.mj_step(self.model, self.data)

                viewer.cam.lookat[:] = self.data.qpos[:3]
                viewer.sync()
                if step % 100 == 0:
                    euler = quaternion_to_euler(self.data.qpos[3:7])
                    print(
                        f"[t={step * self.control_dt:6.2f}s] "
                        f"z={self.data.qpos[2]:.3f} "
                        f"ankleL={self.data.xpos[self.ankle_l_id][2]:.3f} "
                        f"ankleR={self.data.xpos[self.ankle_r_id][2]:.3f} "
                        f"roll={math.degrees(float(euler[0])):6.1f}deg "
                        f"pitch={math.degrees(float(euler[1])):6.1f}deg "
                        f"cmd=({self.command[0]:.2f}, {self.command[1]:.2f}, "
                        f"{self.command[2]:.2f})"
                    )
                if realtime:
                    remaining = self.control_dt - (time.monotonic() - wall_start)
                    if remaining > 0.0:
                        time.sleep(remaining)

        if self.joystick_fd is not None:
            os.close(self.joystick_fd)
            self.joystick_fd = None


def parse_args():
    project_root = "/home/cyborg/Desktop/projects"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--xml",
        default=(f"{project_root}/robot_lab/source/sim2sim/assets/temp/"
                 "biped_temp_1_0_fixed.xml"),
    )
    parser.add_argument(
        "--policy",
        default=(f"{project_root}/sim2real_encos/src/deploy_real/policies/"
                 "c10000.pt"),
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--settle-duration", type=float, default=1.0)
    parser.add_argument("--vx", type=float, default=0.0)
    parser.add_argument("--vy", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--no-joystick", action="store_true")
    parser.add_argument("--no-realtime", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    simulator = EncosSim2Sim(
        args.xml,
        args.policy,
        cmd=(args.vx, args.vy, args.yaw),
        use_joystick=not args.no_joystick,
    )
    simulator.run(
        duration=args.duration,
        settle_duration=args.settle_duration,
        realtime=not args.no_realtime,
    )
