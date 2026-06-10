# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""This script replay a motion from a csv file and output it to a npz file

.. code-block:: bash

    # Usage
    python csv_to_npz.py -f path_to_input.csv --input_fps 120
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np
import glob
from tqdm import tqdm
import os

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument("--input_file", "-f", type=str, required=True, help="The path to the input motion csv file.")
parser.add_argument("--input_fps", type=int, default=60, help="The fps of the input motion.")
parser.add_argument(
    "--frame_range",
    nargs=2,
    type=int,
    metavar=("START", "END"),
    help=(
        "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
        " loaded."
    ),
)
parser.add_argument("--output_name", type=str, help="The name of the motion npz file.")
parser.add_argument("--output_fps", type=int, default=50, help="The fps of the output motion.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
if not args_cli.output_name:
    # generate at the same location as input file
    args_cli.output_name = (
        "/".join(args_cli.input_file.split("/")[:-1]) + "/" + args_cli.input_file.split("/")[-1].replace(".csv", ".npz")
    )


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import (
    axis_angle_from_quat, matrix_from_quat,
    quat_apply_inverse, quat_conjugate, quat_mul,
    quat_slerp, subtract_frame_transforms,
)

##
# Pre-defined configs
##
from robot_lab.assets.cyborg import CYBORG_BIPED_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = CYBORG_BIPED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def ToTensor(array, device):
    return torch.tensor(array, dtype=torch.float32, device=device)


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        input_fps: int,
        output_fps: int,
        device: torch.device,
        frame_range: tuple[int, int] | None,
    ):
        self.motion_file = motion_file
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / self.input_fps
        self.output_dt = 1.0 / self.output_fps
        self.current_idx = 0
        self.device = device
        self.frame_range = frame_range
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    def _load_motion(self):
        """Loads the motion from the csv file.

        CSV 格式每行: root_x, root_y, root_z, quat_x, quat_y, quat_z, quat_w, dof_1, ..., dof_N
        quat 是 x,y,z,w 顺序，需要转为 w,x,y,z (Isaac Sim 格式)。
        """
        if self.frame_range is None:
            motion = np.loadtxt(self.motion_file, delimiter=",")
            self.motion_base_poss_input = ToTensor(motion[:, :3], device=self.device)
            self.motion_base_rots_input = ToTensor(motion[:, 3:7], device=self.device)
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw → wxyz
            self.motion_dof_poss_input = ToTensor(motion[:, 7:], device=self.device)
        else:
            motion = np.loadtxt(
                self.motion_file,
                delimiter=",",
                skiprows=self.frame_range[0] - 1,
                max_rows=self.frame_range[1] - self.frame_range[0] + 1,
            )
            self.motion_base_poss_input = ToTensor(motion[:, :3], device=self.device)
            self.motion_base_rots_input = ToTensor(motion[:, 3:7], device=self.device)
            self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]  # xyzw → wxyz
            self.motion_dof_poss_input = ToTensor(motion[:, 7:], device=self.device)

        self.input_frames = self.motion_base_poss_input.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt  # 运动序列所耗时间(s)
        print(f"Motion loaded ({self.motion_file}), duration: {self.duration} sec, frames: {self.input_frames}")

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(motion_file, sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=motion_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        frame_range=args_cli.frame_range,
    )

    # Extract scene entities
    robot = scene["robot"]
    joint_sdk_names = [
        # === 左腿 ===
        "J_hip_l_roll",
        "J_hip_l_yaw",
        "J_hip_l_pitch",
        "J_knee_l_pitch",
        "J_ankle_l_pitch",
        "J_ankle_l_roll",
        # === 右腿 ===
        "J_hip_r_roll",
        "J_hip_r_yaw",
        "J_hip_r_pitch",
        "J_knee_r_pitch",
        "J_ankle_r_pitch",
        "J_ankle_r_roll",
        # === 腰部 ===
        "J_waist_yaw",
        "J_waist_pitch",
        # === 左臂 ===
        "J_arm_l_01",
        "J_arm_l_02",
        "J_arm_l_03",
        "J_arm_l_04",
        "J_arm_l_05",
        "J_arm_l_06",
        "J_arm_l_07",
        # === 右臂 ===
        "J_arm_r_01",
        "J_arm_r_02",
        "J_arm_r_03",
        "J_arm_r_04",
        "J_arm_r_05",
        "J_arm_r_06",
        "J_arm_r_07",
    ]
    robot_joint_indexes = robot.find_joints(joint_sdk_names, preserve_order=True)[0]
    anchor_body_idx = robot.body_names.index("base_link")

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "joint_names": list(robot.joint_names),
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "body_pos_b": [],
        "body_quat_b": [],
        "body_ori_b": [],
        "body_lin_vel_b": [],
        "body_ang_vel_b": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

            # Compute body-frame (b) data relative to anchor body
            anchor_pos = robot.data.body_pos_w[0, anchor_body_idx]
            anchor_quat = robot.data.body_quat_w[0, anchor_body_idx]
            body_pos_w = robot.data.body_pos_w[0]
            body_quat_w = robot.data.body_quat_w[0]
            body_lin_vel_w = robot.data.body_lin_vel_w[0]
            body_ang_vel_w = robot.data.body_ang_vel_w[0]
            num_bodies = body_pos_w.shape[0]

            pos_b, quat_b = subtract_frame_transforms(
                anchor_pos[None, :].repeat(num_bodies, 1),
                anchor_quat[None, :].repeat(num_bodies, 1),
                body_pos_w,
                body_quat_w,
            )
            log["body_pos_b"].append(pos_b.cpu().numpy().copy())
            log["body_quat_b"].append(quat_b.cpu().numpy().copy())
            mat = matrix_from_quat(quat_b)
            log["body_ori_b"].append(mat[..., :2].reshape(num_bodies, -1).cpu().numpy().copy())

            vel_b = quat_apply_inverse(body_quat_w, body_lin_vel_w)
            log["body_lin_vel_b"].append(vel_b.cpu().numpy().copy())
            ang_b = quat_apply_inverse(body_quat_w, body_ang_vel_w)
            log["body_ang_vel_b"].append(ang_b.cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
                "body_pos_b",
                "body_quat_b",
                "body_ori_b",
                "body_lin_vel_b",
                "body_ang_vel_b",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(args_cli.output_name, **log)
            print("[INFO]: Motion npz file saved to", args_cli.output_name)
            return


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    if os.path.isfile(args_cli.input_file):
        motions = [args_cli.input_file]
    else:
        motions = glob.glob(f'{args_cli.input_file}/**/*.csv', recursive=True)

    for motion in tqdm(motions):
        basename = os.path.basename(motion).split('.')[0]
        args_cli.output_name = f"source/robot_lab/robot_lab/tasks/manager_based/beyondamp/config/cyborg/motion/{basename}.npz"
        print(args_cli.output_name)

        run_simulator(motion, sim, scene)


# python scripts/tools/beyondmimic/csv_to_npz.py -f /home/cyborg/Desktop/projects/AMP_mjlab/motion_data_csv/amp --input_fps 120
if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
