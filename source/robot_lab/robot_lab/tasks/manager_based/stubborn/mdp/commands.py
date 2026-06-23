# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        self._body_indexes = body_indexes

        if os.path.isdir(motion_file):
            # 目录模式：加载所有 .npz 并沿时间轴拼接
            npz_files = sorted([f for f in os.listdir(motion_file) if f.endswith(".npz")])
            assert len(npz_files) > 0, f"No .npz files found in {motion_file}"
            fps = None
            joint_pos_list, joint_vel_list = [], []
            body_pos_list, body_quat_list = [], []
            body_lin_vel_list, body_ang_vel_list = [], []
            self._file_offsets = []  # 每个文件在拼接后的起止帧 [start, end)

            offset = 0
            for fname in npz_files:
                full = os.path.join(motion_file, fname)
                data = np.load(full)
                n = data["joint_pos"].shape[0]
                self._file_offsets.append((fname, offset, offset + n))
                joint_pos_list.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
                joint_vel_list.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
                body_pos_list.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
                body_quat_list.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
                body_lin_vel_list.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
                body_ang_vel_list.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
                if fps is None:
                    fps = float(data["fps"])
                offset += n

            self.fps = fps
            self.joint_pos = torch.cat(joint_pos_list, dim=0)
            self.joint_vel = torch.cat(joint_vel_list, dim=0)
            self._body_pos_w = torch.cat(body_pos_list, dim=0)
            self._body_quat_w = torch.cat(body_quat_list, dim=0)
            self._body_lin_vel_w = torch.cat(body_lin_vel_list, dim=0)
            self._body_ang_vel_w = torch.cat(body_ang_vel_list, dim=0)
            self.time_step_total = self.joint_pos.shape[0]
        else:
            # 单文件模式
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
            self._file_offsets = None
            data = np.load(motion_file)
            self.fps = data["fps"]
            self.joint_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            self.joint_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            self._body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            self._body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            self._body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            self.time_step_total = self.joint_pos.shape[0]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        # 加载所有运动序列（拼接），用于随机初始化
        self.motion_pool = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        # 加载主跟踪序列
        self.motion = MotionLoader(self.cfg.track_file, self.body_indexes, device=self.device)

        # 预计算 motion_pool 每一帧 → 主跟踪序列最近帧的映射
        self._pool_to_main = self._compute_pool_to_main()

        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._end_stall = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        self.body_lin_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_ang_vel_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)

        self.robot_body_pos_yaw = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.robot_body_quat_yaw = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.robot_body_quat_yaw[:, :, 0] = 1.0
        self.robot_body_lin_vel_yaw = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.robot_body_ang_vel_yaw = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_failed = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)], device=self.device
        )
        self.kernel = self.kernel / self.kernel.sum()

        self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_anchor_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_lin_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_body_ang_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    def _compute_pool_to_main(self) -> torch.Tensor:
        """预计算 motion_pool 每一帧 → 主跟踪序列最近帧的映射（纯张量，不走 for 循环）。

        策略：先按 roll/pitch 粗筛，再按 joint_pos + joint_vel 精排。
        """
        # ---- 1. 批量计算 gravity Z 分量（roll/pitch 代理） ----
        gravity = torch.tensor([0., 0., -1.], device=self.device)
        # quat_inv 批量化：对每个帧的四元数取逆
        pool_grav = quat_apply(quat_inv(self.motion_pool.body_quat_w[:, 0]), gravity)  # [pool_n, 3]
        main_grav = quat_apply(quat_inv(self.motion.body_quat_w[:, 0]), gravity)        # [main_n, 3]
        grav_diff = torch.abs(pool_grav[:, None, 2] - main_grav[None, :, 2])  # [pool_n, main_n]

        # ---- 2. 批量计算 joint 误差 ----
        jpos_err = ((self.motion_pool.joint_pos[:, None, :] - self.motion.joint_pos[None, :, :]) ** 2).mean(dim=2)
        jvel_err = ((self.motion_pool.joint_vel[:, None, :] - self.motion.joint_vel[None, :, :]) ** 2).mean(dim=2)
        joint_err = jpos_err + 0.1 * jvel_err  # [pool_n, main_n]

        # ---- 3. 粗筛 + 精排 ----
        GRAV_THRESH = 0.3
        candidates = grav_diff < GRAV_THRESH
        no_cand = ~candidates.any(dim=1)  # 没有任何候选的帧

        # 候选帧内取 min joint_err，无候选的取 min grav_diff
        masked = joint_err.clone()
        masked[~candidates] = float("inf")
        best = torch.argmin(masked, dim=1)
        best[no_cand] = torch.argmin(grav_diff, dim=1)[no_cand]

        return best

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_yaw, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_yaw).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_relative_w - self.robot_body_lin_vel_yaw, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_relative_w - self.robot_body_ang_vel_yaw, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]
        if torch.any(episode_failed):
            current_bin_index = torch.clamp(
                (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
            )
            fail_bins = current_bin_index[env_ids][episode_failed]
            self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

        # Sample
        sampling_probabilities = self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(0).unsqueeze(0),
            (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)

        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        n_env = len(env_ids)

        # 从 motion_pool 随机抽帧
        pool_idx = torch.randint(0, self.motion_pool.time_step_total, (n_env,), device=self.device)

        # 初始化机器人到池帧的状态
        root_pos = self.motion_pool.body_pos_w[pool_idx, 0] + self._env.scene.env_origins[env_ids]
        root_ori = self.motion_pool.body_quat_w[pool_idx, 0]
        root_lin_vel = self.motion_pool.body_lin_vel_w[pool_idx, 0]
        root_ang_vel = self.motion_pool.body_ang_vel_w[pool_idx, 0]
        joint_pos = self.motion_pool.joint_pos[pool_idx].clone()
        joint_vel = self.motion_pool.joint_vel[pool_idx].clone()

        # 小扰动（与 BeyondMimic 原版一致）
        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (n_env, 6), device=self.device)
        root_pos += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori = quat_mul(orientations_delta, root_ori)
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (n_env, 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos, root_ori, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )

        # 用预计算映射找到主序列最近帧，从那里开始跟踪
        self.time_steps[env_ids] = self._pool_to_main[pool_idx]

    def _update_command(self):
        self.time_steps += 1

        # 到达运动末尾时：先卡在最后一帧 N 步，再重新采样
        at_end = self.time_steps >= self.motion.time_step_total
        self._end_stall = torch.where(at_end, self._end_stall + 1, torch.zeros_like(self._end_stall))
        self.time_steps = torch.where(at_end, self.motion.time_step_total - 1, self.time_steps)

        # stall 步数走完后才触发重新采样（每个 env 随机 50~200 步 = 1~4 秒）
        stall_thresh = torch.randint(50, 201, (self.num_envs,), device=self.device)
        env_ids = torch.where(self._end_stall >= stall_thresh)[0]
        self._resample_command(env_ids)

        # Yaw-aligned frame: remove robot base yaw from all body tracking
        robot_anchor_yaw_inv = quat_inv(yaw_quat(self.robot_anchor_quat_w))
        robot_anchor_yaw_inv_repeat = robot_anchor_yaw_inv[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        # Motion reference → yaw-aligned frame (relative to robot anchor)
        motion_body_pos_rel = self.body_pos_w - self.robot_anchor_pos_w[:, None, :]
        self.body_pos_relative_w = quat_apply(robot_anchor_yaw_inv_repeat, motion_body_pos_rel)
        self.body_quat_relative_w = quat_mul(robot_anchor_yaw_inv_repeat, self.body_quat_w)
        self.body_lin_vel_relative_w = quat_apply(robot_anchor_yaw_inv_repeat, self.body_lin_vel_w)
        self.body_ang_vel_relative_w = quat_apply(robot_anchor_yaw_inv_repeat, self.body_ang_vel_w)

        # Robot body → yaw-aligned frame
        robot_body_pos_rel = self.robot_body_pos_w - self.robot_anchor_pos_w[:, None, :]
        self.robot_body_pos_yaw = quat_apply(robot_anchor_yaw_inv_repeat, robot_body_pos_rel)
        self.robot_body_quat_yaw = quat_mul(robot_anchor_yaw_inv_repeat, self.robot_body_quat_w)
        self.robot_body_lin_vel_yaw = quat_apply(robot_anchor_yaw_inv_repeat, self.robot_body_lin_vel_w)
        self.robot_body_ang_vel_yaw = quat_apply(robot_anchor_yaw_inv_repeat, self.robot_body_ang_vel_w)

        self.bin_failed_count = (
            self.cfg.adaptive_alpha * self._current_bin_failed + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
        )
        self._current_bin_failed.zero_()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                )
                self.goal_anchor_visualizer = VisualizationMarkers(
                    self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                )

                self.current_body_visualizers = []
                self.goal_body_visualizers = []
                for name in self.cfg.body_names:
                    self.current_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                        )
                    )
                    self.goal_body_visualizers.append(
                        VisualizationMarkers(
                            self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                        )
                    )

            self.current_anchor_visualizer.set_visibility(True)
            self.goal_anchor_visualizer.set_visibility(True)
            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].set_visibility(True)
                self.goal_body_visualizers[i].set_visibility(True)

        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
        self.goal_anchor_visualizer.visualize(self.anchor_pos_w, self.anchor_quat_w)

        # Transform yaw-aligned goal body markers back to world frame for visualization
        robot_anchor_yaw = yaw_quat(self.robot_anchor_quat_w)
        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            goal_pos_w = self.robot_anchor_pos_w + quat_apply(robot_anchor_yaw, self.body_pos_relative_w[:, i])
            goal_quat_w = quat_mul(robot_anchor_yaw, self.body_quat_relative_w[:, i])
            self.goal_body_visualizers[i].visualize(goal_pos_w, goal_quat_w)


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    track_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
