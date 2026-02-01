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
import glob
import random

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
        if os.path.isfile(motion_file):
            self.files = glob.glob(motion_file)
        else:
            self.files = glob.glob(f"{motion_file}/**/*.npz", recursive=True)
        assert len(self.files) != 0, f"Invalid file path: {motion_file}"

        self.time_step_total = []
        self.joint_pos = []
        self.joint_vel = []
        self._body_pos_w = []
        self._body_quat_w = []
        self._body_lin_vel_w = []
        self._body_ang_vel_w = []
        self.num_motions = len(self.files)

        for motion_file in self.files:
            data = np.load(motion_file)
            self.fps = data['fps']
            self.joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            self.joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            self._body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            self._body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            self._body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            self._body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            self.time_step_total.append(data['joint_pos'].shape[0])
        self._body_indexes = body_indexes
        
        self.joint_pos = torch.cat(self.joint_pos, dim=0)
        self.joint_vel = torch.cat(self.joint_vel, dim=0)
        self._body_pos_w = torch.cat(self._body_pos_w, dim=0)
        self._body_quat_w = torch.cat(self._body_quat_w, dim=0)
        self._body_lin_vel_w = torch.cat(self._body_lin_vel_w, dim=0)
        self._body_ang_vel_w = torch.cat(self._body_ang_vel_w, dim=0)
        self.time_step_total = torch.tensor(self.time_step_total, device=device, dtype=torch.long)
        self.motion_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(self.time_step_total, dim=0)[:-1]])

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

        self.robot: Articulation = env.scene[cfg.asset_name]    # 通过asset_name获取到机器人信息
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)
        self.motion_ids = torch.arange(self.num_envs, device=self.device) % self.motion.num_motions
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)  # 记录每个机器人当前播放到了动作文件的第几帧

        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0

        # 自适应采样
        self.bin_count = int(self.motion.time_step_total.max().float() // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self.bin_failed_count = torch.zeros((self.motion.num_motions, self.bin_count), dtype=torch.float, device=self.device)  # 记录了从训练开始到现在，所有机器人在第N个格子上摔倒了几次
        self._current_bin_failed = torch.zeros((self.motion.num_motions, self.bin_count), dtype=torch.float, device=self.device)   # 只记录这一个step里有哪些机器人摔倒了，稍后会合进总账
        # 这是一个衰减权重的卷积核
        # 因为机器人在第100帧摔倒了，但通常不是100帧的问题，而是99帧或者98帧的问题，所以不仅会给100帧标记1次失败，还会给99帧标记0.8次失败，这样权重衰减下去
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
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

        self.elastic_pred: torch.Tensor | None = None
        self.elastic_gt: torch.Tensor | None = None

    # @property装饰器， 使得可以像访问变量一样访问函数
    @property
    def _current_steps(self) -> torch.Tensor:
        starts = self.motion.motion_starts[self.motion_ids]
        safe_steps = torch.minimum(self.time_steps, self.motion.time_step_total[self.motion_ids] - 1)
        return starts + safe_steps

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self._current_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self._current_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._current_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._current_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._current_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._current_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self._current_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self._current_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self._current_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self._current_steps, self.motion_anchor_body_index]

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
    
    def calc_elastic_groud_truth(self):
        error_anchor_body_ang_vel = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)
        error_body_pos = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1)
        error_anchor_body_quat = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        error_joint_pos = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        error_joint_vel = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        error_anchor_body_ang_vel = error_anchor_body_ang_vel / 8.0
        error_body_pos = error_body_pos / 0.3
        error_anchor_body_quat = error_anchor_body_quat / 0.8
        error_joint_pos =  error_joint_pos / 3.0
        error_joint_vel = error_joint_vel / 30.0

        self.elastic_gt = torch.stack([
            error_anchor_body_ang_vel,
            error_body_pos,
            error_anchor_body_quat,
            error_joint_pos,
            error_joint_vel
        ], dim=1)
        return self.elastic_gt

    def _update_metrics(self):  # 计算与参考动作的误差
        self.metrics["error_anchor_pos"] = torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_anchor_lin_vel"] = torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1)
        self.metrics["error_anchor_ang_vel"] = torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1)

        self.metrics["error_body_pos"] = torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            dim=-1
        )

        self.metrics["error_body_lin_vel"] = torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(
            dim=-1
        )
        self.metrics["error_body_ang_vel"] = torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(
            dim=-1
        )

        self.metrics["error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        episode_failed = self._env.termination_manager.terminated[env_ids]  # 读取非超时而终结的环境
        if torch.any(episode_failed):
            failed_env_ids = env_ids[episode_failed]
            failed_motion_ids = self.motion_ids[failed_env_ids]
            motion_lengths = self.motion.time_step_total[failed_motion_ids].float()
            progress = self.time_steps[failed_env_ids] / motion_lengths
            # 算出当前是在第几个时间段(Bin)结束掉的
            fail_bin_indices = (progress * self.bin_count).long().clamp(0, self.bin_count - 1)
            self._current_bin_failed.index_put_(
                (failed_motion_ids, fail_bin_indices), 
                torch.ones_like(failed_motion_ids, dtype=torch.float), 
                accumulate=True
            )
        
        # Sample Motion
        motion_failed = self.bin_failed_count.sum(dim=1)
        motion_sampling_probs = torch.sqrt(motion_failed) + self.cfg.adaptive_uniform_ratio
        motion_sampling_probs = motion_sampling_probs / motion_sampling_probs.sum()
        self.motion_ids[env_ids] = torch.multinomial(
            motion_sampling_probs, 
            num_samples=len(env_ids), 
            replacement=True
        )

        # Sample Bin
        reset_motion_ids = self.motion_ids[env_ids]
        sampling_probabilities = self.bin_failed_count[reset_motion_ids] + self.cfg.adaptive_uniform_ratio / float(self.bin_count)    # 基础概率 = 历史失败次数 + 一个很小的均匀底数 (防止有些地方一次都没失败过导致概率为0)
        probs_padded = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(1), 
            (0, self.cfg.adaptive_kernel_size - 1), 
            mode="replicate"
        )
        sampling_probabilities = torch.nn.functional.conv1d(
            probs_padded, 
            self.kernel.view(1, 1, -1)
        ).squeeze(1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum(dim=-1, keepdim=True)
        sampled_bins = torch.multinomial(sampling_probabilities, 1).squeeze(-1)   # 抽签：根据刚才算的概率分布，决定每个环境从哪个 Bin 开始

        # 随机化从段中的具体哪个帧开始
        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total[reset_motion_ids] - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum(dim=1)    # 计算香农熵，用来描述混乱程度和不确定性。熵很高说明概率平坦，熵很低说明概率分布尖锐
        H_norm = H / math.log(self.bin_count)   # 归一化熵
        pmax, imax = sampling_probabilities.max(dim=1)
        self.metrics["sampling_entropy"][env_ids] = H_norm
        self.metrics["sampling_top1_prob"][env_ids] = pmax
        self.metrics["sampling_top1_bin"][env_ids] = imax.float() / self.bin_count

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._adaptive_sampling(env_ids)

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]  # 读取配置里的噪声范围
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)  # 关节也加噪音
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        # 将机器人的状态写到仿真中去
        self.robot.write_joint_state_to_sim(joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )

    def _update_command(self):
        self.time_steps += 1    # 这一帧结束了，进度条往前走一格

        env_ids = torch.where(self.time_steps >= self.motion.time_step_total[self.motion_ids])[0]    # 找到超时的环境
        self._resample_command(env_ids)

        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(1, len(self.cfg.body_names), 1)

        delta_pos_w = robot_anchor_pos_w_repeat
        delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]   # 将高度调节为参考动作的高度
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))    # 算出机器人当前朝向和参考动作的朝向之间在yaw上的偏差

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w) # 把参考动作的身体部件全部旋转一下，使得与机器人脸朝一个方向

        # 位置对齐：
        # (a) self.body_pos_w - anchor_pos_w_repeat: 算出参考动作里，手脚相对于它自己基座的偏移量
        # (b) quat_apply(delta_ori_w, ...): 把这个偏移量旋转一下（跟着机器人转）
        # (c) + delta_pos_w: 把旋转后的偏移量，加到机器人的当前位置上
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

        # 每个段中历史累积的失败次数(EMA)
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

        for i in range(len(self.cfg.body_names)):
            self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
            self.goal_body_visualizers[i].visualize(self.body_pos_relative_w[:, i], self.body_quat_relative_w[:, i])


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand    # 这里把config和class锁死了。这意味着只要用了MotionCommandCfg，系统就必须去实例化MotionCommand这个类

    asset_name: str = MISSING   # MISSING表示先空着，但是一定要赋值

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.05
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
