# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from robot_lab.tasks.manager_based.beyondamp.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)

# ------------------------------ AMP ------------------------------#

def robot_body_pos_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """机器人所有 key body 的世界坐标位置（减去 env_origin，与专家数据对齐）。

    数据已由 MotionCommand.body_indexes 自动过滤到配置的 body_names。

    注意：这里减去 env_origins 是为了和 AMP 专家数据（MotionLoader 从 .npz 加载，
    录制时 env_origin 在原点附近）保持坐标系一致。否则 4096 个环境分布在 ±80m 范围
    的 body_pos_w 会和专家数据产生巨大分布偏移，破坏判别器训练。
    Returns: (num_envs, num_key_bodies * 3)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    pos = command.robot_body_pos_w - env.scene.env_origins.unsqueeze(1)
    return pos.view(env.num_envs, -1)


def robot_body_quat_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """机器人所有 key body 的世界坐标朝向（四元数）。

    Returns: (num_envs, num_key_bodies * 4)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_body_quat_w.view(env.num_envs, -1)


def robot_body_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """机器人所有 key body 的世界坐标线速度。

    Returns: (num_envs, num_key_bodies * 3)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_body_lin_vel_w.view(env.num_envs, -1)


def robot_body_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """机器人所有 key body 的世界坐标角速度。

    Returns: (num_envs, num_key_bodies * 3)
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_body_ang_vel_w.view(env.num_envs, -1)