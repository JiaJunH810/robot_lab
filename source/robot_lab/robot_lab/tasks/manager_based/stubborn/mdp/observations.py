# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import matrix_from_quat, quat_apply, quat_inv, quat_mul, subtract_frame_transforms, yaw_quat

from robot_lab.tasks.manager_based.stubborn.mdp.commands import MotionCommand

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
    # Already in yaw-aligned frame relative to robot anchor
    return command.robot_body_pos_yaw.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Already in yaw-aligned frame: 4D quat → 6D rotation matrix (first 2 cols)
    mat = matrix_from_quat(command.robot_body_quat_yaw)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Motion anchor position relative to robot anchor, in yaw-aligned frame
    yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    pos = quat_apply(yaw_inv, command.anchor_pos_w - command.robot_anchor_pos_w)
    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    # Motion anchor orientation in yaw-aligned frame
    yaw_inv = quat_inv(yaw_quat(command.robot_anchor_quat_w))
    ori = quat_mul(yaw_inv, command.anchor_quat_w)
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)
