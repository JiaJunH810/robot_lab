# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.utils.math import (
    matrix_from_quat, 
    subtract_frame_transforms, 
    quat_unique,
    quat_apply_inverse,
    yaw_quat
)
from robot_lab.tasks.manager_based.vaemimic.mdp.commands import MotionCommand

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

def projected_gravity(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_anchor_quat_w = command.robot_anchor_quat_w
    gravity = torch.tensor([0.0, 0.0, -1.0], device=env.device).repeat(env.num_envs, 1)
    return quat_apply_inverse(robot_anchor_quat_w, gravity)

def robot_body_pos_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)

def robot_body_ori_r(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
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

def vqvae_code(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    code = command.vqvae_code
    return code.view(env.num_envs, -1)

def future_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_pos = command.get_future_obs(attr="anchor_pos_w", horizon=command.cfg.horizon).reshape(-1, 3)
    future_quat = command.get_future_obs(attr="anchor_quat_w", horizon=command.cfg.horizon).reshape(-1, 4)
    current_pos = command.get_future_obs(attr="anchor_pos_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 3)
    current_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 4)
    current_quat_yaw = yaw_quat(current_quat)
    pos, _ = subtract_frame_transforms(
        current_pos,
        current_quat_yaw,
        future_pos,
        future_quat,
    )
    return pos.view(env.num_envs, command.cfg.horizon, 3)

def future_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_pos = command.get_future_obs(attr="anchor_pos_w", horizon=command.cfg.horizon).reshape(-1, 3)
    future_quat = command.get_future_obs(attr="anchor_quat_w", horizon=command.cfg.horizon).reshape(-1, 4)
    current_pos = command.get_future_obs(attr="anchor_pos_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 3)
    current_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 4)
    current_quat_yaw = yaw_quat(current_quat)
    _, ori = subtract_frame_transforms(
        current_pos,
        current_quat_yaw,
        future_pos,
        future_quat,
    )
    ori = quat_unique(ori)
    return ori.view(env.num_envs, command.cfg.horizon, 4)

def future_anchor_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_vel = command.get_future_obs(attr="anchor_lin_vel_w", horizon=command.cfg.horizon).reshape(-1, 3)
    current_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 4)
    current_quat_yaw = yaw_quat(current_quat)
    vel = quat_apply_inverse(current_quat_yaw, future_vel)
    return vel.view(env.num_envs, command.cfg.horizon, 3)

def future_anchor_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_pos = command.get_future_obs(attr="anchor_pos_w", horizon=command.cfg.horizon)
    height = future_pos[..., 2]
    return height.view(env.num_envs, command.cfg.horizon, 1)

def future_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_body_pos = command.get_future_obs(attr="future_body_pos_w", horizon=command.cfg.horizon)
    _, horizon, num_body, _ = future_body_pos.shape
    current_anchor_pos = command.get_future_obs(attr="anchor_pos_w", horizon=1).unsqueeze(2).expand(-1, horizon, num_body, -1).reshape(-1, 3)
    current_anchor_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).unsqueeze(2).expand(-1, horizon, num_body, -1).reshape(-1, 4)
    current_anchor_quat_yaw = yaw_quat(current_anchor_quat)
    future_body_pos = future_body_pos.reshape(-1, 3)
    pos, _ = subtract_frame_transforms(
        current_anchor_pos,
        current_anchor_quat_yaw,
        future_body_pos,
        None
    )
    return pos.view(env.num_envs, horizon, 3 * num_body)