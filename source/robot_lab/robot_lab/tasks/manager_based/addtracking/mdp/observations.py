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
    yaw_quat,
    quat_mul,
    quat_conjugate
)
from robot_lab.tasks.manager_based.addtracking.mdp.commands import MotionCommand

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

# ---------- future ---------- # 

def future_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_pos = command.get_future_obs(attr="anchor_pos_w", horizon=command.cfg.horizon).reshape(-1, 3)
    future_quat = command.get_future_obs(attr="anchor_quat_w", horizon=command.cfg.horizon).reshape(-1, 4)
    current_pos = command.get_future_obs(attr="anchor_pos_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 3)
    current_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 4)
    pos, _ = subtract_frame_transforms(
        current_pos,
        current_quat,
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
    _, ori = subtract_frame_transforms(
        current_pos,
        current_quat,
        future_pos,
        future_quat,
    )
    ori = quat_unique(ori)
    return ori.view(env.num_envs, command.cfg.horizon, 4)

def future_anchor_vel_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    future_vel = command.get_future_obs(attr="anchor_lin_vel_w", horizon=command.cfg.horizon).reshape(-1, 3)
    current_quat = command.get_future_obs(attr="anchor_quat_w", horizon=1).expand(-1, command.cfg.horizon, -1).reshape(-1, 4)
    vel = quat_apply_inverse(current_quat, future_vel)
    return vel.view(env.num_envs, command.cfg.horizon, 3)

# ---------- discriminator ---------- # 

def root_pos_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_root_pos = command.robot_anchor_pos_w
    motion_root_pos = command.anchor_pos_w
    root_pos_diff = motion_root_pos - robot_root_pos
    return root_pos_diff

def root_quat_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_root_quat = command.robot_anchor_quat_w
    motion_root_quat = command.anchor_quat_w

    robot_mat = matrix_from_quat(robot_root_quat)
    motion_mat = matrix_from_quat(motion_root_quat)

    robot_6d = robot_mat[..., :2].reshape(robot_mat.shape[0], -1)
    motion_6d = motion_mat[..., :2].reshape(motion_mat.shape[0], -1)
    return motion_6d - robot_6d

def root_lin_vel_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_root_lin_vel = command.robot_anchor_lin_vel_w
    motion_root_lin_vel = command.anchor_lin_vel_w
    return motion_root_lin_vel - robot_root_lin_vel

def root_ang_vel_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_root_ang_vel = command.robot_anchor_ang_vel_w
    motion_root_ang_vel = command.anchor_ang_vel_w
    return motion_root_ang_vel - robot_root_ang_vel

def joint_pos_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_joint_pos = command.robot_joint_pos
    motion_joint_pos = command.joint_pos
    return motion_joint_pos - robot_joint_pos

def joint_vel_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_joint_vel = command.robot_joint_vel
    motion_joint_vel = command.joint_vel
    return motion_joint_vel - robot_joint_vel

def body_pos_diff(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    robot_body_pos_w = command.robot_body_pos_w
    robot_anchor_pos_w = command.robot_anchor_pos_w
    
    motion_body_pos_w = command.body_pos_w
    motion_anchor_pos_w = command.anchor_pos_w

    robot_body_pos_rel = robot_body_pos_w - robot_anchor_pos_w.unsqueeze(1)
    motion_body_pos_rel = motion_body_pos_w - motion_anchor_pos_w.unsqueeze(1)

    body_pos_diff = motion_body_pos_rel - robot_body_pos_rel
    return body_pos_diff.view(body_pos_diff.shape[0], -1)
