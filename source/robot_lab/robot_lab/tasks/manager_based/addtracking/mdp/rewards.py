# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from robot_lab.tasks.manager_based.addtracking.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def motion_global_anchor_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_global_anchor_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_relative_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_relative_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_global_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward

def whole_com_balance(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_ground_pos = contact_sensor.data.contact_pos_w[..., 0, :]   # 与地面接触的点信息

    valid_mask = ~torch.isnan(contact_ground_pos[..., 0])   # 只需根据x轴判断哪些是有效接触点
    num_contacts = torch.sum(valid_mask, dim=1, keepdim=True)   # 计算每个环境的接触点数量
    contact_ground_pos = torch.nan_to_num(contact_ground_pos, nan=0.)   # 将nan设置为0
    contact_ground_pos_xy = contact_ground_pos[..., :2]
    contact_center_xy = torch.sum(contact_ground_pos_xy, dim=1) / num_contacts.clamp(min=1.0) # 防止除0

    # 计算接触点到中心的距离平方(方差)
    dist_square = torch.sum((contact_ground_pos_xy - contact_center_xy.unsqueeze(1))**2, dim=-1)    # 求出每个环境每个刚体到中心的距离
    sum_dist_square = torch.sum(dist_square * valid_mask.float(), dim=1)    # 求和每个环境所有接触的刚体到中心的距离
    
    spread = torch.sqrt(sum_dist_square / num_contacts.clamp(min=1.0).squeeze(-1))  # 防止除0
    whole_com_pos_w = command.robot_whole_com_pos_w[..., :2]
    dist_com_to_center = torch.norm(whole_com_pos_w - contact_center_xy, dim=-1)

    reward = torch.exp(-dist_com_to_center / (std**2 * spread.clamp(min=0.0005))) # 防止除0
    is_air = (num_contacts == 0).squeeze(-1)
    reward = reward * (~is_air).float()
    return reward
