# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel

def delayed_joint_pos_rel(
    env: ManagerBasedEnv,
    action_name: str,
) -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    return action.delayed_joint_obs[:, :action.action_dim]


def delayed_joint_vel_rel(
    env: ManagerBasedEnv,
    action_name: str,
) -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    return action.delayed_joint_obs[:, action.action_dim:]


def delayed_base_ang_vel(
    env: ManagerBasedEnv,
    action_name: str,
) -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    return action.delayed_imu_obs[:, :3]


def delayed_projected_gravity(
    env: ManagerBasedEnv,
    action_name: str,
) -> torch.Tensor:
    action = env.action_manager.get_term(action_name)
    return action.delayed_imu_obs[:, 3:]

def phase(
    env: ManagerBasedRLEnv,
    cycle_time: float,
    command_name: str,
    command_threshold: float,
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long,)

    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase),],dim=-1)
    command = env.command_manager.get_command(command_name)
    moving = (
        (torch.linalg.norm(command[:, :2], dim=1) > command_threshold)
        | (torch.abs(command[:, 2]) > command_threshold)
    )

    return phase_tensor * moving.unsqueeze(-1).to(phase_tensor.dtype)

def external_force_xy(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_ids=[0]),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    # shape: [num_envs, num_bodies, 3]
    force = robot.permanent_wrench_composer.composed_force_as_torch

    # 取根部 body 的 x、y 外力
    return force[:, 0, :2]

def external_torque(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_ids=[0]),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]

    torque = robot.permanent_wrench_composer.composed_torque_as_torch

    # 根部 roll、pitch、yaw 外力矩
    return torque[:, 0, :]