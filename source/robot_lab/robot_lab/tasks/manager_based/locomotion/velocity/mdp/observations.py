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
    recovery_tilt_threshold: float,
    action_name: str | None = None,
) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long,)

    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase),],dim=-1)

    # 对齐 EngineAI（zqsa01 compute_observations）：相位观测恒定输入（0 命令也给出
    # 真实 sin/cos），不置 0。站立行为由奖励门控决定，观测保持与真实相位一致
    return phase_tensor
