# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_contact_time(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_air = contact_sensor.compute_first_air(env.step_dt, env.physics_dt)[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_contact_time < threshold) * first_air, dim=-1)
    return reward


def track_root_height(env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg,
                      mask_delay: bool = False, delay_env_rew_ratio: float = 1.0) -> torch.Tensor:
    """Reward for maintaining default standing height.

    When mask_delay=True, delay envs in the buffer period get the reward
    amplified by delay_env_rew_ratio, while all other envs receive the base
    reward unchanged. This gives normal envs a dense height signal for learning
    to stand, while still incentivising delay envs to get up.
    """
    asset = env.scene[asset_cfg.name]
    desired_height = asset.data.default_root_state[:, 2]
    cur_root_height = asset.data.root_pos_w[:, 2]
    height_error = torch.square(desired_height - cur_root_height)
    reward = torch.exp(-height_error / std**2)

    if mask_delay:
        from robot_lab.tasks.manager_based.beyondamp.mdp.terminations import DelayedTerminationManager
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            in_buffer = tm._delay_env_mask & (tm._delay_counters > 0)
            reward = torch.where(in_buffer, reward * delay_env_rew_ratio, reward)

    return reward


def self_collisions(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize self-collisions indicated by contact on arm/waist bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)
