# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils import math as math_utils

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

    When mask_delay=True, only delay envs currently in the buffer period receive
    the reward (amplified by delay_env_rew_ratio). Normal envs and delay envs
    that have already recovered get zero. This matches AMP_mjlab's behaviour.
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
            reward = torch.where(in_buffer, reward * delay_env_rew_ratio, torch.zeros_like(reward))

    return reward


def track_head_height(
    env: ManagerBasedRLEnv,
    std: float,
    head_offset: tuple[float, float, float],
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward for maintaining head at its default height above the root.

    Head world position is computed by applying the root's orientation to a
    fixed local offset (head position in root frame), then adding root
    position. The reward penalises deviation of the resulting head Z from
    the default root Z plus the offset's z-component.

    When mask_delay=True, only delay envs currently in the buffer period
    receive the reward (amplified by delay_env_rew_ratio). Normal envs and
    delay envs that have already recovered get zero.

    Args:
        env: The reinforcement learning environment.
        std: Standard deviation for the Gaussian reward kernel.
        head_offset: (x, y, z) offset of head from root in root local frame.
        mask_delay: Whether to zero out reward for non-delay environments.
        delay_env_rew_ratio: Reward multiplier for delay environments in buffer.
        asset_cfg: Scene entity configuration for the robot asset.
    """
    asset = env.scene[asset_cfg.name]
    root_pos = asset.data.root_pos_w
    root_quat = asset.data.root_quat_w

    # Head world position = root_pos + rotate(root_quat, head_offset_local)
    head_offset_local = torch.tensor(head_offset, device=env.device, dtype=root_pos.dtype)
    head_pos_w = root_pos + math_utils.quat_apply(root_quat, head_offset_local)

    desired_head_z = asset.data.default_root_state[:, 2] + head_offset[2]
    cur_head_z = head_pos_w[:, 2]
    height_error = torch.square(desired_head_z - cur_head_z)
    reward = torch.exp(-height_error / std**2)

    if mask_delay:
        from robot_lab.tasks.manager_based.beyondamp.mdp.terminations import DelayedTerminationManager
        tm = env.termination_manager
        if isinstance(tm, DelayedTerminationManager):
            in_buffer = tm._delay_env_mask & (tm._delay_counters > 0)
            reward = torch.where(in_buffer, reward * delay_env_rew_ratio, torch.zeros_like(reward))

    return reward


# ---- Delay env helpers ----

def _get_delay_env_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Get mask of delay envs currently in the buffer period."""
    from robot_lab.tasks.manager_based.beyondamp.mdp.terminations import DelayedTerminationManager
    tm = env.termination_manager
    if isinstance(tm, DelayedTerminationManager):
        return tm._delay_env_mask & (tm._delay_counters > 0)
    return None


def _apply_delay_env_reward_scaling(
    env: ManagerBasedRLEnv,
    reward: torch.Tensor,
    mask_delay: bool,
    delay_env_rew_ratio: float,
) -> torch.Tensor:
    """Scale reward for delay envs by delay_env_rew_ratio.

    When mask_delay=True and ratio=0.0, delay envs get 0 reward.
    Used for velocity tracking — fallen envs can't track velocity.
    """
    if not mask_delay:
        return reward
    delay_mask = _get_delay_env_mask(env)
    if delay_mask is None:
        return reward
    return torch.where(delay_mask, reward * delay_env_rew_ratio, reward)


# ---- Velocity tracking rewards ----


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[]),
) -> torch.Tensor:
    """Track xy linear velocity command in world frame.

    Command is specified in body frame (vx, vy, 0); transformed to world
    frame using the anchor body's yaw quaternion, then compared against
    the robot's actual anchor linear velocity.
    """
    from isaaclab.utils.math import quat_apply_yaw

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # Command in body frame (vx, vy, 0) → world frame via yaw-only rotation
    cmd_xyz_b = torch.cat([command[:, :2], torch.zeros_like(command[:, :1])], dim=-1)
    anchor_quat_w = asset.data.body_quat_w[:, anchor_cfg.body_ids[0]]
    cmd_xyz_w = quat_apply_yaw(anchor_quat_w, cmd_xyz_b)

    lin_vel_error = torch.sum(
        torch.square(cmd_xyz_w - asset.data.body_lin_vel_w[:, anchor_cfg.body_ids[0]]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[]),
) -> torch.Tensor:
    """Track yaw angular velocity + suppress roll/pitch in body frame.

    Error has two components:
      1. z: command yaw rate vs actual anchor angular velocity z (world frame)
      2. xy: body-frame roll/pitch angular velocity (target = 0, suppress wobble)
    """
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    anchor_ang_vel_w = asset.data.body_ang_vel_w[:, anchor_cfg.body_ids[0]]
    ang_vel_z_error = torch.square(command[:, 2] - anchor_ang_vel_w[:, 2])

    # xy: body-frame roll/pitch angular velocity, target = 0
    anchor_ang_vel_b = quat_apply_inverse(
        asset.data.body_quat_w[:, anchor_cfg.body_ids[0]],
        anchor_ang_vel_w,
    )
    ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

    total_error = ang_vel_z_error + ang_vel_xy_error
    reward = torch.exp(-total_error / std**2)
    return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    std: float,
    mask_delay: bool = False,
    delay_env_rew_ratio: float = 1.0,
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[]),
) -> torch.Tensor:
    """Suppress roll/pitch angular velocity of a specified body (e.g. base_link)."""
    from isaaclab.utils.math import quat_apply_inverse

    asset = env.scene[body_cfg.name]
    body_ang_vel_w = asset.data.body_ang_vel_w[:, body_cfg.body_ids[0]]
    body_ang_vel_b = quat_apply_inverse(
        asset.data.body_quat_w[:, body_cfg.body_ids[0]],
        body_ang_vel_w,
    )
    ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_b[:, :2]), dim=-1)
    reward = torch.exp(-ang_vel_xy_error / std**2)
    return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def self_collisions(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float) -> torch.Tensor:
    """Penalize self-collisions indicated by contact on arm/waist bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize feet sliding when in contact with the ground."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    )
    asset: RigidObject = env.scene[asset_cfg.name]

    cur_footvel_translated = (
        asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[:, :].unsqueeze(1)
    )
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_lateral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_lateral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
