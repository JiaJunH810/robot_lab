# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

from robot_lab.tasks.manager_based.stubborn.mdp.commands import MotionCommand
from robot_lab.tasks.manager_based.stubborn.mdp.rewards import _get_body_indexes


def bad_anchor_pos(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.norm(command.anchor_pos_w - command.robot_anchor_pos_w, dim=1) > threshold


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def probabilistic_bad_anchor_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, p_term: float = 0.005
) -> torch.Tensor:
    """Bernoulli-based probabilistic termination for anchor height error.

    Instead of terminating immediately when the anchor height error exceeds the
    threshold, terminate with probability p_term at each step.  This gives the
    policy an expected E[T] = 1/p_term steps of recovery window.

    Args:
        env: The environment.
        command_name: The name of the motion command.
        threshold: Height error threshold (m).  Paper default: 0.25.
        p_term: Per-step termination probability.  Paper default: 0.005.

    Returns:
        A boolean tensor indicating which environments should terminate.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    exceeded = torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold
    return exceeded & (torch.rand_like(exceeded.float()) < p_term)


def probabilistic_bad_anchor_ori(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    threshold: float,
    p_term: float = 0.005,
) -> torch.Tensor:
    """Bernoulli-based probabilistic termination for anchor orientation error.

    Instead of terminating immediately when the anchor orientation error exceeds
    the threshold, terminate with probability p_term at each step.

    Args:
        env: The environment.
        asset_cfg: The asset configuration for gravity vector.
        command_name: The name of the motion command.
        threshold: Gravity-projected orientation error threshold.  Paper default: 1.57 (π/2).
        p_term: Per-step termination probability.  Paper default: 0.005.

    Returns:
        A boolean tensor indicating which environments should terminate.
    """
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    exceeded = (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold
    return exceeded & (torch.rand_like(exceeded.float()) < p_term)


def bad_motion_body_pos(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.norm(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_yaw[:, body_indexes], dim=-1)
    return torch.any(error > threshold, dim=-1)


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_yaw[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)
