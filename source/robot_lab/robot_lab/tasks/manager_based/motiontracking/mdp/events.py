# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

from isaaclab.assets import Articulation
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv

from robot_lab.tasks.manager_based.motiontracking.mdp.terminations import DelayedTerminationManager


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def install_delayed_termination(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    delay_env_ratio: float = 0.0,
    max_delay_steps: int = 0,
) -> None:
    """Startup event: install DelayedTerminationManager for a fraction of envs.

    Delay envs get ``max_delay_steps`` extra frames after a termination fires
    before the episode actually resets, giving the robot time to recover
    (e.g. get up after a fall).

    Args:
        env: The environment.
        env_ids: Unused (required by event interface).
        delay_env_ratio: Fraction of envs (0.0–1.0) to mark as delay envs.
        max_delay_steps: Number of steps before a suppressed termination is
            released. Set to 0 to disable the delay.
    """
    num_delay = int(env.num_envs * delay_env_ratio)
    if num_delay <= 0 or max_delay_steps <= 0:
        return

    delay_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    delay_indices = torch.randperm(env.num_envs, device=env.device)[:num_delay]
    delay_mask[delay_indices] = True

    env.termination_manager = DelayedTerminationManager(
        base=env.termination_manager,
        delay_env_mask=delay_mask,
        max_delay_steps=max_delay_steps,
    )
    print(
        f"[install_delayed_termination] {num_delay}/{env.num_envs} envs, "
        f"max_delay_steps={max_delay_steps}"
    )
