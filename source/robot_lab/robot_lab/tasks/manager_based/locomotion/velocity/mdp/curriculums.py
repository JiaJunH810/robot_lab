# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

        # Buffer to accumulate completed episode rewards across evaluation windows
        env._curriculum_lin_vel_buffer = []

    # ----- DEBUG -----
    n_ids = len(env_ids) if isinstance(env_ids, torch.Tensor) else 0
    step = env.common_step_counter
    mod = step % env.max_episode_length
    hit = (mod == 0)
    # Print every call within +/- 3 steps of eval boundaries, plus periodic heartbeat
    near_boundary = (mod <= 3 or mod >= env.max_episode_length - 3)
    if hit or near_boundary:
        marker = "*** EVAL ***" if hit else "~near"
        print(f"[CURRICULUM lin_vel TRACE] step={step} mod={mod} n_envs={n_ids} "
              f"buf={len(env._curriculum_lin_vel_buffer)} {marker}")
    elif step % 500 == 0:
        print(f"[CURRICULUM lin_vel STAT] step={step} buf={len(env._curriculum_lin_vel_buffer)} "
              f"range={base_velocity_ranges.lin_vel_x}")
    # ----- END DEBUG -----

    # Collect completed episode rewards whenever envs reset (regardless of global step alignment)
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
        for env_id in env_ids:
            env._curriculum_lin_vel_buffer.append(
                episode_sums[env_id].item() / env.max_episode_length_s
            )

    # Evaluate at regular intervals, using accumulated rewards from all completed episodes
    if hit:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        if len(env._curriculum_lin_vel_buffer) > 0:
            mean_reward = sum(env._curriculum_lin_vel_buffer) / len(env._curriculum_lin_vel_buffer)
        else:
            # Fallback: use env_ids directly (handles step=0 case where buffer is empty
            # but all envs are resetting simultaneously)
            mean_reward = torch.mean(episode_sums[env_ids]).item() / env.max_episode_length_s

        threshold = 0.8 * reward_term_cfg.weight
        n_envs = len(env._curriculum_lin_vel_buffer)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if mean_reward > threshold:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()
            print(f"[CURRICULUM lin_vel] step={step} UPGRADE: "
                  f"mean_reward={mean_reward:.4f} threshold={threshold:.4f} "
                  f"n_envs={n_envs} new_range={base_velocity_ranges.lin_vel_x}")
        else:
            print(f"[CURRICULUM lin_vel] step={step} NO UPGRADE: "
                  f"mean_reward={mean_reward:.4f} threshold={threshold:.4f} "
                  f"n_envs={n_envs} range={base_velocity_ranges.lin_vel_x}")

        # Clear buffer for next evaluation window
        env._curriculum_lin_vel_buffer = []

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

        # Buffer to accumulate completed episode rewards across evaluation windows
        env._curriculum_ang_vel_buffer = []

    # ----- DEBUG -----
    n_ids = len(env_ids) if isinstance(env_ids, torch.Tensor) else 0
    step = env.common_step_counter
    mod = step % env.max_episode_length
    hit = (mod == 0)
    # Print every call within +/- 3 steps of eval boundaries, plus periodic heartbeat
    near_boundary = (mod <= 3 or mod >= env.max_episode_length - 3)
    if hit or near_boundary:
        marker = "*** EVAL ***" if hit else "~near"
        print(f"[CURRICULUM ang_vel TRACE] step={step} mod={mod} n_envs={n_ids} "
              f"buf={len(env._curriculum_ang_vel_buffer)} {marker}")
    elif step % 500 == 0:
        print(f"[CURRICULUM ang_vel STAT] step={step} buf={len(env._curriculum_ang_vel_buffer)} "
              f"range={base_velocity_ranges.ang_vel_z}")
    # ----- END DEBUG -----

    # Collect completed episode rewards whenever envs reset (regardless of global step alignment)
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    if isinstance(env_ids, torch.Tensor) and len(env_ids) > 0:
        for env_id in env_ids:
            env._curriculum_ang_vel_buffer.append(
                episode_sums[env_id].item() / env.max_episode_length_s
            )

    # Evaluate at regular intervals, using accumulated rewards from all completed episodes
    if hit:
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        if len(env._curriculum_ang_vel_buffer) > 0:
            mean_reward = sum(env._curriculum_ang_vel_buffer) / len(env._curriculum_ang_vel_buffer)
        else:
            # Fallback: use env_ids directly (handles step=0 case)
            mean_reward = torch.mean(episode_sums[env_ids]).item() / env.max_episode_length_s

        threshold = 0.8 * reward_term_cfg.weight
        n_envs = len(env._curriculum_ang_vel_buffer)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if mean_reward > threshold:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()
            print(f"[CURRICULUM ang_vel] step={step} UPGRADE: "
                  f"mean_reward={mean_reward:.4f} threshold={threshold:.4f} "
                  f"n_envs={n_envs} new_range={base_velocity_ranges.ang_vel_z}")
        else:
            print(f"[CURRICULUM ang_vel] step={step} NO UPGRADE: "
                  f"mean_reward={mean_reward:.4f} threshold={threshold:.4f} "
                  f"n_envs={n_envs} range={base_velocity_ranges.ang_vel_z}")

        # Clear buffer for next evaluation window
        env._curriculum_ang_vel_buffer = []

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)
