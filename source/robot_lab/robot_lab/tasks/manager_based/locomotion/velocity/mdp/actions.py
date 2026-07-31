# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Sequence

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ActionTerm
from isaaclab.utils import configclass


class DelayedJointPositionAction(JointPositionAction):
    """Joint position action with per-environment delay in physics steps."""

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)

        if isinstance(cfg.delay_steps, (list, tuple)):
            self._min_delay, self._max_delay = int(cfg.delay_steps[0]), int(cfg.delay_steps[1])
        else:
            self._min_delay = self._max_delay = int(cfg.delay_steps)

        if self._min_delay < 0 or self._max_delay < self._min_delay:
            raise ValueError(
                f"Invalid delay_steps={cfg.delay_steps!r}. Expected a non-negative integer or an ordered "
                "(min, max) pair."
            )

        self._env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        self._delay_per_env = torch.randint(
            self._min_delay, self._max_delay + 1, (self.num_envs,), device=self.device
        )

        # Index zero is the newest raw action; index d is d physics steps old.
        self._action_buffer = torch.zeros(
            self._max_delay + 1, self.num_envs, self.action_dim, device=self.device
        )

    def process_actions(self, actions: torch.Tensor):
        # Called once per policy step. The latest raw action is held between calls.
        super().process_actions(actions)

    def apply_actions(self):
        # Called once per physics step, so one buffer index equals sim.dt.
        if self._max_delay > 0:
            self._action_buffer[1:] = self._action_buffer[:-1].clone()
        self._action_buffer[0] = self._raw_actions

        delayed_raw_actions = self._action_buffer[self._delay_per_env, self._env_ids]
        delayed_targets = delayed_raw_actions * self._scale + self._offset
        if self.cfg.clip is not None:
            delayed_targets = torch.clamp(
                delayed_targets, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._asset.set_joint_position_target(delayed_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = self._env_ids
        elif isinstance(env_ids, slice):
            env_ids = self._env_ids[env_ids]
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        self._action_buffer[:, env_ids] = 0.0
        self._delay_per_env[env_ids] = torch.randint(
            self._min_delay, self._max_delay + 1, (len(env_ids),), device=self.device
        )
        super().reset(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed joint position actions.

    ``delay_steps`` can be a fixed integer or an inclusive ``(min, max)``
    range in physics steps, sampled independently for each environment on reset.
    """

    class_type: type[ActionTerm] = DelayedJointPositionAction

    delay_steps: int | tuple[int, int] = 0
    """Fixed delay or inclusive per-environment delay range, in physics steps."""
