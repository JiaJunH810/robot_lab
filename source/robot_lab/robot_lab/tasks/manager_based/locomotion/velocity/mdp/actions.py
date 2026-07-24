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
    """Joint position action with configurable per-environment delay.

    Policy outputs are buffered in a FIFO.  On each ``process_actions()`` call
    the oldest buffered action is popped, the incoming action is pushed, and
    the popped action is applied to the simulation.

    With ``delay_steps=(1, 2)`` the policy is exposed to a mix of 1-step and
    2-step latencies, making it robust to real-world jitter.
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env):
        super().__init__(cfg, env)
        # Normalize config to (min, max) + per-env tensor
        if isinstance(cfg.delay_steps, (list, tuple)):
            self._min_delay, self._max_delay = int(cfg.delay_steps[0]), int(cfg.delay_steps[1])
            self._delay_per_env = torch.randint(
                self._min_delay, self._max_delay + 1, (self.num_envs,), device=self.device
            )
        else:
            self._min_delay = self._max_delay = int(cfg.delay_steps)
            self._delay_per_env = None  # fixed delay for all envs

        if self._min_delay < 0 or self._max_delay < self._min_delay:
            raise ValueError(
                f"Invalid delay_steps={cfg.delay_steps!r}. Expected a non-negative integer or an ordered "
                "(min, max) pair."
            )

        if self._max_delay > 0:
            # FIFO: [oldest, ..., newest], shape (max_delay, num_envs, action_dim)
            self._action_buffer = torch.zeros(
                self._max_delay, self.num_envs, self.action_dim, device=self.device
            )

    def process_actions(self, actions: torch.Tensor):
        if self._max_delay == 0:
            super().process_actions(actions)
            return

        # --- select delayed action per environment ---
        if self._delay_per_env is not None:
            delayed = torch.zeros_like(actions)
            for d in range(self._min_delay, self._max_delay + 1):
                mask = (self._delay_per_env == d)
                if mask.any():
                    if d == 0:
                        delayed[mask] = actions[mask]
                    else:
                        # buffer[max_delay - d] → action from d steps ago
                        delayed[mask] = self._action_buffer[self._max_delay - d, mask]
        else:
            # fixed delay: pop the oldest entry (same as before)
            d = self._max_delay
            delayed = self._action_buffer[self._max_delay - d].clone()

        # --- FIFO shift: drop oldest, append new ---
        if self._max_delay > 1:
            self._action_buffer[:-1] = self._action_buffer[1:].clone()
        self._action_buffer[-1] = actions

        # apply delayed action (scale / offset / clip via parent)
        super().process_actions(delayed)

    def reset(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        if self._max_delay > 0:
            self._action_buffer[:, env_ids] = 0.0
            if self._delay_per_env is not None:
                n = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
                self._delay_per_env[env_ids] = torch.randint(
                    self._min_delay, self._max_delay + 1, (n,), device=self.device
                )
        super().reset(env_ids)


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed joint position actions.

    ``delay_steps`` can be a fixed integer or an inclusive ``(min, max)``
    range sampled independently for each environment on reset.
    """

    class_type: type[ActionTerm] = DelayedJointPositionAction

    delay_steps: int | tuple[int, int] = 0
    """Fixed delay or inclusive per-environment delay range, in policy steps."""
