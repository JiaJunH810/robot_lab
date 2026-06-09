# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch

from isaaclab.managers.termination_manager import TerminationManager


class DelayedTerminationManager(TerminationManager):
    def __init__(
        self,
        base: TerminationManager,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
    ) -> None:
        # 接管原始终止管理器的所有内部状态（_terminated_buf, _truncated_buf 等），
        # 不重新初始化，避免丢失已有状态
        self.__dict__.update(base.__dict__)

        self._delay_env_mask = delay_env_mask          # (num_envs,) bool 张量，标记哪些 env 是 delay env
        self._delay_counters = torch.zeros_like(       # 每个 env 的延迟计数器
            delay_env_mask, dtype=torch.long
        )
        self._max_delay_steps = max_delay_steps        # 延迟步数上限（250 步 ≈ 5 秒）

    def compute(self) -> torch.Tensor:
        # 1. 先执行原始终止逻辑，填充 self._truncated_buf 和 self._terminated_buf
        dones = super().compute()

        if self._max_delay_steps <= 0:
            return dones  # 未启用延迟，直接返回原始结果

        # 2. 对于 delay env 中刚触发了 termination 的，计数器 +1
        delay_and_done = self._delay_env_mask & dones
        self._delay_counters[delay_and_done] += 1

        # 3. 计数器未到期 → 拦截 reset 信号
        #    机器人仍然继续仿真，有机会自己恢复（例如从倒地中站起来）
        not_ready = delay_and_done & (self._delay_counters < self._max_delay_steps)
        self._terminated_buf[not_ready] = False

        # 4. 计数器到期 → 释放 reset 信号，清零计数器
        #    缓冲期耗尽还没恢复，真正终止并 reset
        ready = delay_and_done & (self._delay_counters >= self._max_delay_steps)
        self._delay_counters[ready] = 0

        # 5. delay env 中当前没有触发 done 的 → 说明在缓冲期内自行恢复了
        #    清零计数器，下次摔倒可以重新获得完整的缓冲时间
        self._delay_counters[self._delay_env_mask & ~dones] = 0

        return self._truncated_buf | self._terminated_buf
