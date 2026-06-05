# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.envs.mdp.actions.actions_cfg import JointActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class MotionDeltaJointPositionAction(JointAction):
    """Joint action that applies network output as a delta on top of reference motion joint positions.

    final_target = motion_reference_joint_pos + network_output * scale
    """

    cfg: MotionDeltaJointPositionActionCfg

    def __init__(self, cfg: MotionDeltaJointPositionActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._offset = 0.0

    def apply_actions(self):
        command = self._env.command_manager.get_term(self.cfg.command_name)
        ref_joint_pos = command.joint_pos
        targets = ref_joint_pos[:, self._joint_ids] + self.processed_actions
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)


@configclass
class MotionDeltaJointPositionActionCfg(JointActionCfg):
    """Configuration for motion delta joint position action."""

    class_type: type[MotionDeltaJointPositionAction] = MotionDeltaJointPositionAction

    command_name: str = "motion"
    """Name of the motion command to read reference joint positions from."""
