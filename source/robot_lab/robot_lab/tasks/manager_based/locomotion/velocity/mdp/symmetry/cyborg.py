# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Left-right symmetry transforms for the Cyborg humanoid."""

from __future__ import annotations

import torch
from tensordict import TensorDict


_HISTORY_LENGTH = 15

# Policy/action joint order:
# hip roll L/R, hip yaw L/R, hip pitch L/R, knee L/R,
# ankle pitch L/R, ankle roll L/R.
_JOINT_PERMUTATION = (1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10)
_JOINT_SIGNS = (-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 1)


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
    permutation = torch.tensor(_JOINT_PERMUTATION, device=values.device)
    signs = values.new_tensor(_JOINT_SIGNS)
    return values[..., permutation] * signs


def _mirror_policy_observations(observations: torch.Tensor) -> torch.Tensor:
    """Mirror the 705-D, term-major Cyborg policy observation."""
    mirrored = observations.clone()

    phase = mirrored[..., 0:30].reshape(-1, _HISTORY_LENGTH, 2)
    angular_velocity = mirrored[..., 30:75].reshape(-1, _HISTORY_LENGTH, 3)
    projected_gravity = mirrored[..., 75:120].reshape(-1, _HISTORY_LENGTH, 3)
    velocity_command = mirrored[..., 120:165].reshape(-1, _HISTORY_LENGTH, 3)
    joint_position = mirrored[..., 165:345].reshape(-1, _HISTORY_LENGTH, 12)
    joint_velocity = mirrored[..., 345:525].reshape(-1, _HISTORY_LENGTH, 12)
    previous_action = mirrored[..., 525:705].reshape(-1, _HISTORY_LENGTH, 12)

    # Swapping the left and right legs shifts the gait phase by half a cycle.
    phase *= -1.0
    angular_velocity *= angular_velocity.new_tensor((-1.0, 1.0, -1.0))
    projected_gravity *= projected_gravity.new_tensor((1.0, -1.0, 1.0))
    velocity_command *= velocity_command.new_tensor((1.0, -1.0, -1.0))
    joint_position[:] = _mirror_joints(joint_position)
    joint_velocity[:] = _mirror_joints(joint_velocity)
    previous_action[:] = _mirror_joints(previous_action)

    return mirrored


@torch.no_grad()
def compute_symmetric_states(
    env,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Return the original and left-right mirrored policy observations/actions."""
    if obs is not None:
        batch_size = obs.batch_size[0]
        observations_augmented = obs.repeat(2)
        observations_augmented["policy"][:batch_size] = obs["policy"]
        observations_augmented["policy"][batch_size:] = _mirror_policy_observations(obs["policy"])
    else:
        observations_augmented = None

    if actions is not None:
        batch_size = actions.shape[0]
        actions_augmented = actions.new_empty((batch_size * 2, actions.shape[1]))
        actions_augmented[:batch_size] = actions
        actions_augmented[batch_size:] = _mirror_joints(actions)
    else:
        actions_augmented = None

    return observations_augmented, actions_augmented
