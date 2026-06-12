# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import os
import torch
from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING
import glob

from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
)

from robot_lab.tasks.manager_based.beyondamp.mdp.terminations import DelayedTerminationManager


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

Category = {
    "recovery": [],
    "walk": []
}


class CategoryView:
    """Zero-copy view into one category's frames within the parent MotionLoader.

    Delegates all data property access to the parent, automatically slicing
    tensors to only include frames belonging to this category.  Non-tensor
    attributes (e.g. ``fps``) pass through unchanged.

    Usage::

        loader = MotionLoader(".../motion/", body_indexes)
        loader.walk.joint_pos     # → only walk frames
        loader.recovery.body_pos_w  # → only recovery frames
        loader.joint_pos          # → all frames (parent directly)
    """

    def __init__(self, parent: "MotionLoader", clip_indices: torch.Tensor):
        self._p = parent
        self._clip_idx = clip_indices

        # Filtered file list — shadow parent's so __getattr__ doesn't leak all files
        self.files = [parent.files[i] for i in clip_indices.tolist()]

        # Per-category clip metadata (local to this category)
        self.time_step_total = parent.time_step_total[clip_indices]
        self.num_motions = len(clip_indices)

        if self.num_motions == 0:
            # Empty category — create zero-length placeholders
            empty = torch.zeros(0, dtype=torch.long, device=clip_indices.device)
            self.motion_starts = empty
            self.motion_starts_global = empty
            self._frame_idx = empty
            self.total_frames = 0
            return

        self.motion_starts = torch.cat([
            torch.tensor([0], device=clip_indices.device),
            torch.cumsum(self.time_step_total, dim=0)[:-1],
        ])
        self.total_frames = int(self.time_step_total.sum().item())

        # Clip starts in parent's global frame coordinates (for reset_from_motion)
        self.motion_starts_global = parent.motion_starts[clip_indices]

        # Global frame indices: positions of this category's frames in parent tensors
        frames: list[torch.Tensor] = []
        for i in clip_indices:
            s = int(parent.motion_starts[i].item())
            length = int(parent.time_step_total[i].item())
            frames.append(torch.arange(s, s + length, device=clip_indices.device))
        self._frame_idx = torch.cat(frames)  # shape: (total_frames,)

    def __getattr__(self, name: str):
        # Block access to private internals (except _body_indexes used by properties)
        if name.startswith("_") and name != "_body_indexes":
            raise AttributeError(name)
        attr = getattr(self._p, name)
        if isinstance(attr, torch.Tensor):
            return attr[self._frame_idx]
        return attr


class MotionLoader:
    """Load and concatenate motion data from .npz files.

    All body/joint data is concatenated along the time axis so that random
    global frame indices can be used to sample across all motions uniformly.
    """

    def __init__(self, motion_file: str, body_indexes: Sequence[int], device: str = "cpu"):
        if os.path.isfile(motion_file):
            files = glob.glob(motion_file)
        else:
            files = glob.glob(f"{motion_file}/**/*.npz", recursive=True)
        assert len(files) != 0, f"Invalid file path: {motion_file}"
        self._init_from_files(files, body_indexes, device)

    def _init_from_files(self, files: list[str], body_indexes: Sequence[int], device: str):
        """Shared init: load .npz files, concatenate tensors, build motion index."""
        self.files = files
        self.time_step_total = []
        self.joint_pos = []
        self.joint_vel = []
        self._body_pos_w = []
        self._body_quat_w = []
        self._body_lin_vel_w = []
        self._body_ang_vel_w = []
        self._body_pos_b = []
        self._body_quat_b = []
        self._body_ori_b = []
        self._body_lin_vel_b = []
        self._body_ang_vel_b = []

        # Local category index accumulator (don't mutate module-level Category)
        cat_indices = {name: [] for name in Category}
        fps = None

        for idx, motion_file in enumerate(files):
            category = os.path.basename(os.path.dirname(motion_file))
            assert category in Category, f"Unknown motion category '{category}' from {motion_file}"
            cat_indices[category].append(idx)

            data = np.load(motion_file)
            if fps is None:
                fps = data["fps"]
            else:
                assert data["fps"] == fps, f"FPS mismatch: expected {fps}, got {data['fps']} from {motion_file}"

            self.joint_pos.append(torch.tensor(data["joint_pos"], dtype=torch.float32, device=device))
            self.joint_vel.append(torch.tensor(data["joint_vel"], dtype=torch.float32, device=device))
            self._body_pos_w.append(torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device))
            self._body_quat_w.append(torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device))
            self._body_lin_vel_w.append(torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device))
            self._body_ang_vel_w.append(torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device))
            self._body_pos_b.append(torch.tensor(data["body_pos_b"], dtype=torch.float32, device=device))
            self._body_quat_b.append(torch.tensor(data["body_quat_b"], dtype=torch.float32, device=device))
            self._body_ori_b.append(torch.tensor(data["body_ori_b"], dtype=torch.float32, device=device))
            self._body_lin_vel_b.append(torch.tensor(data["body_lin_vel_b"], dtype=torch.float32, device=device))
            self._body_ang_vel_b.append(torch.tensor(data["body_ang_vel_b"], dtype=torch.float32, device=device))
            self.time_step_total.append(data["joint_pos"].shape[0])
        for key in cat_indices:
            cat_indices[key] = torch.tensor(cat_indices[key], dtype=torch.long, device=device)
        self.fps = fps

        self._body_indexes = body_indexes
        self.joint_pos = torch.cat(self.joint_pos, dim=0)
        self.joint_vel = torch.cat(self.joint_vel, dim=0)
        self._body_pos_w = torch.cat(self._body_pos_w, dim=0)
        self._body_quat_w = torch.cat(self._body_quat_w, dim=0)
        self._body_lin_vel_w = torch.cat(self._body_lin_vel_w, dim=0)
        self._body_ang_vel_w = torch.cat(self._body_ang_vel_w, dim=0)
        self._body_pos_b = torch.cat(self._body_pos_b, dim=0)
        self._body_quat_b = torch.cat(self._body_quat_b, dim=0)
        self._body_ori_b = torch.cat(self._body_ori_b, dim=0)
        self._body_lin_vel_b = torch.cat(self._body_lin_vel_b, dim=0)
        self._body_ang_vel_b = torch.cat(self._body_ang_vel_b, dim=0)
        self.time_step_total = torch.tensor(self.time_step_total, device=device, dtype=torch.long)
        self.total_frames = self.joint_pos.shape[0]
        self.motion_starts = torch.cat([torch.tensor([0], device=device), torch.cumsum(self.time_step_total, dim=0)[:-1]])
        self.num_motions = len(files)

        # Create zero-copy category views (e.g. self.walk, self.recovery)
        for cat_name, clip_idx in cat_indices.items():
            setattr(self, cat_name, CategoryView(self, clip_idx))

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]

    @property
    def body_pos_b(self) -> torch.Tensor:
        return self._body_pos_b[:, self._body_indexes]

    @property
    def body_quat_b(self) -> torch.Tensor:
        return self._body_quat_b[:, self._body_indexes]

    @property
    def body_lin_vel_b(self) -> torch.Tensor:
        return self._body_lin_vel_b[:, self._body_indexes]

    @property
    def body_ori_b(self) -> torch.Tensor:
        return self._body_ori_b[:, self._body_indexes]

    @property
    def body_ang_vel_b(self) -> torch.Tensor:
        return self._body_ang_vel_b[:, self._body_indexes]


class LocomotionCommand(CommandTerm):
    """Combined motion + velocity command, matching AMP_mjlab architecture.

    - Loads motion data for AMP discriminator sampling and reset-from-motion.
    - Generates body-frame velocity commands (vx, vy, omega_z) resampled periodically.
    - Robot state reset from motion data is delegated to ``reset_from_motion()``,
      called by the reset event.
    """

    cfg: LocomotionCommandCfg

    def __init__(self, cfg: LocomotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )

        self.motion = MotionLoader(self.cfg.motion_file, self.body_indexes, device=self.device)

        # Round-robin assign each env to a motion
        self.motion_ids = torch.arange(self.num_envs, device=self.device) % self.motion.num_motions
        self._motion_starts = self.motion.motion_starts[self.motion_ids]
        self._motion_lengths = self.motion.time_step_total[self.motion_ids]

        # Velocity command: (vx, vy, omega_z) in body frame
        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_yaw"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self.vel_command_b

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    # ---- Delay env mask ----

    def _get_delay_env_mask(self) -> torch.Tensor:
        """Return the delay-env boolean mask from DelayedTerminationManager.

        Returns all-False if the termination manager hasn't been wrapped yet
        (e.g. during __init__ before the first reset).
        """
        term_mgr = self._env.termination_manager
        if isinstance(term_mgr, DelayedTerminationManager):
            return term_mgr._delay_env_mask
        return torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ---- Robot state reset from motion data (called by event) ----

    def reset_from_motion(self, env_ids: Sequence[int]):
        """Reset robot state from random frames within each env's assigned motion.

        Delay envs sample from recovery motions; normal envs sample from walk
        motions.  Each env is round-robin assigned a clip within its category.
        """
        if len(env_ids) == 0:
            return

        delay_mask = self._get_delay_env_mask()

        # Per-env round-robin: delay→recovery, normal→walk
        walk_ids = torch.arange(self.num_envs, device=self.device) % self.motion.walk.num_motions
        rec_ids = torch.arange(self.num_envs, device=self.device) % self.motion.recovery.num_motions

        starts = torch.where(
            delay_mask,
            self.motion.recovery.motion_starts_global[rec_ids],
            self.motion.walk.motion_starts_global[walk_ids],
        )
        lengths = torch.where(
            delay_mask,
            self.motion.recovery.time_step_total[rec_ids],
            self.motion.walk.time_step_total[walk_ids],
        )

        s = starts[env_ids]
        l = lengths[env_ids]
        local_idx = (torch.rand(len(env_ids), device=self.device) * (l.float() - 1)).long()
        idx = s + local_idx

        anchor_idx = self.motion_anchor_body_index
        root_pos = self.motion.body_pos_w[idx, anchor_idx].clone()
        root_quat = self.motion.body_quat_w[idx, anchor_idx].clone()
        root_lin_vel = self.motion.body_lin_vel_w[idx, anchor_idx].clone()
        root_ang_vel = self.motion.body_ang_vel_w[idx, anchor_idx].clone()

        positions = self._env.scene.env_origins[env_ids].clone()
        positions[:, 2] = root_pos[:, 2]

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        positions += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_quat = quat_mul(orientations_delta, root_quat)

        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel += rand_samples[:, :3]
        root_ang_vel += rand_samples[:, 3:]

        joint_pos = self.motion.joint_pos[idx].clone()
        joint_vel = self.motion.joint_vel[idx].clone()

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
        joint_pos = torch.clip(joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1])

        self.robot.write_root_state_to_sim(
            torch.cat([positions, root_quat, root_lin_vel, root_ang_vel], dim=-1),
            env_ids=env_ids,
        )
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # ---- Velocity command resampling ----

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        r = torch.empty(len(env_ids), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.vel_command_b[env_ids, 2] = r.uniform_(*self.cfg.ranges.ang_vel_z)

        # Zero out tiny commands to avoid jitter
        self.vel_command_b[env_ids, :] *= (
            torch.norm(self.vel_command_b[env_ids, :], dim=1) > 0.1
        ).unsqueeze(1)

        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self) -> None:
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

    def _update_metrics(self) -> None:
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        self.metrics["error_vel_xy"] += (
            torch.norm(
                self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b[:, :2],
                dim=-1,
            )
            / max_command_step
        )
        self.metrics["error_vel_yaw"] += (
            torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
            / max_command_step
        )

    # ---- AMP discriminator expert data ----

    def sample_expert_transition(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample (state, next_state) pairs within each env's assigned motion."""
        max_local = (self._motion_lengths - 2).clamp(min=1)
        local_t = (torch.rand(self.num_envs, device=self.device) * max_local.float()).long()
        t = self._motion_starts + local_t
        t_next = t + 1

        expert_t_parts, expert_tp1_parts = [], []
        for term_name in self.cfg.amp_obs_terms:
            data = getattr(self.motion, term_name)
            expert_t_part = data[t]
            expert_tp1_part = data[t_next]
            if expert_t_part.dim() > 2:
                expert_t_part = expert_t_part.view(expert_t_part.shape[0], -1)
                expert_tp1_part = expert_tp1_part.view(expert_tp1_part.shape[0], -1)
            expert_t_parts.append(expert_t_part)
            expert_tp1_parts.append(expert_tp1_part)
        return torch.cat(expert_t_parts, dim=-1), torch.cat(expert_tp1_parts, dim=-1)


@configclass
class LocomotionCommandCfg(CommandTermCfg):
    """Configuration for LocomotionCommand (motion + velocity)."""

    class_type: type = LocomotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    amp_obs_terms: list[str] = MISSING

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = (-1.5, 3.0)
        lin_vel_y: tuple[float, float] = (-1.0, 1.0)
        ang_vel_z: tuple[float, float] = (-1.57, 1.57)

    ranges: Ranges = Ranges()
    rel_standing_envs: float = 0.05
