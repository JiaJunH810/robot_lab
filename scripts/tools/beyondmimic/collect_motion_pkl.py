# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Run a trained policy to track a motion sequence from frame 0 and record the result to .pkl.

.. code-block:: bash

    python scripts/tools/beyondmimic/collect_motion_pkl.py \
        --task RobotLab-Isaac-BeyondMimic-Flat-Cyborg \
        --checkpoint /path/to/model.pt \
        --output /path/to/output.pkl
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Collect motion data using a trained policy.")
parser.add_argument("--task", type=str, required=True, help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments.")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to the policy checkpoint (.pt).")
parser.add_argument("--output", type=str, default=None,
                    help="Output .pkl path. Default: <checkpoint_dir>/<motion>_collect.pkl")
parser.add_argument("--seed", type=int, default=42, help="Random seed.")

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import importlib.metadata as metadata
import joblib
import numpy as np
import time
import torch

import gymnasium as gym

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, yaw_quat

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

import robot_lab.tasks  # noqa: F401


def _write_frame_state(motion_cmd, robot, env_origins, frame_idx: int):
    """Write the motion frame at *frame_idx* into the simulator for env 0."""
    device = motion_cmd.device
    env_origin = env_origins[0:1]

    # Root state
    root_pos = motion_cmd.motion.body_pos_w[frame_idx, motion_cmd.motion_anchor_body_index].clone().to(device)
    root_quat = motion_cmd.motion.body_quat_w[frame_idx, motion_cmd.motion_anchor_body_index].clone().to(device)
    root_lin_vel = motion_cmd.motion.body_lin_vel_w[frame_idx, motion_cmd.motion_anchor_body_index].clone().to(device)
    root_ang_vel = motion_cmd.motion.body_ang_vel_w[frame_idx, motion_cmd.motion_anchor_body_index].clone().to(device)
    root_states = robot.data.default_root_state.clone()
    root_states[0, :2] = root_pos[:2] + env_origin[0, :2]
    root_states[0, 2] = root_pos[2]
    root_states[0, 3:7] = root_quat
    root_states[0, 7:10] = root_lin_vel
    root_states[0, 10:] = root_ang_vel
    robot.write_root_state_to_sim(root_states)

    # Joint state
    joint_pos = motion_cmd.motion.joint_pos[frame_idx:frame_idx + 1].clone().to(device)
    jvel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, jvel)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    # Convert deprecated policy cfg to actor/critic (same as play.py / train.py)
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    agent_cfg.seed = args_cli.seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else "cuda:0"

    # Disable randomizations
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.events.randomize_rigid_body_material = None
    env_cfg.events.randomize_push_robot = None
    env_cfg.scene.terrain.max_init_terrain_level = None
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # Disable early terminations — only time_out survives
    for key in list(env_cfg.terminations.__dict__.keys()):
        if key not in ("time_out",) and not key.startswith("_"):
            setattr(env_cfg.terminations, key, None)

    # ---------- create env ----------
    env = gym.make(args_cli.task, cfg=env_cfg)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # Grab motion command handle BEFORE wrapper (wrapper's __init__ calls reset)
    motion_cmd = env.unwrapped.command_manager.get_term("motion")
    total_frames = int(motion_cmd.motion.time_step_total)
    robot = env.unwrapped.scene["robot"]

    # Monkey-patch: no-op during wrapper reset (robot state will be set explicitly below)
    motion_cmd._resample_command = lambda env_ids: None

    # Replace _update_command with linear version that never resamples at sequence end
    def _linear_update():
        motion_cmd.time_steps += 1
        # same relative-body-transform computation as the original _update_command
        n = len(motion_cmd.cfg.body_names)
        ap = motion_cmd.anchor_pos_w[:, None, :].repeat(1, n, 1)
        aq = motion_cmd.anchor_quat_w[:, None, :].repeat(1, n, 1)
        rp = motion_cmd.robot_anchor_pos_w[:, None, :].repeat(1, n, 1)
        rq = motion_cmd.robot_anchor_quat_w[:, None, :].repeat(1, n, 1)
        dp = rp
        dp[..., 2] = ap[..., 2]
        dq = yaw_quat(quat_mul(rq, quat_inv(aq)))
        motion_cmd.body_quat_relative_w = quat_mul(dq, motion_cmd.body_quat_w)
        motion_cmd.body_pos_relative_w = dp + quat_apply(dq, motion_cmd.body_pos_w - ap)

    motion_cmd._update_command = _linear_update

    # Create RSL-RL wrapper (calls reset → our monkey-patches kick in)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Set time_steps to 0 and write frame-0 state explicitly
    motion_cmd.time_steps[:] = 0
    _write_frame_state(motion_cmd, robot, env.unwrapped.scene.env_origins, 0)

    # ---------- load checkpoint ----------
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir

    print(f"[INFO] Loading checkpoint: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_cfg={"actor": True, "critic": False, "optimizer": False})
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # ---------- data buffers ----------
    joint_idx = robot.find_joints([
        "J_hip_l_roll", "J_hip_l_yaw", "J_hip_l_pitch", "J_knee_l_pitch",
        "J_ankle_l_pitch", "J_ankle_l_roll",
        "J_hip_r_roll", "J_hip_r_yaw", "J_hip_r_pitch", "J_knee_r_pitch",
        "J_ankle_r_pitch", "J_ankle_r_roll",
        "J_waist_yaw", "J_waist_pitch",
        "J_arm_l_01", "J_arm_l_02", "J_arm_l_03", "J_arm_l_04",
        "J_arm_l_05", "J_arm_l_06", "J_arm_l_07",
        "J_arm_r_01", "J_arm_r_02", "J_arm_r_03", "J_arm_r_04",
        "J_arm_r_05", "J_arm_r_06", "J_arm_r_07",
    ], preserve_order=True)[0]
    root_body_idx = robot.body_names.index("base_link")

    log = {
        "fps": int(1.0 / (env_cfg.sim.dt * env_cfg.decimation)),
        "dof_pos": [],
        "root_pos": [],
        "root_rot": [],
    }

    def _record():
        log["dof_pos"].append(robot.data.joint_pos[0, joint_idx].cpu().numpy().copy())
        log["root_pos"].append(robot.data.body_pos_w[0, root_body_idx].cpu().numpy().copy())
        q = robot.data.body_quat_w[0, root_body_idx].cpu().numpy().copy()
        log["root_rot"].append(q[[1, 2, 3, 0]])  # wxyz → xyzw

    # Determine output path
    if args_cli.output:
        output_path = args_cli.output
    else:
        motion_basename = os.path.splitext(os.path.basename(env_cfg.commands.motion.motion_file))[0]
        output_path = os.path.join(os.path.dirname(resume_path), f"{motion_basename}_collect.pkl")

    dt = env.unwrapped.step_dt
    obs = env.get_observations()

    print(f"[INFO] Collecting {total_frames} frames with policy ...")

    # Record frame-0 initial state
    _record()

    frame_count = 1
    while simulation_app.is_running():
        # Stop when next _update_command would index out of bounds
        if motion_cmd.time_steps[0].item() >= total_frames - 1:
            break

        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        _record()
        frame_count += 1
        if frame_count % 100 == 0:
            print(f"  Frame {frame_count}/{total_frames}")

        sleep_time = dt - (time.time() - start_time)
        if sleep_time > 0:
            time.sleep(sleep_time)

    # ---------- save ----------
    for k in ["dof_pos", "root_pos", "root_rot"]:
        log[k] = np.stack(log[k], axis=0)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    joblib.dump(log, output_path)
    print(f"[INFO] Saved to {output_path}")
    print(f"  dof_pos:   {log['dof_pos'].shape}")
    print(f"  root_pos:  {log['root_pos'].shape}")
    print(f"  root_rot:  {log['root_rot'].shape}")
    print(f"  fps:       {log['fps']}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
