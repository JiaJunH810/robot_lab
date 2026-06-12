# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from robot_lab.tasks.manager_based.beyondamp.rsl_rl_amp.amp_on_policy_runner import AMPOnPolicyRunner
from robot_lab.tasks.manager_based.beyondamp.rsl_rl_amp.amp_vecenv_wrapper import AMPRslRlVecEnvWrapper

@configclass
class CyborgBeyondAMPFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "cyborg_beyondamp"
    runner_type = AMPOnPolicyRunner
    wrapper_type = AMPRslRlVecEnvWrapper

    amp_discr_hidden_dims = [1024, 512, 256]
    amp_reward_coef = 0.1
    amp_task_reward_lerp = 0.75
    amp_min_normalized_std = 0.05
    amp_replay_buffer_size = 100000

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name = "rsl_rl_amp.amp_ppo:AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
