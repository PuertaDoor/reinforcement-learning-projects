import sys
import os
import copy
import numpy as np

import torch
import torch.nn as nn
import hydra
import optuna

from omegaconf import DictConfig
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.utils.chrono import Chrono

from bbrl_algos.models.loggers import Logger

from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.plotters import Plotter
from bbrl_algos.models.exploration_agents import AddGaussianNoise, AddOUNoise
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


# Create the TD3 Agent
def create_td3_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()

    # Create two Q-networks and their target counterparts
    critic1 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.q1,
    ).set_name("critic1")
    target_critic1 = copy.deepcopy(critic1).set_name("target-critic1")
    
    critic2 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.q2,
    ).set_name("critic2")
    target_critic2 = copy.deepcopy(critic2).set_name("target-critic2")

    actor = ContinuousDeterministicActor(
        obs_size,
        cfg.algorithm.architecture.actor_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.act,
    )
    target_actor = copy.deepcopy(actor).set_name("target-actor")
    
    if cfg.algorithm.noise == 'AddGaussianNoise':
        noise_agent = AddGaussianNoise(
            cfg.algorithm.action_noise,
            seed=cfg.algorithm.seed.explorer,
        )
    else:
        noise_agent = AddOUNoise(
            cfg.algorithm.ou_std,
            seed=cfg.algorithm.seed.explorer,
        )
    tr_agent = Agents(train_env_agent, actor, noise_agent)
    ev_agent = Agents(eval_env_agent, actor)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    
    return train_agent, eval_agent, actor, critic1, critic2, target_critic1, target_critic2, target_actor


# Configure the optimizer
def setup_optimizers(cfg, actor, critic1, critic2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    actor_optimizer = get_class(cfg.actor_optimizer)(actor.parameters(), **actor_optimizer_args)

    critic1_optimizer_args = get_arguments(cfg.critic_optimizer)
    critic1_optimizer = get_class(cfg.critic_optimizer)(critic1.parameters(), **critic1_optimizer_args)

    critic2_optimizer_args = get_arguments(cfg.critic_optimizer)
    critic2_optimizer = get_class(cfg.critic_optimizer)(critic2.parameters(), **critic2_optimizer_args)

    return actor_optimizer, critic1_optimizer, critic2_optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values1, q_values2, target_q_values):
    # Compute temporal difference
    q_next = target_q_values
    target = (
        reward[:-1].squeeze()
        + cfg.algorithm.discount_factor * q_next.squeeze(-1) * must_bootstrap.int()
    )
    mse = nn.MSELoss()
    critic_loss = mse(target, q_values1.squeeze(-1)) + mse(target, q_values2.squeeze(-1))
    return critic_loss

def compute_actor_loss(q_values):
    return -q_values.mean()


def run_td3(cfg, logger, trial=None):
    best_reward = float("-inf")

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the TD3 Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic1,
        critic2,
        target_critic1,
        target_critic2,
        target_actor
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent1 = TemporalAgent(critic1)
    q_agent2 = TemporalAgent(critic2)
    target_q_agent1 = TemporalAgent(target_critic1)
    target_q_agent2 = TemporalAgent(target_critic2)

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic1_optimizer, critic2_optimizer = setup_optimizers(cfg, actor, critic1, critic2)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions(filter_key="env/done")
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        for _ in range(cfg.algorithm.optim_n_updates):
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action"
            ]
            if nb_steps > cfg.algorithm.learning_starts:
                # Determines whether values of the critic should be propagated
                # True if the episode reached a time limit or if the task was not done
                # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
                must_bootstrap = torch.logical_or(~done[1], truncated[1])

                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                # the detach_actions=True changes nothing in the results
                q_agent1(rb_workspace, t=0, n_steps=1, detach_actions=True)
                q1_values = rb_workspace["critic1/q_values"]

                q_agent2(rb_workspace, t=0, n_steps=1, detach_actions=True)
                q2_values = rb_workspace["critic2/q_values"]

                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    ag_actor(rb_workspace, t=1, n_steps=1)

                    # compute Q_values from both target critics: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    target_q_agent1(rb_workspace, t=1, n_steps=1, detach_actions=True)
                    target_q_values1 = rb_workspace["target-critic1/q_values"]
                    target_q_agent2(rb_workspace, t=1, n_steps=1, detach_actions=True)
                    target_q_values2 = rb_workspace["target-critic2/q_values"]
                    # q_agent(rb_workspace, t=1, n_steps=1)
                # Use the minimum of the two target Q-values
                post_q_values = torch.min(target_q_values1, target_q_values2)

                # Compute critic loss
                critic_loss = compute_critic_loss(
                    cfg, reward, must_bootstrap, q1_values[0], q2_values[0], post_q_values[1]
                )
                logger.add_log("critic_loss", critic_loss, nb_steps)
                critic1_optimizer.zero_grad()
                critic2_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic1.parameters(), cfg.algorithm.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    critic2.parameters(), cfg.algorithm.max_grad_norm
                )
                critic1_optimizer.step()
                critic2_optimizer.step()

                # Delayed policy updates
                if nb_steps % cfg.algorithm.policy_delay == 0:
                    # Actor update
                    # Now we determine the actions the current policy would take in the states from the RB
                    ag_actor(rb_workspace, t=0, n_steps=1)
                    # We determine the Q values resulting from actions of the current policy
                    q_agent1(rb_workspace, t=0, n_steps=1)
                    # and we back-propagate the corresponding loss to maximize the Q values
                    q1_values = rb_workspace["critic1/q_values"]  # Use one of the Q-values for actor loss
                    actor_loss = compute_actor_loss(q1_values)
                    logger.add_log("actor_loss", actor_loss, nb_steps)
                    # if -25 < actor_loss < 0 and nb_steps > 2e5:
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        actor.parameters(), cfg.algorithm.max_grad_norm
                    )
                    actor_optimizer.step()
                    
                    # Soft update of target networks
                    soft_update_params(critic1, target_critic1, cfg.algorithm.tau)
                    soft_update_params(critic2, target_critic2, cfg.algorithm.tau)
                    soft_update_params(actor, target_actor, cfg.algorithm.tau)


        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")

            rewards = eval_workspace["env/cumulated_reward"][-1]

            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)

            if mean > best_reward:
                best_reward = mean

            print(f"nb_steps: {nb_steps}, reward: {mean:.0f}, best: {best_reward:.0f}")

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    mean,
                    "./td3_best_agents/",
                    "td3",
                )
                if cfg.plot_agents:
                    plot_policy(
                        eval_agent.agent.agents[1],
                        eval_env_agent,
                        best_reward,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                        stochastic=False,
                    )
                    plot_critic(
                        critic1,
                        eval_env_agent,
                        best_reward,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                    )
                    plot_critic(
                        critic2,
                        eval_env_agent,
                        best_reward,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                    )

    return best_reward


# %%
@hydra.main(
    config_path="./",
    # config_name="td3_pendulum.yaml",
    config_name="td3_pendulum_optuna.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_td3)
    else:
        logger = Logger(cfg_raw)
        run_td3(cfg_raw, logger)


if __name__ == "__main__":
    main()
