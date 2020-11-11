import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from agent import Agent
import utils

import hydra


class QuantileAgent(Agent):
    """Quantile algorithm."""
    def __init__(self, obs_dim, action_dim, action_range, device, critic_cfg,
                 actor_cfg, discount, init_temperature, alpha_lr, alpha_betas,
                 actor_lr, actor_betas, actor_update_frequency, critic_lr,
                 critic_betas, critic_tau, critic_target_update_frequency,
                 batch_size, learnable_temperature, quantile, n_samples,
                 expectation, entropy_reg, tune_entropy, all_actions):
        super().__init__()

        self.action_range = action_range
        self.device = torch.device(device)
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.batch_size = batch_size
        self.learnable_temperature = learnable_temperature
        self.quantile = quantile
        self.expectation = expectation
        self.entropy_reg = entropy_reg
        self.tune_entropy = tune_entropy
        self.all_actions = all_actions
        self.n_samples = n_samples

        self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
        self.critic_target = hydra.utils.instantiate(critic_cfg).to(
            self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -action_dim

        # optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr,
                                                betas=actor_betas)

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr,
                                                 betas=critic_betas)

        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr,
                                                    betas=alpha_betas)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def act(self, obs, sample=False, propensity=False):
        obs = torch.FloatTensor(obs).to(self.device)
        obs = obs.unsqueeze(0)
        dist = self.actor(obs)
        action = dist.sample() if sample else dist.mean
        prob = dist.log_prob(action).sum(dim=-1, keepdim=True).exp()
        action = action.clamp(*self.action_range)
        assert action.ndim == 2 and action.shape[0] == 1
        if propensity:
            return utils.to_np(action[0]), utils.to_np(prob[0])
        else:
            return utils.to_np(action[0])

    def update_critic(self, obs, action, reward, next_obs, not_done, logger,
                      step):
        dist = self.actor(next_obs)
        next_action = dist.sample()
        log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
        target_Q1, target_Q2 = self.critic_target(next_obs, next_action)

        # remove this to convert from soft-Q to TD3
        target_V = torch.min(target_Q1,
                             target_Q2) #- self.alpha.detach() * log_prob
        target_Q = reward + (not_done * self.discount * target_V)
        target_Q = target_Q.detach()

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)
        logger.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(logger, step)

    # def quantile_v_single(self, single_obs):
    #     n = 100
    #     dist = self.actor(single_obs)
    #     actions = dist.sample((n,))
    #     repeated_obs = torch.unsqueeze(single_obs, dim=0).repeat(
    #         (n, *[1 for _ in single_obs.shape]))
    #     obs_action = torch.cat([repeated_obs, actions], dim=-1)
    #     values = self.critic.Q1(obs_action).reshape(-1)
    #     quantile_value = torch.kthvalue(values, int(n * self.quantile)).values
    #     return quantile_value

    # def quantile_v_sequential(self, obs):
    #     result = torch.Tensor(obs.shape[0], 1).to(self.device)
    #     for i in range(obs.shape[0]):
    #         result[i][0] = self.quantile_v_single(obs[i])
    #     return result

    # def quantile_v(self, obs):
    #     bsize = obs.shape[0]
    #     n = 32
    #     dist = self.actor(obs)
    #     actions = dist.sample((n,))  # n x bsize x *action_shape
    #     actions = actions.transpose(0, 1)  # bsize x n x *action_shape
    #     actions = actions.reshape((bsize * n, *actions.shape[2:]))
    #     # -> bsize * n x *action_shape  =  [b0s0, b0s1, ..., b1s0, b1s1]
    #     repeated_obs = obs.repeat_interleave(n, dim=0)
    #     obs_action = torch.cat([repeated_obs, actions], dim=-1)
    #     values = self.critic.Q1(obs_action).reshape((bsize, n))
    #     if self.expectation:
    #         return values.mean(dim=1)
    #     else:
    #         k = int(n * self.quantile)
    #         quantile_values = torch.kthvalue(values, k).values
    #         return quantile_values.reshape((bsize, 1))
        

    def update_actor_and_alpha(self, obs, logger, step):
        dist = self.actor(obs)

        # rsample action for entropy regularization
        action = dist.rsample()
        r_log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        # sample actions for quantile
        bsize = obs.shape[0]
        n = self.n_samples
        dist = self.actor(obs)
        sampled_actions = dist.sample((n,))  # n x bsize x action_dim
        actions = sampled_actions.transpose(0, 1)  # bsize x n x action_dim
        actions = actions.reshape((bsize * n, *actions.shape[2:]))
        # -> bsize * n x action_dim  =  [b0s0, b0s1, ..., b1s0, b1s1]

        log_prob = dist.log_prob(sampled_actions) # n x bsize x action_dim
        log_prob = log_prob.transpose(0, 1).sum(-1, keepdim=True) # bsize x n x  1
        log_prob = log_prob.reshape((bsize * n, 1)) # bsize * n x  1

        repeated_obs = obs.repeat_interleave(n, dim=0) # b_size * n x obs_dim
        q1, q2 = self.critic(repeated_obs, actions)
        q_values = torch.min(q1, q2)
        #obs_action = torch.cat([repeated_obs, actions], dim=-1)
        #q_values = self.critic.Q1(obs_action) # bsize * n x 1
        reshaped_q_values = q_values.reshape((bsize, n)) # bsize x n
        
        if self.expectation:
            baseline = reshaped_q_values.mean(dim=1, keepdim=True) # bsize x 1
            repeated_baseline = baseline.repeat_interleave(n, dim=0) # bsize * n x 1
        else:
            k = int(n * self.quantile)
            quantile_values = torch.kthvalue(reshaped_q_values, k).values
            baseline = quantile_values.reshape((bsize, 1)) # bsize x 1
            repeated_baseline = baseline.repeat_interleave(n, dim=0) # bsize * n x 1

        if self.all_actions:
            advantage = torch.detach(q_values - repeated_baseline)
        else:
            # sample new action for policy gradient
            action = dist.sample() 
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            # obs_action = torch.cat([obs, action], dim=-1)
            # q_value = self.critic.Q1(obs_action)
            q1, q2 = self.critic(repeated_obs, actions)
            q_value = torch.min(q1, q2)
            advantage = torch.detach(q_value - baseline)

        if self.tune_entropy:
            actor_loss = (self.alpha.detach() * r_log_prob).mean() - \
                             (log_prob * advantage).mean()
        else:
            actor_loss = (self.entropy_reg * r_log_prob).mean() - \
                             (log_prob * advantage).mean()

        logger.log('train_actor/loss', actor_loss, step)
        logger.log('train_actor/target_entropy', self.target_entropy, step)
        logger.log('train_actor/entropy', -log_prob.mean(), step)
        logger.log('train_actor/advantage', advantage.mean(), step)
        logger.log('train_actor/advstd', advantage.std(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(logger, step)

        if self.learnable_temperature:
            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha *
                          (-log_prob - self.target_entropy).detach()).mean()
            logger.log('train_alpha/loss', alpha_loss, step)
            logger.log('train_alpha/value', self.alpha, step)
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

    def update(self, replay_buffer, logger, step):
        obs, action, reward, next_obs, not_done, not_done_no_max = replay_buffer.sample(
            self.batch_size)

        logger.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done_no_max,
                           logger, step)

        if step % self.actor_update_frequency == 0:
            self.update_actor_and_alpha(obs, logger, step)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target,
                                     self.critic_tau)

    def save(self, model_dir, step):
        torch.save(self.actor.state_dict(),
                   '%s/actor_%s.pt' % (model_dir, step))
        torch.save(self.critic.state_dict(),
                   '%s/critic_%s.pt' % (model_dir, step))

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step)))
        self.critic.load_state_dict(
           torch.load('%s/critic_%s.pt' % (model_dir, step)))
