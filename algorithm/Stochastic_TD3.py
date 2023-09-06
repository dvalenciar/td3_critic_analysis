"""
Stochastic TD3
Here the critic return a probability distribution.
A single critic is used
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F

from networks.stochastic_critic_td3 import Actor
from networks.stochastic_critic_td3 import Stochastic_Critic as Critic

class STC_TD3(object):
    def __init__(self,
                 observation_size,
                 action_num,
                 device):

        self.actor_net  = Actor(observation_size=observation_size, action_num = action_num).to(device)
        self.critic_net = Critic(observation_size=observation_size, action_num = action_num).to(device)

        self.target_actor_net  = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = 0.99
        self.tau   = 0.005

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.action_num = action_num
        self.device     = device

        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_net_optimiser  = torch.optim.Adam(self.actor_net.parameters(),   lr=lr_actor)
        self.critic_net_optimiser = torch.optim.Adam(self.critic_net.parameters(),  lr=lr_critic)


    def select_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise  = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()
        return action

    def train_policy(self, experiences):
        self.learn_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, min=-0.5, max=0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u , std = self.target_critic_net(next_states, next_actions)





            # target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            # target_q_values = torch.minimum(target_q_values_one, target_q_values_two)
            #
            # q_target = rewards + self.gamma * (1 - dones) * target_q_values

