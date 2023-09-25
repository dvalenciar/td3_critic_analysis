"""
Normal TD3
Nothing special here
Original Paper: https://arxiv.org/abs/1802.09477v3
"""


import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F

from networks.normal_td3 import Actor
from networks.normal_td3 import Critic

class TD3(object):
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
            action       = self.actor_net(state_tensor)
            action       = action.cpu().data.numpy().flatten()
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
        states  = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, min=-0.5, max=0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_one   = F.mse_loss(q_values_one, q_target)
        critic_loss_two   = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_q_one, actor_q_two = self.critic_net(states, self.actor_net(states))

            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss     = -actor_q_values.mean()

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net_optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_net.Q1.parameters(), self.critic_net.Q1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critic_net.Q2.parameters(), self.critic_net.Q2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f'{path}/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'{path}/{filename}_critic.pht')
        logging.info("models has been saved...")