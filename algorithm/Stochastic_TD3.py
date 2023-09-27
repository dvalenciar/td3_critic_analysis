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


from networks.stochastic_critic_td3 import Actor
from networks.stochastic_critic_td3 import Stochastic_Critic as Critic

class STC_TD3(object):
    def __init__(self,
                 observation_size=10,
                 action_num=2,
                 device='cuda',
                 ensemble_size=2):

        self.gamma = 0.99
        self.tau   = 0.005
        self.learn_counter      = 0
        self.policy_update_freq = 2
        self.action_num         = action_num
        self.device             = device


        self.actor_net        = Actor(observation_size=observation_size, action_num = action_num).to(device)
        self.target_actor_net = copy.deepcopy(self.actor_net).to(device)

        lr_actor   = 1e-4
        self.actor_net_optimiser = torch.optim.Adam(self.actor_net.parameters(), lr=lr_actor)

        # ------------- Ensemble of critics ------------------#
        self.ensemble_size    = ensemble_size
        self.ensemble_critics = torch.nn.ModuleList()
        critics = [Critic(observation_size=observation_size, action_num = action_num) for _ in range(self.ensemble_size)]
        self.ensemble_critics.extend(critics)
        self.ensemble_critics.to(device)

        # Ensemble of target critics
        self.target_ensemble_critics = copy.deepcopy(self.ensemble_critics).to(device)

        lr_ensemble_critic = 1e-3
        self.ensemble_critics_optimizers = [torch.optim.Adam(self.ensemble_critics[i].parameters(), lr=lr_ensemble_critic) for i in range(self.ensemble_size)]
        #-----------------------------------------#

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

        with (torch.no_grad()):
            next_actions = self.target_actor_net(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)  # TODO what about this value 0.2 can be changed?
            target_noise = torch.clamp(target_noise, min=-0.5, max=0.5) # TODO same here
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u_set   = []
            std_set = []
            for target_critic_net in self.target_ensemble_critics:
                u, std = target_critic_net(next_states, next_actions)
                u_set.append(u)
                std_set.append(std)

            # -------- Key part here -------------- #
            # mean value
            # average the distributions to create a unique distribution, encapsulating the whole outputs
            # note, this is not a mixture of gaussians
            u_aver   = torch.mean(torch.concat(u_set, dim=1), dim=1).unsqueeze(0).reshape(batch_size, 1)
            std_aver = torch.mean(torch.concat(std_set, dim=1), dim=1).unsqueeze(0).reshape(batch_size, 1)

            # minimum value
            #u_min =  torch.min(torch.concat(u_set, dim=1), dim=1).values.unsqueeze(0).reshape(batch_size, 1)
            #std_min = what to do with the right std order, maybe need to take the de index value .index of the mean values

            # kalman filter


            # Create the target distribution = aX+b
            u_target   =  rewards +  self.gamma * u_aver * (1 - dones)
            std_target =  self.gamma * std_aver
            target_distribution = torch.distributions.normal.Normal(u_target, std_target)

        for critic_net, critic_net_optimiser in zip(self.ensemble_critics, self.ensemble_critics_optimizers):
            u_current, std_current = critic_net(states, actions)
            current_distribution   = torch.distributions.normal.Normal(u_current, std_current)

            # Compute each critic loss
            critic_individual_loss = torch.distributions.kl_divergence(current_distribution, target_distribution).mean() # todo try other divergence too

            # Update each Critic
            critic_net_optimiser.zero_grad()
            critic_individual_loss.backward()
            critic_net_optimiser.step()


        if self.learn_counter % self.policy_update_freq == 0:  # todo try if i change the freq update

            actor_q_u_set = []
            for critic_net in self.ensemble_critics:
                actor_q_u, actor_q_std = critic_net(states, self.actor_net(states))
                actor_q_u_set.append(actor_q_u)
            actor_q_u_aver = torch.mean(torch.concat(actor_q_u_set, dim=1), dim=1).unsqueeze(0).reshape(batch_size, 1)
            actor_loss     = -actor_q_u_aver.mean()

            #actor_distribution = torch.distributions.normal.Normal(actor_q_u_aver, actor_q_std_aver)  # todo what can i do with this?

            # Update Actor
            self.actor_net_optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net_optimiser.step()

            # Update ensemble of target critics
            for critic_net, target_critic_net in zip (self.ensemble_critics, self.target_ensemble_critics):
                for target_param, param in zip(target_critic_net.parameters(), critic_net.parameters()):
                    target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            # Update target actor
            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f'{path}/{filename}_actor.pht')
        torch.save(self.ensemble_critics.state_dict(),f'models/{filename}_ensemble.pht')
        logging.info("models has been saved...")