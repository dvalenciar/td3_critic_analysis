

import os
import cv2
import time
import torch
import random
import numpy as np
import pandas as pd
from datetime import datetime
from argparse import ArgumentParser

import logging
logging.basicConfig(level=logging.INFO)

from dm_control import suite
from memory import MemoryBuffer
from algorithm import STC_TD3



def develop(env, agent, action_size, args):
    max_steps_training        = args.max_steps_training
    max_steps_pre_exploration = args.max_steps_pre_exploration
    batch_size                = args.batch_size
    G                         = args.G

    # Needed classes
    # ------------------------------------#
    memory = MemoryBuffer()

    # Training Loop
    # ------------------------------------#
    episode_timesteps = 0
    episode_reward    = 0
    episode_num       = 0

    time_step  = env.reset()
    state      = np.hstack(list(time_step.observation.values()))

    for total_step_counter in range(1, int(max_steps_training) + 1):
        episode_timesteps += 1

        if total_step_counter <= max_steps_pre_exploration:
            logging.info(f"Running Pre-Exploration Steps {total_step_counter}/{max_steps_pre_exploration}")
            action = np.random.uniform(-1, +1, size=action_size)
        else:
            action = agent.select_action_from_policy(state)

        time_step = env.step(action)
        next_state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()

        memory.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
        state = next_state
        if total_step_counter > max_steps_pre_exploration:
            experience = memory.sample(batch_size)
            agent.train_policy((
                experience['state'],
                experience['action'],
                experience['reward'],
                experience['next_state'],
                experience['done'],
            ))

        if done:
                logging.info(f"TRAIN T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f}")
                # Reset environment
                start_time = time.time()
                time_step  = env.reset()
                state      = np.hstack(list(time_step.observation.values()))
                episode_reward    = 0
                episode_timesteps = 0
                episode_num       += 1


def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--G', type=int, default=1)

    parser.add_argument('--max_steps_training', type=int, default=1000000)
    parser.add_argument('--max_steps_pre_exploration', type=int, default=1000)
    parser.add_argument('--number_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env',  type=str, default="ball_in_cup")
    parser.add_argument('--task', type=str, default="catch")
    args   = parser.parse_args()
    return args

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = define_parse_args()
    domain_name = args.env
    task_name   = args.task
    seed        = args.seed

    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})
    time_step   = env.reset()
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]

    observation      = np.hstack(list(time_step.observation.values()))
    observation_size = len(observation)
    logging.info(f" Observation Size for this Env: {observation_size}")

    agent = STC_TD3(
        observation_size=observation_size,
        action_num=action_size,
        device=device
    )

    develop(env, agent, action_size, args)


if __name__ == '__main__':
    main()