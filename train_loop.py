
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
from algorithm import TD3

def define_parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--G', type=int, default=1)

    parser.add_argument('--max_steps_training', type=int, default=1000000)
    parser.add_argument('--max_steps_pre_exploration', type=int, default=1000)
    parser.add_argument('--number_eval_episodes', type=int, default=10)

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--env',  type=str, default="ball_in_cup")
    parser.add_argument('--task', type=str, default="catch")
    args   = parser.parse_args()
    return args


def save_reward_values(data_reward, filename):
    data = pd.DataFrame.from_dict(data_reward)
    data.to_csv(f"data_results/{filename}", index=False)


def grab_frame(env):
    frame = env.physics.render(camera_id=0, height=480, width=600)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert to BGR for use with OpenCV
    return frame


def  evaluation_loop(env, agent, total_step_counter, file_name, historical_reward_evaluation, args):
    fps        = 30
    video_name = f'videos_evaluation/{file_name}_{total_step_counter}.mp4'
    frame = grab_frame(env)
    height, width, channels = frame.shape
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    historical_episode_reward = []

    for episode_num in range(args.number_eval_episodes):
        start_time = time.time()
        time_step  = env.reset()
        state = np.hstack(list(time_step.observation.values()))
        done  = False
        episode_reward    = 0
        episode_timesteps = 0

        while not done:
            if episode_num == 0:
                video.write(grab_frame(env))
            episode_timesteps += 1
            action    = agent.select_action_from_policy(state, evaluation=True)
            time_step = env.step(action)
            state, reward, done = np.hstack(list(time_step.observation.values())), time_step.reward, time_step.last()
            episode_reward += reward

        episode_duration = time.time() - start_time
        logging.info( f" EVALUATION | Eval Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")
        historical_episode_reward.append(episode_reward)

    mean_reward_evaluation = np.round(np.mean(historical_episode_reward), 2)
    historical_reward_evaluation["avg_episode_reward"].append(mean_reward_evaluation)
    historical_reward_evaluation["step"].append(total_step_counter)

    save_reward_values(historical_reward_evaluation, file_name+"_evaluation")
    video.release()


def train(env, agent, action_size, file_name, args):
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

    historical_reward_training   = {"step": [], "episode_reward": []}
    historical_reward_evaluation = {"step": [], "avg_episode_reward": []}

    # To store zero at the beginning
    historical_reward_evaluation["step"].append(0)
    historical_reward_evaluation["avg_episode_reward"].append(0)

    start_time = time.time()
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
        episode_reward += reward

        if total_step_counter > max_steps_pre_exploration:
            for _ in range(G):
                experience = memory.sample(batch_size)
                agent.train_policy((
                    experience['state'],
                    experience['action'],
                    experience['reward'],
                    experience['next_state'],
                    experience['done'],
                ))

        if done:
            episode_duration = time.time() - start_time

            logging.info(f"TRAIN T:{total_step_counter} | Episode {episode_num + 1} was completed with {episode_timesteps} steps | Reward= {episode_reward:.3f} | Duration= {episode_duration:.2f} Sec")

            historical_reward_training["step"].append(total_step_counter)
            historical_reward_training["episode_reward"].append(episode_reward)

            if total_step_counter % 10_000 == 0:
                logging.info("*************--Evaluation Loop--*************")
                save_reward_values(historical_reward_training, file_name + "_training")
                evaluation_loop(env, agent, total_step_counter, file_name, historical_reward_evaluation, args)
                logging.info("--------------------------------------------")

            # Reset environment
            start_time = time.time()
            time_step  = env.reset()
            state      = np.hstack(list(time_step.observation.values()))
            episode_reward    = 0
            episode_timesteps = 0
            episode_num       += 1

    agent.save_models(filename=file_name)
    save_reward_values(historical_reward_training, file_name + "_training")
    logging.info("All GOOD AND DONE :)")


def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f" Working with = {device}")

    args = define_parse_args()
    domain_name = args.env
    task_name   = args.task
    seed        = args.seed
    logging.info(f" Environment and Task Selected: {domain_name}_{task_name}")

    # ------------------------------------------------#
    env         = suite.load(domain_name, task_name, task_kwargs={'random': seed})
    time_step   = env.reset()
    action_spec = env.action_spec()
    action_size = action_spec.shape[0]
    logging.info(f" Number of Action for this Env: {action_size}")

    observation      = np.hstack(list(time_step.observation.values()))
    observation_size = len(observation)
    logging.info(f" Observation Size for this Env: {observation_size}")
    # ------------------------------------------------#

    # set seeds
    # ------------------------------------------------#
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # ------------------------------------------------#
    # ------------------------------------------------#
    # Create Directories
    logging.info(f" Creating Folders")
    dir_exists = os.path.exists("videos_evaluation")
    if not dir_exists:
        os.makedirs("videos_evaluation")
    dir_exists = os.path.exists("data_results")
    if not dir_exists:
        os.makedirs("data_results")
    dir_exists = os.path.exists("models")
    if not dir_exists:
        os.makedirs("models")
    # ------------------------------------------------#

    # ------------------------------------------------#
    logging.info(f" Initializing Algorithm.....")
    agent = TD3(
        observation_size=observation_size,
        action_num=action_size,
        device=device
    )
    # ------------------------------------------------#

    date_time_str = datetime.now().strftime("%m_%d_%H_%M")
    file_name = domain_name + "_" + task_name + "_" + "TD3" +"_" + str(date_time_str)

    logging.info("Initializing Training Loop....")
    train(env, agent, action_size,file_name, args)



if __name__ == '__main__':
    main()