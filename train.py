import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.util import MemoryFactory
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util import RLParser

import cares_reinforcement_learning.util.configurations as configurations
from cares_reinforcement_learning.util.configurations import TrainingConfig, AlgorithmConfig, EnvironmentConfig

from algorithm import STC_TD3

import cares_reinforcement_learning.train_loops.policy_loop as pbe

import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from pydantic import BaseModel, Field
from typing import List, Optional

class STC_TD3Config(AlgorithmConfig):
    algorithm: str = Field("STC_TD3", Literal=True)
    actor_lr: Optional[float] = 1e-4
    critic_lr: Optional[float] = 1e-3
    
    gamma: Optional[float] = 0.99
    tau: Optional[float] = 0.005

    ensemble_size: Optional[int] = 2
    
    memory: Optional[str] = "MemoryBuffer"

def main():
    parser = RLParser()
    parser.add_algorithm(STC_TD3Config)
    env_config, training_config, alg_config = parser.parse_args()
    
    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()
    
    env = env_factory.create_environment(env_config)

    iterations_folder = f"{alg_config.algorithm}-{env_config.task}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
    glob_log_dir = f'{Path.home()}/cares_rl_logs/{iterations_folder}'

    training_iterations = training_config.number_training_iterations

    seed = training_config.seed
    for training_iteration in range(0, training_iterations):
        logging.info(f"Training iteration {training_iteration+1}/{training_iterations} with Seed: {seed}")
        hlp.set_seed(seed)
        env.set_seed(seed)

        logging.info(f"Algorithm: {alg_config.algorithm}")
        agent = network_factory.create_network(env.observation_space, env.action_num, alg_config)
        if agent == None and alg_config.algorithm == "STC_TD3":
          agent = STC_TD3(
              observation_size=env.observation_space,
              action_num=env.action_num,
              device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
              ensemble_size=alg_config.ensemble_size
        )
        else:
            raise ValueError(f"Unkown agent for default algorithms {alg_config.algorithm}")

        # TODO manage arguements for future memory types
        memory = memory_factory.create_memory(alg_config.memory, args=[])
        logging.info(f"Memory: {alg_config.memory}")

        #create the record class - standardised results tracking
        log_dir = f"{seed}"
        record = Record(glob_log_dir=glob_log_dir, 
                        log_dir=log_dir, 
                        algorithm=alg_config.algorithm, 
                        task=env_config.task, 
                        network=agent, 
                        plot_frequency=training_config.plot_frequency, 
                        checkpoint_frequency=training_config.checkpoint_frequency)
        record.save_config(env_config, "env_config")
        record.save_config(training_config, "train_config")
        record.save_config(alg_config, "alg_config")
    
        # Train the policy or value based approach
        pbe.policy_based_train(env, agent, memory, record, training_config)
        
        record.save()
        
        seed += 10

if __name__ == '__main__':
    main()

