import time
import argparse
import logging
logging.basicConfig(level=logging.INFO)

from cares_reinforcement_learning.util import NetworkFactory
from cares_reinforcement_learning.util import MemoryFactory
from cares_reinforcement_learning.util import Record
from cares_reinforcement_learning.util import EnvironmentFactory
from cares_reinforcement_learning.util import arguement_parser as ap

from algorithm import STC_TD3

import cares_reinforcement_learning.train_loops.policy_loop as pbe

import torch
import random
import numpy as np
from pathlib import Path
from datetime import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def create_parser():
    env_parser = ap.environment_parser()
    alg_parser, alg_parsers = ap.algorithm_args(parent_parser=env_parser)
    
    # create the parser for STC_TD3 with default parameters
    parser_STC_TD3 = alg_parsers.add_parser('STC_TD3', help='STC_TD3', parents=[env_parser])
    parser_STC_TD3.add_argument('--ensemble_size', type=int, default=2)

    parser = ap.environment_args(parent_parser=alg_parser)
    return parser

def main():
    parser = create_parser()
    args = vars(parser.parse_args()) # converts to a dictionary

    args["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Device: {args['device']}")

    env_factory = EnvironmentFactory()
    network_factory = NetworkFactory()
    memory_factory = MemoryFactory()
    
    gym_environment = args['gym_environment']
    env = env_factory.create_environment(gym_environment=gym_environment, args=args)

    args["observation_size"] = env.observation_space
    logging.info(f"Observation Size: {args['observation_size']}")

    args['action_num'] = env.action_num
    logging.info(f"Action Num: {args['action_num']}")

    iterations_folder = f"{args['algorithm']}-{args['task']}-{datetime.now().strftime('%y_%m_%d_%H:%M:%S')}"
    glob_log_dir = f'{Path.home()}/cares_rl_logs/{iterations_folder}'

    training_iterations = args['number_training_iterations']
    for training_iteration in range(0, training_iterations):
        logging.info(f"Training iteration {training_iteration+1}/{training_iterations} with Seed: {args['seed']}")
        set_seed(args['seed'])
        env.set_seed(args['seed'])

        logging.info(f"Algorithm: {args['algorithm']}")
        agent = network_factory.create_network(args["algorithm"], args)
        if agent == None and args["algorithm"] == "STC_TD3":
          agent = STC_TD3(
              observation_size=args["observation_size"],
              action_num=args['action_num'],
              device=args["device"],
              ensemble_size=args["ensemble_size"]
        )
        else:
            raise ValueError(f"Unkown agent for default algorithms {args['algorithm']}")

        memory = memory_factory.create_memory(args['memory'], args)
        logging.info(f"Memory: {args['memory']}")

        #create the record class - standardised results tracking
        log_dir = args['seed']
        record = Record(glob_log_dir=glob_log_dir, log_dir=log_dir, network=agent, config={'args': args})
    
        # Train the policy or value based approach
        pbe.policy_based_train(env, agent, memory, record, args)
        
        record.save()
        
        args['seed'] += 10

if __name__ == '__main__':
    main()

