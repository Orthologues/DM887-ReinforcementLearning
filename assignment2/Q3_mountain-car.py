#! /opt/homebrew/anaconda3/bin/python

from collections import namedtuple
from Q2 import LSTD_DQL_learner
import torch
import numpy as np

"""
Author: Jiawei Zhao
Date: 19.03.2024
Question 2 of Assignment 2 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
Requirements: 

References
1. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
2. https://gymnasium.farama.org/environments/classic_control/
"""

if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" # CUDA doesn't work with the AMD GPUs of MacBook M1
    if device == "cuda:0":
        print("Training on the GPU")
    else:
        print("Training on the CPU")

    ENV = namedtuple('env', ('name', 'n_actions', 'encoding_dim'))

    env1 = ENV('MountainCar-v0', 3, 4) # 'encoding_dim' = 2 * 2, 2 is the number of observations in the env
    env2 = ENV('MountainCar-v0', 3, 1) # 'encoding_dim' = math.ceil(2/2), 2 is the number of observations in the env
    ENVS = [env1, env2]
    
    for idx, env in enumerate(ENVS):
        learner = LSTD_DQL_learner(
            env_name=env.name, 
            n_actions=env.n_actions, 
            encoding_dim=env.encoding_dim, 
            device=device
        )
        learner.run_training_cycle()
        learner.plot_total_reward_mean_and_std(f"mountaincar-{idx}")
        