#! /opt/homebrew/anaconda3/bin/python

from collections import namedtuple
import torch
from Q2 import LSTD_DQL_learner
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

    env1 = ENV('Acrobot-v1', 3, 12) # 'encoding_dim' = 6x2, 6 is the number of observations in the env
    env2 = ENV('Acrobot-v1', 3, 3) # 'encoding_dim' = math.ceil(6/2), 6 is the number of observations in the env
    ENVS = [env1, env2]

    """
    The following code is experimental. It turns out that halving the dimension by the encoder would lead to a better performace
    """
    #for idx, env in enumerate(ENVS):
    #    learner = LSTD_DQL_learner(
    #        env_name=env.name, 
    #        n_actions=env.n_actions, 
    #        encoding_dim=env.encoding_dim, 
    #        device=device
    #    )
    #    learner.run_training_cycle()
    #    learner.plot_total_reward_mean_and_std(f"acrobot-{idx}")

    learner = LSTD_DQL_learner(
        env_name = env2.name, 
        n_actions = env2.n_actions, 
        encoding_dim = env2.encoding_dim, 
        device=device
    )
    learner.run_training_cycle()
    learner.plot_total_reward_mean_and_std(f"acrobot")