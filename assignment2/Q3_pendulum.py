#! /opt/homebrew/anaconda3/bin/python

from collections import namedtuple
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

    # discretize the continuous space [-2, 2] into 10 equal intervals
    env1 = ENV('Pendulum-v1', 10, 6) # 3x2, 3 is the number of observations in the env
    env2 = ENV('Pendulum-v1', 10, 2) # 'encoding_dim' = math.ceil(3/2), 3 is the number of observations in the env
    ENVS = [env1, env2]