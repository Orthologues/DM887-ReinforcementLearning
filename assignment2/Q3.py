#! /opt/homebrew/anaconda3/bin/python

import matplotlib.pyplot as plt
import gymnasium as gym
import torch
from collections import namedtuple
from Q2 import LSTD_DQL_learner

ENV = namedtuple('env', ('name', 'n_actions', 'encoding_dim'))

env1 = ENV('Acrobot-v1', 3, 12) # 6x2, 6 is the number of observations in the env
env2 = ENV('MountainCar-v0', 3, 4) # 2X2, 2 is the number of observations in the env
# discretize the continuous space [-2, 2] into 10 equal intervals
env3 = ENV('Pendulum-v1', 10, 6) # 3x2, 3 is the number of observations in the env

ENVS = [env1, env2, env3]

# use GPU for training if possible
if __name__ == "__main__":
    # turn on interactive mode of pyplot
    plt.ion()
    # use GPU training if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device == "cuda:0":
        print("Running on the GPU")
    # create the learner object
    learner = LSTD_DQL_learner()
    
