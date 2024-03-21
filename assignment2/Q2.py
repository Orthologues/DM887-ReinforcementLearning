#! /opt/homebrew/anaconda3/bin/python

import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""
Author: Jiawei Zhao
Date: 19.03.2024
Question 2 and Question 3 of Assignment 2 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
Requirements: 

References
1. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
2. https://gymnasium.farama.org/environments/classic_control/
"""

T = 500 # max time step per episode
N = 1050 # number of total episodes
N0 = 50 # number of episodes per phase, including the warm-up, the antoencoder update, and the LSTD update phase

"""
general setup of plotting and 
"""
plt.ion()
# use GPU training if possible
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



"""
exploring the state dimensions and action spaces of each of the required gym envs
"""
ENVS = ("Acrobot-v1", "MountainCar-v0", "Pendulum-v1")
def explore_envs():
    for env_name in ENVS:
        env = gym.make(env_name)
        print(f"Number of observations: {env.observation_space.shape[0]}")
        print(f"Action space: {env.action_space}")
        if type(env.action_space).__name__ == 'Discrete':
            print(f"Number of possible actions of {env_name}: {env.action_space.n}")
        else:
            print(f"Action space of {env_name} has to undergo discretization!")
        env.close()

class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder, i.e. feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, encoding_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # Use Sigmoid for normalized input
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)


class ReplayMemory(object):

    def __init__(self, capacity: int, batch_size: int):
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity) # storing only the states to save memory instead using the standard (s, a, r, s') version

    def push(self, state: tuple):
        """Save a state to the buffer instead of a transition defined at standard DQN"""
        self.memory.append(state)

    def sample(self):
        if len(self.memory) < self.batch_size:
            return []
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class LSTD_DQL_learner():

    def __init__(self, device:str, env_name: str, n_actions: int, encoding_dim: int, batch_size: int=32, gamma: float=0.9, lambda_val=1e-2, learning_rate = 1e-3):
        # used for both parts of training
        self.n_actions = n_actions
        self.env = gym.make(env_name)
        self.encoding_dim = encoding_dim
        # used for the LSTD
        self.gamma = gamma
        self.inv_A = [torch.eye(encoding_dim) / lambda_val for _ in range(n_actions)] # torch.eye creates identity matrix by default
        self.b = [torch.zeros(encoding_dim) for _ in range(n_actions) ] # Initialize b as a zero vector
        self.theta = torch.rand((n_actions, encoding_dim)).to(device) # creates a matrix of random numbers between 0 and 1 with the given shape
        # used for the autoencoder
        self.mem= ReplayMemory(T*N, batch_size)
        self.batch_size = batch_size
        self.features()
        self.learning_rate = learning_rate
        self.state_dim = self.env.observation_space.shape[0]
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim).to(device)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)


    def extract_features(self, state):
        # Use the encoder part of the autoencoder to extract features without using gradient descent
        with torch.no_grad():
            phi_s = self.autoencoder.encode(state)
        self.features = phi_s
    

    def update_LSTD_inv_A_and_b(self, s, s_next, r):
        phi_s = self.extract_features(s)
        phi_s_next = self.extract_features(s_next)
        tau: torch.Tensor = phi_s - self.gamma * phi_s_next
        v: torch.Tensor = torch.matmul(tau.t(), self.inv_A)

        # Update inv_A using the Sherman-Morrison formula
        numerator = torch.matmul(torch.matmul(self.inv_A, phi_s.view(-1, 1)), v.view(1, -1)) #torch.Tensor.view(-1, 1) transforms a tensor into a column vector, torch.Tensor.view(1, -1) transforms a tensor into a row vector
        denominator = 1 + torch.matmul(v, phi_s)
        self.inv_A -= numerator / denominator

        # Update b = b + r * phi(s_t)
        self.b += r * phi_s



if __name__ == "__main__":
    """
    only for experimental purposes
    """
    #explore_envs()
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" # CUDA doesn't work with the AMD GPUs of MacBook M1
    if device == "cuda:0":
        print("Training on the GPU")
    else:
        print("Training on the CPU")
    # define the objects use for gym env training after exploring the envs
    ENV = namedtuple('env', ('name', 'n_actions', 'encoding_dim'))
    env1 = ENV('Acrobot-v1', 3, 12) # 6x2, 6 is the number of observations in the env
    env2 = ENV('MountainCar-v0', 3, 4) # 2X2, 2 is the number of observations in the env
    # discretize the continuous space [-2, 2] into 10 equal intervals
    env3 = ENV('Pendulum-v1', 10, 6) # 3x2, 3 is the number of observations in the env
    ENVS = [env1, env2, env3]