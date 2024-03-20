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
Assignment 2 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
Requirements: 

References
1. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

T = 500 # max time step per episode
N = 1050 # number of total episodes
N0 = 50 # number of episodes per phase

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
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 'state_next'))
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition to the buffer"""
        self.memory.append(self.transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class LSTD_DQL_learner():

    def __init__(self, capacity: int, env_name: str, encoding_dim: int, batch_size: int=32, gamma: float=0.9, lambda_val=1e-3):
        self.batch_size = batch_size
        self.mem= ReplayMemory(T*N, batch_size)
        self.gamma = gamma
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.encoding_dim = encoding_dim
        self.inv_A = torch.eye(encoding_dim) / lambda_val # torch.eye creates identity matrix by default
        self.b = torch.zeros(encoding_dim) # Initialize b as a zero vector
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3)


    def extract_features(self, state):
        # Use the encoder part of the autoencoder to extract features without using gradient descent
        with torch.no_grad():
            phi_s = self.autoencoder.encode(state)
        return phi_s
    

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



    