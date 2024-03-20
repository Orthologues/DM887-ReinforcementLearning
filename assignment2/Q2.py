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
References
1. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

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

    def __init__(self, capacity: int, env_name: str, encoding_dim: int, batch_size: int=32, gamma: float=0.9):
        self.mem= ReplayMemory(capacity, batch_size)
        self.inv_A = None
        self.gamma = gamma
        # "phi" is the synonym of "feature extractor", which is an autoencoder here 
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.encoding_dim = encoding_dim
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=1e-3)


    def extract_features(self, state):
        # Use the encoder part of the autoencoder to extract features without using gradient descent
        with torch.no_grad():
            phi_s = self.autoencoder.encode(state)
        return phi_s
    

    def update_LSTD_inv_A_online(self, state, next_state):
        td_tensor = self.extract_features(state) - self.gamma * self.extract_features(next_state)
        col_vec = td_tensor.view(-1, 1)  # transformation to a column vector
        row_vec = td_tensor.view(1, -1)  # transformation to a row vector

        # Sherman-Morrison formula
        uv = torch.mm(v, u)
        inv_A_uv = torch.mm(self.inv_A, uv)
        denominator = 1 + torch.mm(u, torch.mm(self.inv_A, v))
        updated_inv_A = self.inv_A - inv_A_uv / denominator
        self.inv_A = updated_inv_A