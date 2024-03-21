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
from typing import List


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

T = 1000 # max time step per episode
N = 500 # total number of episodes which consists of 10 cycles
N0 = 50 # number of episodes per cycle consisting of one warm-up phase, then an autoencoder update phase and an LSTD update phase repeated twice
N1 = 10 # number of episodes per phase, including the warm-up, the antoencoder update, and the LSTD update phase

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
    def __init__(self, state_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        # Encoder, i.e. feature extractor
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32),
            nn.ReLU(True),
            nn.Linear(32, encoding_dim),
            nn.ReLU(True)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(True),
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, state_dim),
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

    def push(self, state: torch.tensor):
        """Save a state to the buffer instead of a transition defined at standard DQN"""
        self.memory.append(state)

    def sample(self) -> List[torch.tensor]:
        if len(self.memory) < self.batch_size:
            return list(self.memory)
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class LSTD_DQL_learner():

    def __init__(self, device:str, env_name: str, n_actions: int, encoding_dim: int, batch_size: int=32, gamma: float=0.9, lambda_val=1e-3, learning_rate = 1e-2, epsilon=0.2):
        # used for both parts of training
        self.n_actions = n_actions
        self.episode = 0
        self.env = gym.make(env_name)
        self.encoding_dim = encoding_dim
        self.steps_done = 0
        self.eps = epsilon
        self.eps_decay_denom = T
        self.device = device
        # used for the LSTD
        self.gamma = gamma
        self.inv_A: torch.Tensor = torch.eye((n_actions, encoding_dim)).to(device) / lambda_val  # torch.eye creates identity matrix
        self.b: torch.Tensor = torch.zeros((n_actions, encoding_dim)).to(device) # Initialize b 
        self.theta: torch.Tensor = torch.rand((n_actions, encoding_dim)).to(device) # creates a matrix of random numbers between 0 and 1 with the given shape
        # used for the autoencoder
        self.mem= ReplayMemory(T*N, batch_size)
        self.batch_size = batch_size
        self.lr = learning_rate
        self.state_dim = self.env.observation_space.shape[0]
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim).to(device)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=learning_rate)


    def get_Q_sa_tensor(self, phi_s):
        return torch.matmul(self.theta, phi_s).view(-1, 1) # .view(-1, 1) enforces transformation of an 1D vector to a column vector to enable torch.matmul


    def get_phi_s(self, state): # extract the features from the state using the autoencoder
        # Use the encoder part of the autoencoder to extract features without using gradient descent
        with torch.no_grad():
            phi_s = self.autoencoder.encode(state)
        return phi_s
    

    def epsilon_greedy_action(self, state) -> torch.Tensor:
        rand_num = random.random() # a random number between 0 and 1
        EPS_LOW = self.eps # 0.2
        EPS_HIGH = 1 - self.eps # 0.8
        # actions at the start of a training cycle are highly random since $EPS_THRESHOLD starts at almost 0.2 and converges to 0.8
        EPS_THRESHOLD = EPS_LOW + (EPS_HIGH - EPS_LOW) * (1. - math.exp(-1. * self.steps_done / self.eps_decay_denom))
        self.steps_done += 1
        # select a policy
        if rand_num < EPS_THRESHOLD:
            phi_s = self.get_phi_s(state)
            return torch.argmax(self.get_Q_sa_tensor(phi_s))
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.device, dtype=torch.long)
        

    def optimize_autoencoder(self):
        if len(self.mem) < self.batch_size:
            return
        sampled_states: List[torch.tensor] = self.mem.sample(self.batch_size)
        state_batch = torch.cat(sampled_states)
        states_reconstructed = self.autoencoder.forward(state_batch)
        loss = torch.nn.MSELoss()(states_reconstructed, state_batch)
        # Optimize the model
        self.autoencoder_optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer accumulated as previous steps
        loss.backward() # computes dloss/dx
        self.autoencoder_optimizer.step() # update thr weights


    def get_reset_state(self):
        state, _ = self.env.reset()
        # F.sigmoid is used for normalization
        return F.sigmoid(torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0))


    def reset_inv_A_and_b_for_LSTD(self):
        self.inv_A: torch.Tensor = torch.eye((self.n_actions, self.encoding_dim)).to(self.device) / self.lambda_val  # torch.eye creates identity matrix
        self.b: torch.Tensor = torch.zeros((self.n_actions, self.encoding_dim)).to(self.device) # Initialize b 
        self.theta: torch.Tensor = torch.rand((self.n_actions, self.encoding_dim)).to(self.device) # creates a matrix of random numbers between 0 and 1 with the given shape


    def run_warm_up_phase(self):
        for _ in range(N1):
            state = self.get_reset_state()
            for _ in range(T):
                action = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated

                if done:
                    break
 
                next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
                self.mem.push(state)
                state = next_state

            self.episode += 1


    def run_autoencoder_update_phase(self):
        for _ in range(N1):
            state = self.get_reset_state()
            action = self.epsilon_greedy_action(state)
            observation, _, terminated, truncated, _ = self.env.step(action.item()) # reward: float
            done = terminated or truncated
            self.episode += 1

            if done:
                break
 
            next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
            self.mem.push(state)
            state = next_state
            self.optimize_autoencoder()


    def run_LSTD_update_phase(self):
        for _ in range(N1):
            state = self.get_reset_state()

            self.episode += 1


    def run_training_cycle(self):
        for _ in range(int(N/N0)):
            self.run_warm_up_phase()
            for _ in range( int((N0-N1)/(2*N1)) ):
                self.reset_inv_A_and_b_for_LSTD()
                self.run_autoencoder_update_phase()
                if self.episode + N1 >= N:
                    self.reset_inv_A_and_b_for_LSTD()
                self.run_LSTD_update_phase()


if __name__ == "__main__":

    """
    general setup of plotting and 
    """
    plt.ion()
    # use GPU training if possible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    """
    only for experimental purposes
    """
    #explore_envs()
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" # CUDA doesn't work with the AMD GPUs of MacBook M1
    if device == "cuda:0":
        print("Training on the GPU")
    else:
        print("Training on the CPU")
    # define the params for gym env training after exploring the envs
    ENV = namedtuple('env', ('name', 'n_actions', 'encoding_dim'))
    env1 = ENV('Acrobot-v1', 3, 12) # 6x2, 6 is the number of observations in the env
    env2 = ENV('MountainCar-v0', 3, 4) # 2X2, 2 is the number of observations in the env
    # discretize the continuous space [-2, 2] into 10 equal intervals
    env3 = ENV('Pendulum-v1', 10, 6) # 3x2, 3 is the number of observations in the env
    ENVS = [env1, env2, env3]