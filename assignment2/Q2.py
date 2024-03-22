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
from typing import List, Tuple
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

EVAL_EPISODE_INTERVAL = 10
EVAL_EPISODE = 5

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

    def __init__(self, 
                 device:str, 
                 env_name: str, 
                 n_actions: int, 
                 encoding_dim: int, 
                 batch_size: int=32, 
                 gamma: float=0.9, 
                 lambda_val=1e-3, 
                 alpha = 1e-2, 
                 epsilon=0.2,
                 T = 1000,
                 N = 500,
                 N0 = 50,
                 N1 = 10,
                 N_eval = 5
                ):
        # used for both parts of training
        self.T = T # max time step per episode
        self.N = N # total number of episodes consisting of multiple cycles
        self.N0 = N0 # number of episodes per cycle consisting of one warm-up phase, then an autoencoder update phase and an LSTD update phase repeated twice
        self.N1 = N1 # number of episodes per phase, including the warm-up, the antoencoder update, and the LSTD update phase
        self.N_eval = N_eval
        assert( N%N0==0 )
        assert( (N0-N1) % (2*N1) ==0 )
        
        self.total_reward: float = 0
        self.total_reward_list: List[float] = []
        self.GD_step: int = 0
        self.n_actions = n_actions
        self.training_episode = 0
        self.env = gym.make(env_name)
        self.encoding_dim = encoding_dim
        self.steps_done = 0
        self.eps = epsilon
        self.eps_decay_denom = T
        self.device = device
        # used for the LSTD
        self.gamma = gamma
        self.inv_A: torch.Tensor = torch.eye((n_actions, encoding_dim, encoding_dim)).to(device) / lambda_val  # torch.eye creates identity matrix
        self.b: torch.Tensor = torch.zeros((n_actions, encoding_dim, encoding_dim)).to(device) # Initialize b 
        self.theta: torch.Tensor = torch.rand((n_actions, encoding_dim, encoding_dim)).to(device) # creates a matrix of random numbers between 0 and 1 with the given shape
        # used for the autoencoder
        self.mem= ReplayMemory(T*N, batch_size)
        self.batch_size = batch_size
        self.lr = alpha
        self.state_dim = self.env.observation_space.shape[0]
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim).to(device)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=alpha)
        self.eval_records: List[Tuple[int, float, float]] = [] 


    def Q_sa_2D_tensor(weight, phi_s):
        # returns to shape [encoding_dim, 1]
        return torch.matmul(weight, phi_s.view(-1, 1)) # .view(-1, 1) enforces transformation of an 1D vector to a column vector to enable torch.matmul


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
            phi_s: torch.Tensor = self.get_phi_s(state)
            result = torch.matmul(self.theta, phi_s.view(-1, 1))
            result_squeezed = torch.squeeze(result, -1)  # Squeezing the last dimension
            return torch.argmax(result_squeezed, dim=0)
        
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


    def reset_LSTD_weights(self):
        self.inv_A: torch.Tensor = torch.eye((self.n_actions, self.encoding_dim, self.encoding_dim)).to(self.device) / self.lambda_val  # torch.eye creates identity matrix
        self.b: torch.Tensor = torch.zeros((self.n_actions, self.encoding_dim, self.encoding_dim)).to(self.device) # Initialize b 
        self.theta: torch.Tensor = torch.rand((self.n_actions, self.encoding_dim, self.encoding_dim)).to(self.device) # creates a matrix of random numbers between 0 and 1 with the given shape


    def run_warm_up_phase(self):

        for _ in range(self.N1):
            state = self.get_reset_state()
            self.training_episode += 1

            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated
                
                if done:
                    break
 
                next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
                self.mem.push(state)
                state = next_state


    def run_autoencoder_update_phase(self):

        self.reset_LSTD_weights()

        for _ in range(self.N1):
            state = self.get_reset_state()
            self.training_episode += 1
 
            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated

                if done:
                    break
 
                next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
                self.mem.push(state)
                state = next_state
                self.optimize_autoencoder()
                self.GD_step += 1


    def run_LSTD_update_phase(self):

        # if it is the final phase of the entire training process, reset the LSTD weights
        if self.training_episode + self.N1 >= self.N:
            self.reset_LSTD_weights()

        for _ in range(self.N1):
            state = self.get_reset_state()
            self.training_episode += 1

            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated

                if done:
                    break
 
                next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
                self.mem.push(state)

                phi_s: torch.Tensor = self.get_phi_s(state)
                phi_s_next: torch.Tensor = self.get_phi_s(next_state)

                # calculting temporal difference
                tau: torch.Tensor = phi_s - self.gamma * phi_s_next # shape of tau would be (encoding_dim, 1)
                v: torch.Tensor = torch.matmul(tau.t(), self.inv_A[action]) # shape of tau.t() would be (1, encoding_dim), and self.inv_A[action] would be (encoding_dim, encoding_dim)
                
                """
                Update inv_A using the Sherman-Morrison formula
                """
                # numerator would have the shape of (encoding_dim, encoding_dim)
                numerator = torch.matmul(torch.matmul(self.inv_A[action], phi_s.view(-1, 1)), v.view(1, -1)) #torch.Tensor.view(-1, 1) transforms a tensor into a column vector, torch.Tensor.view(1, -1) transforms a tensor into a row vector
                # denominator would become a scalar due to its shape [1, 1]
                denominator = 1 + torch.matmul(v, phi_s)
                self.inv_A[action] -= numerator / denominator

                """
                Update b
                """
                self.b[action] += ((reward + self.gamma * torch.max(self.Q_sa_2D_tensor(self.theta[action], phi_s_next), dim=0)) * phi_s.squeeze()).unsqueeze(1)

                """
                Update action-specific LSTD weights
                """
                self.theta[action] = torch.matmul(self.inv_A[action], self.b[action])

                state = next_state


    def run_training_cycle(self):
        for _ in range(int(self.N/self.N0)):
            self.steps_done = 0 # reset the steps_done variable to 0 at the start of each training cycle to increase randomization at eps-greedy policy
            self.run_warm_up_phase()
            self.run_eval_mode()
            for _ in range( int((self.N0- self.N1)/(2* self.N1)) ):
                self.run_autoencoder_update_phase()
                self.run_eval_mode()
                self.run_LSTD_update_phase()
                self.run_eval_mode()


    def run_eval_mode(self):
        """
        at the evaluation mode, 
        1. The evaluation gets repeated for $N_episode episodes
        2. The replay buffer doesn't get updated at each time step
        3. The weights of the autoencoder don't get updated
        4. The LSTD weights "theta" and the A^(-1) & b tensors don't get updated
        5. The member variable affecting eps-greedy policy ".steps_done" does not get updated
        6. The member variable ".GD_step" does not get updated
        7. The reward given by eps-greedy actions at each time step of an evaluation episode gets summed up to calculate the total reward of an episode
        8. The mean value and the standard deviation of the total rewards across $N_episode episodes gets calculated and added to a member variable at the end of the method
        """
        total_rewards: List[float] = []

        for _ in range(self.N_eval):

            total_reward = 0
            state = self.get_reset_state()
            
            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated
                total_reward += reward

                if done:
                    break

                next_state = F.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0))
                state = next_state

            # when an evaluation episode ends, append the total reward to the list
            total_rewards.append(total_reward)

        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        # return to the three variables that are necessary to generate the required plots
        self.eval_records.append((self.GD_step, mean_reward, std_reward))
