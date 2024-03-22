#! /opt/homebrew/anaconda3/bin/python

import gymnasium as gym
import math
import random
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
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
                 T = 1000, # number of maximum time step per episode
                 N = 1000, # number of total training episodes
                 N0 = 100, # number of episodes per cycle
                 N1 = 20, # number of warm-up episodes per cycle
                 N2 = 5, # number of autoencoder update episodes per cycle
                 N3 = 35, # number of LSTD
                 N_eval = 5, # number of repeated evaluation episodes per round
                ):
        # used for both parts of training
        self.T = T # max time step per episode
        self.N = N # total number of episodes consisting of multiple cycles
        self.N0 = N0 # number of episodes per cycle consisting of one warm-up phase, then an autoencoder update phase and an LSTD update phase repeated twice
        self.N1 = N1 
        self.N2 = N2
        self.N3 = N3
        self.N_eval = N_eval
        assert( N%N0==0 )
        assert( (N0-N1) % (N2+N3)==0 )
        
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
        self.lambda_val = lambda_val
        self.reset_LSTD_weights()

        # used for the autoencoder
        self.mem= ReplayMemory(T*N, batch_size)
        self.batch_size = batch_size
        self.lr = alpha
        self.state_dim = self.env.observation_space.shape[0]
        self.autoencoder = Autoencoder(self.state_dim, encoding_dim).to(device)
        self.autoencoder_optimizer = optim.Adam(self.autoencoder.parameters(), lr=alpha)
        self.eval_records: List[Tuple[int, float, float]] = [] 


    def Q_sa_2D_tensor(self, weight, phi_s):
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
            return torch.argmax(self.Q_sa_2D_tensor(self.theta, phi_s), dim=0)
        
        return torch.tensor([self.env.action_space.sample()], device=self.device, dtype=torch.long)
    

    def optimize_autoencoder(self):
        if len(self.mem) < self.batch_size:
            return
        sampled_states: List[torch.tensor] = self.mem.sample()
        state_batch = torch.stack(sampled_states, dim=0)
        states_reconstructed = self.autoencoder.forward(state_batch)
        loss = torch.nn.MSELoss()(states_reconstructed, state_batch)
        # Optimize the model
        self.autoencoder_optimizer.zero_grad() # clears x.grad for every parameter x in the optimizer accumulated as previous steps
        loss.backward() # computes dloss/dx
        self.autoencoder_optimizer.step() # update thr weights


    def get_reset_state(self):
        state, _ = self.env.reset()
        # torch.sigmoid is used for normalization
        return torch.sigmoid(torch.tensor(state, dtype=torch.float32, device=self.device))


    def reset_LSTD_weights(self):
        self.inv_A: torch.Tensor = torch.eye(self.encoding_dim).unsqueeze(0).repeat(self.n_actions, 1, 1).to(self.device) / self.lambda_val  # torch.eye creates identity matrix
        self.b: torch.Tensor = torch.zeros((self.n_actions, self.encoding_dim)).to(self.device) # Initialize b 
        self.theta: torch.Tensor = torch.rand((self.n_actions, self.encoding_dim)).to(self.device) 


    def run_warm_up_phase(self):

        for _ in range(self.N1):
            state = self.get_reset_state()
            self.training_episode += 1

            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated
                
                if done:
                    break
 
                next_state = torch.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device))
                self.mem.push(state)
                state = next_state


    def run_autoencoder_update_phase(self):
        
        for _ in range(self.N2):
            state = self.get_reset_state()
            self.training_episode += 1
 
            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated

                if done:
                    break
 
                next_state = torch.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device))
                self.mem.push(state)
                state = next_state
                self.optimize_autoencoder()
                self.GD_step += 1


    def run_LSTD_update_phase(self):

        self.reset_LSTD_weights()

        for _ in range(self.N3):
            state = self.get_reset_state()
            self.training_episode += 1

            for _ in range(self.T):
                action = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(action.item()) # reward: float
                done = terminated or truncated

                if done:
                    break
 
                next_state = torch.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device))
                self.mem.push(state)

                phi_s: torch.Tensor = self.get_phi_s(state)
                phi_s_next: torch.Tensor = self.get_phi_s(next_state)

                # calculting temporal difference
                tau: torch.Tensor = phi_s - self.gamma * phi_s_next # shape of tau would be (1, encoding_dim)
                v: torch.Tensor = torch.matmul(tau.view(1, -1), self.inv_A[action]) # shape of tau.t() would be (1, encoding_dim), and self.inv_A[action] would be (encoding_dim, encoding_dim)
                
                """
                Update inv_A using the Sherman-Morrison formula
                """
                # numerator would have the shape of (encoding_dim, encoding_dim)
                numerator = torch.matmul(torch.matmul(self.inv_A[action], phi_s.view(-1, 1)), v.view(1, -1)).squeeze(0) #torch.Tensor.view(-1, 1) transforms a tensor into a column vector, torch.Tensor.view(1, -1) transforms a tensor into a row vector
                # denominator would become a scalar due to its shape [1, 1]
                denominator = 1 + torch.matmul(v.view(1, -1), phi_s)
                self.inv_A[action] -= numerator / denominator

                """
                Update b, the term before "* phi_s.squeeze()" is a scalar
                """
                self.b[action] += ( reward + self.gamma * torch.max(self.Q_sa_2D_tensor(self.theta, phi_s_next), dim=0).values ) * phi_s.squeeze()

                """
                Update action-specific LSTD weights
                """
                self.theta[action] = torch.matmul(self.inv_A[action], self.b[action].view(-1, 1)).squeeze()

                state = next_state


    def run_training_cycle(self):
        for _ in range(int(self.N/self.N0)):
            self.steps_done = 0
            self.run_warm_up_phase()
            self.run_eval_mode()
            for _ in range( int((self.N0- self.N1)/(self.N2 + self.N3)) ):
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

                next_state = torch.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device))
                state = next_state

            # when an evaluation episode ends, append the total reward to the list
            total_rewards.append(total_reward)

        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        # return to the three variables that are necessary to generate the required plots
        self.eval_records.append((self.GD_step, mean_reward, std_reward))


    # the plotting function 
    def plot_total_reward_mean_and_std(self, fname_suffix: str):

        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 9))
        mean_lines = []

        gd_time_steps = [ record[0] for record in self.eval_records ]
        total_r_mean = [ record[1] for record in self.eval_records ]
        mean_line, = ax.plot(gd_time_steps, total_r_mean, color='b')
        mean_lines.append(mean_line)
        # Fill between mean +/- standard error with semi-transparent color
        mean_upper = [ record[1] + record[2] for record in self.eval_records ]
        mean_lower = [ record[1] - record[2] for record in self.eval_records ]
        ax.fill_between(gd_time_steps, mean_upper, mean_lower, alpha=0.2, color='g')
        # Add labels and title
        ax.set_xlabel("Gradient-Descent time step")
        ax.set_ylabel("Mean Total Episode Reward")
        ax.set_title(f"Epsilon-greedy DQN with Autoencoder and LSTD ({fname_suffix})")
        fig.savefig(f"learning-curve-{fname_suffix}.png", dpi=150, bbox_inches='tight') 