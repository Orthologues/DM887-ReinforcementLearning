#! /opt/homebrew/anaconda3/bin/python

from collections import namedtuple
from Q2 import LSTD_DQL_learner
import torch
import math
import random
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

def get_modified_reward(reward, action_discrete, action_continuous, actions):
    """
    Modifies the reward by replacing the torque^2 term.
    
    Args:
    - reward: The original continuous reward from the environment.
    - action_idx: The index of the chosen action.
    - actions: The array or tensor of discretized actions (medians of intervals).
    
    Returns:
    - Modified reward as per the new definition.
    """
    # Assume the original reward function is: -theta^2 - 0.1*theta_dt^2 - 0.001*torque^2
    # And that the torque^2 term has been isolated for replacement
    
    # Extract the selected action's value (median of the interval)
    selected_action = actions[action_idx]
    
    # Replace the torque^2 term with the selected action's squared value
    modified_reward = reward + 0.001 * selected_action ** 2  # Adjust based on your exact reward formula
    
    return modified_reward


# create a class "LSTD_DQL_pendulum_learner" inherited from "LSTD_DQL_learner" designed for the pendulum problem
class LSTD_DQL_pendulum_learner(LSTD_DQL_learner):

    def __init__(self, device: str, env_name: str, n_actions: int, encoding_dim: int, batch_size: int = 32, gamma: float = 0.9, lambda_val=0.001, alpha=0.01, epsilon=0.2, T=1000, N=500, N0=50, N1=10, N_eval=5):
        super().__init__(device, env_name, n_actions, encoding_dim, batch_size, gamma, lambda_val, alpha, epsilon, T, N, N0, N1, N_eval)

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

    # Discretize using NumPy
    action_intervals = np.linspace(-2, 2, num=11)  # 11 points create 10 intervals
    actions = (action_intervals[:-1] + action_intervals[1:]) / 2  # Medians of intervals
