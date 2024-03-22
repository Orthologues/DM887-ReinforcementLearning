#! /opt/homebrew/anaconda3/bin/python

from collections import namedtuple
from Q2 import LSTD_DQL_learner
import torch
import math
import random
import numpy as np
from typing import List

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


# create a class "LSTD_DQL_pendulum_learner" inherited from "LSTD_DQL_learner" designed for the pendulum problem
class LSTD_DQL_pendulum_learner(LSTD_DQL_learner):

    def __init__(self, device: str, env_name: str, n_actions: int, encoding_dim: int, batch_size: int = 32, gamma: float = 0.9, lambda_val=0.001, alpha=0.01, epsilon=0.2, T=1000, N=1000, N0=100, N1=10, N2=5, N3=40, N_eval=5):
        super().__init__(device, env_name, n_actions, encoding_dim, batch_size, gamma, lambda_val, alpha, epsilon, T, N, N0, N1, N2, N3, N_eval)
    
        self.action_intervals = np.linspace(-2, 2, num=n_actions+1)  # Creates 100 intervals
        self.discretized_actions = (self.action_intervals[:-1] + self.action_intervals[1:]) / 2  # Medians of intervals


    def discretize_action(self, sampled_action):
        """
        Map a sampled action to the index of the discretized action.

        Args:
        - sampled_action: The action sampled from the environment's action space.
        - discretized_actions: Array of action medians representing each interval.

        Returns:
        - Index of the discretized action.
        """
        interval_length = 4.0 / len(self.discretized_actions)  # Total range divided by number of actions
        # Find the index by calculating how many interval lengths the sampled action is from the start (-2)
        action_index = int((sampled_action + 2) / interval_length)
        # Clamp the index to be within valid range
        action_index = max(0, min(action_index, len(self.discretized_actions) - 1))
        return action_index


    # overwrite the epsilon greedy action
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
            ac_index = torch.argmax(self.Q_sa_2D_tensor(self.theta, phi_s), dim=0).item()
        else:
            action_cont = torch.tensor([self.env.action_space.sample()], device=self.device).item()
            ac_index = self.discretize_action(action_cont)

        return self.discretized_actions[ac_index], ac_index


    def run_warm_up_phase(self):

        for _ in range(self.N1):
            state = self.get_reset_state()
            self.training_episode += 1

            for _ in range(self.T):
                ac_value, _ = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(torch.Tensor([ac_value])) # reward: float
                done = terminated or truncated
                
                if done:
                    break
 
                next_state = torch.sigmoid(torch.tensor(observation, dtype=torch.float32, device=self.device))
                self.mem.push(state)
                state = next_state


    # overwrite all methods with 'epsilon_greedy_action'
    def run_autoencoder_update_phase(self):

        for _ in range(self.N2):
            state = self.get_reset_state()
            self.training_episode += 1
 
            for _ in range(self.T):
                ac_value, _ = self.epsilon_greedy_action(state)
                observation, _, terminated, truncated, _ = self.env.step(torch.Tensor([ac_value])) # reward: float
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
                ac_value, action = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(torch.Tensor([ac_value])) # reward: float

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
                ac_value, _ = self.epsilon_greedy_action(state)
                observation, reward, terminated, truncated, _ = self.env.step(torch.Tensor([ac_value])) # reward: float
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


if __name__ == "__main__":
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu" # CUDA doesn't work with the AMD GPUs of MacBook M1
    if device == "cuda:0":
        print("Training on the GPU")
    else:
        print("Training on the CPU")

    ENV = namedtuple('env', ('name', 'n_actions', 'encoding_dim'))

    # discretize the continuous space [-2, 2] into 100 equal intervals
    env = ENV('Pendulum-v1', 100, 2) # 'encoding_dim' = math.ceil(3/2), 3 is the number of observations in the env

    learner = LSTD_DQL_pendulum_learner(
        env_name=env.name, 
        n_actions=env.n_actions, 
        encoding_dim=env.encoding_dim, 
        device=device
    )
    learner.run_training_cycle()
    learner.plot_total_reward_mean_and_std("pendulum")
        
