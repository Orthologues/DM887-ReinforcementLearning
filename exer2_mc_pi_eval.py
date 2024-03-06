#! /opt/homebrew/anaconda3/bin/python

"""
Author: Jiawei Zhao
Date: 28.02.2024

Steps: 
1) Create a GridWorld environment of size 10x10 or bigger.

2) Initialize your value function to an arbitrarily large value, e.g. J(x) = +pow(10, 6) for all x if you store returnss, J(x) = -pow(10, 6) if you store rewards.

3) Perform policy evaluation by implementing two versions of Monte Carlo simulation: a) First-visit, b) Every-visit. Define a convergence criterion, e.g. an epsilon tolerance below which the value of a state is not counted as changed, and run the algorithm until this condition is satisfied. Count how many samples you have taken and how many simulation rounds you have started.

4) Compare the sample and simulation counts for the two algorithm versions.

5) Compare the learned J to the initialized J.
"""

import numpy as np
from gridworld import GridWorld
from typing import Tuple, Set, List, Dict, Union
from random import choices
from math import pow

"""
Assignment of constants to be used in Monte-Carlo policy evaluation
"""
# Step 1
MY_GRID: str=\
    """
    wwwwwwwwwwwwwwwwww
    wa      w  w     w
    w   o   w        w
    www     www    www
    w   o        o   w
    wwwww    o     www
    w     wwwwww     w
    w     w    w  o  w
    ww         ww   gw
    wwwwwwwwwwwwwwwwww
    """
EPISILON = 1e-4
GAMMA = 0.9

# Monte-Carlo policy evaluation
class Monte_Carlo_learner:

    # epsilon represents the probability that an agent is going to take a random action
    def __init__(self, env: GridWorld, gamma=GAMMA, epsilon = 0.2):
        self.gamma = gamma
        self.env = env
        self.simulated_samples = 0
        self.episode = 0
        self.epsilon = epsilon
        self.Q_sa: np.ndarray[float] = np.full((env.state_count, env.action_size), -pow(10, 6)) # tracking cumulative rewards
        self.N_sa: np.ndarray[int] = np.zeros((env.state_count, env.action_size)) # tracking numbers of visits to each state-action pair # tracking the numbers that each state-action pair has been visited in an episode
        self.S_A_R: List[Tuple[int, int, int]] = [] # tracking the state-action-reward tuples that have been visited in an episode
        self.G_t: List[float] = [] # tracking the cumulative returns at each step per episode
        self.pi = np.zeros(shape=env.state_count)
        self.viable_actions: Dict[int, Set[int]] = dict()
        self.states_to_coords: List[Tuple[int, int]] = list(self.env.state_dict.keys())
        self.coords_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.states_to_coords)}

    def get_next_state(self, state: int, action: int) -> int:
        coord = self.states_to_coords[state]
        next_coord: Tuple[Union[int, float], Union[int, float]] = (pow(10, 6), pow(10, 6))
        if action == 0:
            next_coord = (coord[0]+1, coord[1])
        elif action == 1:
            next_coord = (coord[0], coord[1]+1)
        elif action == 2:
            next_coord = (coord[0]-1, coord[1])
        elif action == 3:
            next_coord = (coord[0], coord[1]-1)
        else:
            pass
        # check if an action is possible given a state
        if next_coord in self.coords_to_states:
            next_state = self.coords_to_states[next_coord]
            return next_state
        return state

    def get_reward(self, state: int) -> int:
        # get the reward by state_dict of the grid environment
        return self.env.state_dict[self.states_to_coords[state]]['reward']
    
    def calculate_returns(self):
        for i in range(len(self.S_A_R)):
            returns = 0
            for k, (_, _, reward) in enumerate(self.S_A_R[i+1: ]):
                returns += reward * pow(self.gamma, k)
            self.G_t.append(returns)
    
    def reset_learning(self):
        self.Q_sa: np.ndarray[float] = np.full((self.env.state_count, self.env.action_size), -pow(10, 6)) 
        self.N_sa: np.ndarray[int] = np.zeros((self.env.state_count, self.env.action_size)) 
        self.pi = np.random.choice(self.env.action_values, size=self.env.state_count)

    def reset_episode(self):
        self.S_A_R = []
        self.G_t = []

    def initialize_random_policies(self):
        for s in range(self.env.state_count):
            viable_a: Set[int] = set()
            for a in self.env.action_values:
                next_s = self.get_next_state(s, a)
                if next_s == s:
                    continue
                viable_a.add(a)
            self.pi[s] = np.random.choice(list(viable_a))
            self.viable_actions[s] = viable_a

    def generate_episode(self):
        if self.episode == 0:
            self.initialize_random_policies()
        state = 0
        while True:
            if self.env.state_dict[self.states_to_coords[state]]['done']:
                break
            # create a probability dictionary using epsilon-greedy behavior policy  
            prob_dict: Dict[int, float] = dict()
            viable_a = self.viable_actions[state]
            size_a = len(viable_a)
            for a in viable_a:
                if a == self.pi[state]:
                    prob_dict[a] = 1 - self.epsilon + self.epsilon/size_a
                else:
                    prob_dict[a] = self.epsilon/size_a
            # choose the updated policy using the probability list
            chosen_pi = choices(list(prob_dict.keys()), list(prob_dict.values()), k=1)[0]
            action = chosen_pi
            next_state = self.get_next_state(state, action)
            self.S_A_R.append((state, action, self.get_reward(next_state)))
            state = next_state
    
    def run_first_visit_pi_evaluation(self):
        seen_S_A_pairs: Set[Tuple[int, int]] = set()
        # run the first-visit Monte Carlo simulation
        for i, (state, action, _) in enumerate(self.S_A_R):
            if (state, action) not in seen_S_A_pairs:
                seen_S_A_pairs.add((state, action))
                self.N_sa[state, action] += 1
                self.Q_sa[state, action] += (self.G_t[i] - self.Q_sa[state, action])/self.N_sa[state, action]
                self.simulated_samples += 1

    def run_every_visit_pi_evaluation(self):
        for i, (state, action, _) in enumerate(self.S_A_R):
            self.N_sa[state, action] += 1
            self.Q_sa[state, action] += (self.G_t[i] - self.Q_sa[state, action])/self.N_sa[state, action]
            self.simulated_samples += 1

    def improve_pi(self):
        # implementing epsilon-greedy policy improvement
        for s, _, r in self.S_A_R:
            self.pi[s] = np.argmax(self.Q_sa[s, ])
            
    
    def run_mc_simulation(self, mode="first-visit"):
        if mode not in {"first-visit", "every-visit"}:
            raise ValueError(f"mode {mode} is not supported")
        
        self.reset_learning()
        # start the simulation        
        if mode == "first-visit":
            delta = np.inf
            while True:
                self.generate_episode()
                self.calculate_returns()
                Q_sa_prev = self.Q_sa.copy()
                self.run_first_visit_pi_evaluation()
                self.improve_pi()
                self.episode += 1
                delta = min(delta, np.max(np.abs(Q_sa_prev - self.Q_sa)))
                if self.episode%100==0:
                    print(f"Delta at episode {self.episode}: {delta}")
                # exit the loop when the model converges
                if delta < 2e-3:
                    break
                self.reset_episode()
        else:
            delta = np.inf
            while True:
                self.generate_episode()
                self.calculate_returns()
                Q_sa_prev = self.Q_sa.copy()
                self.run_every_visit_pi_evaluation()
                self.improve_pi()
                self.episode += 1
                delta = min(delta, np.max(np.abs(Q_sa_prev - self.Q_sa)))
                if self.episode%100==0:
                    print(f"Delta at episode {self.episode}: {delta}")
                # exit the loop when the model converges
                if delta < 2e-3:
                    break
                self.reset_episode()


if __name__ == '__main__':
    # The agent can steer the environment with 90% accuracy, so slip shall be 1-0.9=0.1
    my_env = GridWorld(MY_GRID, random_state=42)
    original_Q_sa = np.full((my_env.state_count, my_env.action_size), -pow(10, 6))
    mc_learner = Monte_Carlo_learner(my_env)
    mc_learner.run_mc_simulation()
    print(f"\"First-visit\" policy evaluation, Simulated episodes: {mc_learner.episode}, Simulated samples: {mc_learner.simulated_samples}")
    updated_el_percentage = (original_Q_sa.size - np.count_nonzero(mc_learner.Q_sa==original_Q_sa))/original_Q_sa.size * 100
    print(f"Percentage of updated elements at Q_sa using first-visit policy evaluation: {round(updated_el_percentage, 2)}%")
    print("------------------------------------------------------------------------------------------")
    mc_learner.run_mc_simulation(mode="every-visit")
    print(f"\"Every-visit\" policy evaluation, Simulated episodes: {mc_learner.episode}, Simulated samples: {mc_learner.simulated_samples}")
    updated_el_percentage = (original_Q_sa.size - np.count_nonzero(mc_learner.Q_sa==original_Q_sa))/original_Q_sa.size * 100
    print(f"Percentage of updated elements at Q_sa using every-visit policy evaluation: {round(updated_el_percentage, 2)}%")