#! /opt/homebrew/anaconda3/bin/python

"""
Author: Jiawei Zhao
Date: 28.02.2024

Steps: 
1) Create a GridWorld environment of size 10x10 or bigger.

2) Initialize your value function to an arbitrarily large value, e.g. J(x) = +inf for all x if you store returnss, J(x) = -inf if you store rewards.

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
    w   o   w  w   a w
    w       w        w
    www  o  www    www
    w  o        o    w
    wwwww    o     www
    w     wwwwww     w
    w     w    w     w
    ww         ww   gw
    wwwwwwwwwwwwwwwwww
    """
EPISILON = 1e-4
GAMMA = 0.9

# Monte-Carlo policy evaluation
class Monte_Carlo_learner:

    # epsilon represents the probability that an agent is going to take a random action
    def __init__(self, env: GridWorld, gamma=GAMMA, epsilon = 1e-2):
        np.random.seed(42)
        self.gamma = gamma
        self.env = env
        self.simulated_samples = 0
        self.simulatd_episodes = 0
        self.epsilon = epsilon
        self.Q_sa: np.ndarray[float] = np.full((env.state_count, env.action_size), -pow(10, 7)) # tracking cumulative rewards
        self.N_sa: np.ndarray[int] = np.zeros((env.state_count, env.action_size)) # tracking numbers of visits to each state-action pair # tracking the numbers that each state-action pair has been visited in an episode
        self.S_A_R: List[Tuple[int, int, int]] = [] # tracking the state-action-reward tuples that have been visited in an episode
        self.G_t: List[float] = [] # tracking the cumulative returns at each step per episode
        self.pi = np.random.choice(env.action_values, size=env.state_count)
        self.viable_cells: List[Tuple[int, int]] = list(self.env.state_dict.keys())
        self.coord_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.viable_cells)}

    def get_next_state(self, state: int, action: int) -> int:
        coord = self.viable_cells[state]
        next_coord: Tuple[Union[int, float], Union[int, float]] = (np.inf, np.inf)
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
        if next_coord in self.coord_to_states:
            next_state = self.coord_to_states[next_coord]
            return next_state
        return state

    def get_reward(self, state: int) -> int:
        # get the reward by state_dict of the grid environment
        return self.env.state_dict[self.viable_cells[state]]['reward']
    
    def calculate_returns(self):
        for i in range(len(self.S_A_R)):
            returns = 0
            for k, (_, _, reward) in enumerate(self.S_A_R[i+1: ]):
                returns += reward * pow(self.gamma, k)
            self.G_t.append(returns)
    
    def reset_learning(self):
        np.random.seed(42)
        self.Q_sa: np.ndarray[float] = np.full((self.env.state_count, self.env.action_size), -pow(10, 7)) 
        self.N_sa: np.ndarray[int] = np.zeros((self.env.state_count, self.env.action_size)) 
        self.pi = np.random.choice(self.env.action_values, size=self.env.state_count)

    def reset_episode(self):
        self.S_A_R = []
        self.G_t = []

    def generate_episode(self):
        state = 0
        while True:
            if self.env.state_dict[self.viable_cells[state]]['done']:
                break
            action = np.random.choice(self.env.action_values)
            next_state = self.get_next_state(state, action)
            if next_state == state:
                continue
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
            optimal_a = np.argmax(self.Q_sa[s, ])
            # get a set of all viable actions
            viable_a: Set[int] = set()
            for a in self.env.action_values:
                if self.get_next_state(s, a) != s:
                    viable_a.add(a)
            # create a probability dictionary with epsilon
            prob_dict: Dict[int, float] = dict()
            size_a = len(viable_a)
            for a in viable_a:
                if a == optimal_a:
                    prob_dict[a] = 1 - self.epsilon + self.epsilon/size_a
                else:
                    prob_dict[a] = self.epsilon/size_a
            # choose the updated policy using the probability list
            updated_pi = choices(list(prob_dict.keys()), list(prob_dict.values()), k=1)[0]
            self.pi[s] = updated_pi
    
    def run_mc_simulation(self, mode="first-visit"):

        self.reset_learning()
        
        if mode not in {"first-visit", "every-visit"}:
            raise ValueError(f"mode {mode} is not supported")
        
        # start the simulation        
        if mode == "first-visit":
            jackpot_count = 0
            while True:
                self.generate_episode()
                self.calculate_returns()
                Q_sa_prev = self.Q_sa.copy()
                self.run_first_visit_pi_evaluation()
                self.improve_pi()
                self.simulatd_episodes += 1
                delta = np.max(np.abs(Q_sa_prev - self.Q_sa))
                if self.S_A_R[-1][2] == 100:
                    jackpot_count += 1
                # exit the loop when the model converges
                if delta < self.epsilon*0.1 and jackpot_count >= 10:
                    break
                self.reset_episode()
        else:
            jackpot_count = 0
            while True:
                self.generate_episode()
                self.calculate_returns()
                Q_sa_prev = self.Q_sa.copy()
                self.run_every_visit_pi_evaluation()
                self.improve_pi()
                self.simulatd_episodes += 1
                delta = np.max(np.abs(Q_sa_prev - self.Q_sa))
                if self.S_A_R[-1][2] == 100:
                    jackpot_count += 1
                # exit the loop when the model converges
                if delta < self.epsilon*0.1 and jackpot_count >= 10:
                    break
                self.reset_episode()
        print(f"The number of episodes hitting the goal: {jackpot_count}")


if __name__ == '__main__':
    # The agent can steer the environment with 90% accuracy, so slip shall be 1-0.9=0.1
    my_env = GridWorld(MY_GRID, slip=0.1, random_state=42)
    original_Q_sa = np.full((my_env.state_count, my_env.action_size), -pow(10, 7))
    mc_learner = Monte_Carlo_learner(my_env)
    mc_learner.run_mc_simulation()
    print(f"\"First-visit\" policy evaluation, Simulated episodes: {mc_learner.simulatd_episodes}, Simulated samples: {mc_learner.simulated_samples}")
    updated_el_percentage = (original_Q_sa.size - np.count_nonzero(mc_learner.Q_sa==original_Q_sa))/original_Q_sa.size * 100
    print(f"Percentage of updated elements at Q_sa using first-visit policy evaluation: {round(updated_el_percentage, 2)}%")
    print("------------------------------------------------------------------------------------------")
    mc_learner.run_mc_simulation(mode="every-visit")
    print(f"\"Every-visit\" policy evaluation, Simulated episodes: {mc_learner.simulatd_episodes}, Simulated samples: {mc_learner.simulated_samples}")
    updated_el_percentage = (original_Q_sa.size - np.count_nonzero(mc_learner.Q_sa==original_Q_sa))/original_Q_sa.size * 100
    print(f"Percentage of updated elements at Q_sa using every-visit policy evaluation: {round(updated_el_percentage, 2)}%")