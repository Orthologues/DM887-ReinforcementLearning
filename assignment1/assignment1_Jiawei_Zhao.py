#! /opt/homebrew/anaconda3/bin/python

from typing import Tuple, List, Dict, Union, Set
from gridworld import GridWorld
import numpy as np
from random import choices
from math import pow


"""
Author: Jiawei Zhao
Date: 05.03.2024
Assignment 1 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
"""

"""
Q1. [4 points] Design two GridWorld games of grid size 20x20. 
Game A does not need to fulfill extra specific requirements.
Game B needs to fulfill the following requirements:
    It can be played well if an agent learns to sacrifice immediate rewards to achieve a larger delayed reward
"""

GRID_A: str=\
    """
    wwwwwwwwwwwwwwwwwwww
    wa       w         w
    w       w          w
    www     www      www
    w                  w
    w    o             w
    w                  w
    w                  w
    w       wwwww      w
    w         w o      w
    w         w        w
    w         w        w
    w                  w
    w                  w
    w        o         w
    wwww             www
    w          w       w
    w    g     w       w
    w          w       w
    wwwwwwwwwwwwwwwwwwww
    """

# adding lots of holes on the shortest paths from the agent to the goal
GRID_B: str=\
    """
    wwwwwwwwwwwwwwwwwwww
    wa      w          w
    w       w          w
    www  o  www      www
    w       o          w
    w    o      o      w
    w      o   o       w
    w        o     o   w
    w   o   wwwww      w
    w         w     o  w
    w   o o   w        w
    w        ow        w
    w    o             w
    w                  w
    w                  w
    wwww             www
    w          w       w
    w          w       w
    w          w      gw
    wwwwwwwwwwwwwwwwwwww
    """

def define_two_grid_worlds() -> Tuple[GridWorld, GridWorld]:
    # create env a using default immediate rewards
    env_a = GridWorld(GRID_A, random_state=42)
    # initialize env b, then edit its rewards
    env_b = GridWorld(GRID_B, random_state=42)
    """
    Edit env_b.state_dict
    change the immediate cost of falling into a hole to a much smaller value than 100,
    to lure the agent to fall into a hole for higher immediate rewards
    """
    for v in env_b.state_dict.values():
        if v['type'] == 'hole':
            v['reward'] = -5.0
    return env_a, env_b

"""
Q2. [7 points] Train the learning agent using N-step Q-learning for N=1, 2, 3
in tabular form that uses an epsilon-greedy behavior policy with epsilon=0.1, 0.2, 0.3.
Repeat the whole training process ten times.
"""
class Q_learning_trainer():
    def __init__(self, env: GridWorld, epsilon: float, n_step: int, alpha=0.01, gamma=0.9, repetitions=10, plotting_n_step=5):
        self.env = env
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.repetitions = repetitions
        self.plotting_n_step = plotting_n_step        
        # tracking the returns (cost) per time step at each episode to update the Q-table
        self.G_t: np.ndarray[float] = []
        # Q-table that tracks the expected future costs to be minimized
        self.Q_sa: np.ndarray[float] = np.full((env.state_count, env.action_size), pow(10, 7)) 
        # tracking the behaviour policies
        self.episode = 0
        self.time_step = 0
        self.Q_sa_mean_and_stderr: List[Tuple[float, float]] = []
        self.bpi = np.zeros(shape=env.state_count)
        self.states_to_coords: List[Tuple[int, int]] = list(self.env.state_dict.keys())
        self.coords_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.states_to_coords)}
        self.viable_actions: Dict[int, Set[int]] = dict()

    def initialize_random_policies(self):
        for s in range(self.env.state_count):
            viable_a: Set[int] = set()
            for a in self.env.action_values:
                next_s = self.get_next_state(s, a)
                if next_s == s:
                    continue
                viable_a.add(a)
            self.bpi[s] = np.random.choice(list(viable_a))
            self.viable_actions[s] = viable_a
    
    def generate_episode(self):
        if self.episode == 0:
            self.initialize_random_policies()
        pass

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_repetitions(self, repetitions: int):
        self.repetitions = repetitions

    def set_plotting_n_step(self, plotting_n_step: int):
        self.plotting_n_step = plotting_n_step

    def start_new_episode(self):
        self.episode += 1
        self.G_t = []

    def start_training(self):
        np.random.seed(42)
        self.Q_sa: np.ndarray[float] = np.full((self.state_count, self.action_size), pow(10, 7)) 
        self.bpi = np.random.choice(self.env.action_values, size=self.env.state_count)
        self.Q_sa_mean_and_stderr = []

    def get_next_state(self, state: int, action: int) -> int:
        coord = self.states_to_coords[state]
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
        if next_coord in self.coords_to_states:
            next_state = self.coords_to_states[next_coord]
            return next_state
        return state

    def get_immediate_cost(self, state: int) -> int:
        # get the negative immediate reward by state_dict of the grid env
        return -self.env.state_dict[self.states_to_coords[state]]['reward']
    
    def add_Q_table_mean_and_stderr(self):
        if self.episode % self.plotting_n_step == 0:
            mean_stderr_pair = (np.mean(self.Q_sa), np.std(self.Q_sa) / np.sqrt(np.size))
            self.Q_sa_mean_and_stderr.append(mean_stderr_pair)

    def calculate_N_step_returns(self):
        pass
        # calculate the discounted future costs of the last N 

    def update_Q_table(self):
        pass

    def update_behaviour_pi(self, s: int):
        # base it on the epsilon-greedy policy
        self.bpi[s] = np.argmax(self.Q_sa[s, ])

    def run_N_step_Q_learning(self):
        pass

# the plotting function 
def plot_Q_table_mean_and_stderr(list_of_mean_stderr_pairs: List[List[Tuple[float, float]]], episode_step: int):
    pass

# run training and plotting at the main function
if __name__ == "__main__":
    env_a, env_b = define_two_grid_worlds()