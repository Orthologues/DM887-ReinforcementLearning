#! /opt/homebrew/anaconda3/bin/python

from typing import Tuple, List, Dict, Union, Set
from gridworld import GridWorld
import numpy as np
from random import choices
from math import pow, floor
from matplotlib import pyplot as plt


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
    wa    o  w      o  w
    w        w         w
    www     www      www
    w                  w
    w             o    w
    w                  w
    w             o    w
    w o     wwwww      w
    w         w    o   w
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
    w   o     w        w
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
    def __init__(self, env: GridWorld, epsilon=0.05, n_step=1, alpha=0.05, gamma=0.9, max_round=10, max_episode=5000, plotting_n_step=5):
        self.env = env
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_round = max_round
        self.max_episode = max_episode # the criterion for convergence of the Q-table
        self.plotting_n_step = plotting_n_step
        # tracking the returns (cost) per time step at each episode to update the Q-table
        self.G_t: np.ndarray[float] = []
        # V array
        self.V_s: np.ndarray[Union[int, float]] = np.zeros(shape=env.state_count)
        # Q-table that tracks the expected future costs to be minimized, so the initial values are set to be very large
        self.Q_sa: np.ndarray[float] = np.full((env.state_count, env.action_size), pow(10, 6)) 
        self.S_A_R: List[Tuple[int, int, int]] = []
        # tracking the behaviour policies
        self.Q_sa_mean = np.zeros(shape=(max_round, floor(max_episode/plotting_n_step)))
        self.Q_sa_stderr = np.zeros(shape=(max_round, floor(max_episode/plotting_n_step))) 
        self.pi = np.zeros(shape=env.state_count)
        self.states_to_coords: List[Tuple[int, int]] = list(env.state_dict.keys())
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
            self.pi[s] = np.random.choice(list(viable_a))
            self.viable_actions[s] = viable_a
    
    def choose_action_by_epsilon_greedy(self, state: int) -> int:
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
        return choices(list(prob_dict.keys()), weights=list(prob_dict.values()), k=1)[0]

    def generate_episode(self, initial_s: int, episode: int):
        if episode == 1:
            self.initialize_random_policies()
        state = initial_s
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

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def set_gamma(self, gamma: float):
        self.gamma = gamma

    def set_epsilon(self, epsilon: float):
        self.epsilon = epsilon

    def set_max_round(self, max_round: int):
        self.max_round = max_round

    def set_n_step(self, n_step: int):
        self.n_step = n_step

    def set_plotting_n_step(self, plotting_n_step: int):
        self.plotting_n_step = plotting_n_step
    
    def set_max_episode(self, max_episode: int):
        self.max_episode = max_episode

    def start_new_episode(self):
        self.G_t = []

    def start_training(self):
        np.random.seed(42)
        self.Q_sa: np.ndarray[float] = np.full((self.env.state_count, self.env.action_size), pow(10, 7)) 
        self.V_s: np.ndarray[Union[int, float]] = np.zeros(shape=self.env.state_count)
        self.pi = np.random.choice(self.env.action_values, size=self.env.state_count)

    def reset_mean_and_stderr_array(self):
        self.Q_sa_mean = np.zeros(shape=(self.max_round, floor(self.max_episode/self.plotting_n_step)))
        self.Q_sa_stderr = np.zeros(shape=(self.max_round, floor(self.max_episode/self.plotting_n_step))) 

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

    def get_reward(self, state: int) -> int:
        # get the reward by state_dict of the grid environment
        return self.env.state_dict[self.states_to_coords[state]]['reward']

    def get_immediate_cost(self, state: int) -> int:
        # get the negative immediate reward by state_dict of the grid env
        return -self.env.state_dict[self.states_to_coords[state]]['reward']
    
    def update_Q_table_mean_and_stderr(self, round: int, episode: int):
        if episode % self.plotting_n_step == 0:
            self.Q_sa_mean[round-1, episode-1] = np.mean(self.Q_sa)
            self.Q_sa_stderr[round-1, episode-1] = np.std(self.Q_sa) / np.sqrt(np.size)

    def calculate_N_step_returns(self):
        pass
        # calculate the discounted future costs of the last N 

    def update_Q_table(self):
        pass

    def update_behaviour_pi(self, s: int):
        # base it on the epsilon-greedy policy
        self.pi[s] = np.argmax(self.Q_sa[s, ])

    def run_N_step_Q_learning(self) -> List[Tuple[float, float]]:
        for round in range(1, self.max_round+1):
            self.start_training()
            for e in range(1, self.max_episode+1):
                self.start_new_episode()
                self.generate_episode(initial_s=0, episode=e)
                self.update_Q_table()
                self.update_Q_table_mean_and_stderr(round, e)
        
        mean_stderr_dataset: List[Tuple[float, float]] = []
        multi_rounds_averaged_mean = np.mean(self.Q_sa_mean, axis=0)
        multi_rounds_averaged_stderr = np.mean(self.Q_sa_stderr, axis=0)
        for (mean, stderr) in zip(multi_rounds_averaged_mean, multi_rounds_averaged_stderr):
            mean_stderr_dataset.append((mean, stderr))

        self.reset_mean_and_stderr_array()

        return mean_stderr_dataset


# the plotting function 
def plot_Q_table_mean_and_stderr(datasets: List[List[Tuple[float, float]]], data_legends: List[str], fname_suffix: str, plotting_n_step=5):

    assert(len(datasets) == len(data_legends))

    # Define colormap for generating unique colors
    colors = plt.get_cmap("tab10")
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))

    # Plot lines for mean values
    for i, dataset in enumerate(datasets):
        episode = [ i*plotting_n_step for i in range(0, len(dataset)) ]
        mean = [ el[0] for el in dataset ]
        color = colors(i/len(datasets))  # Get unique color from colormap

        ax.plot(episode, mean, color=color)
        # Fill between mean +/- standard error with semi-transparent color
        mean_upper = [ el[0] + el[1] for el in dataset ]
        mean_lower = [ el[0] - el[1] for el in dataset ]
        ax.fill_between(episode, mean_upper, mean_lower, alpha=0.2, color=color)
        
        # Add labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Mean Cumulative Cost")
        ax.set_title("Epsilon-greedy N-step Q-learning")

    # Add legend
    ax.legend(data_legends, loc="upper right")

    fig.savefig(f"learning-curve-{fname_suffix}.png", dpi=150, bbox_inches='tight')



# run training and plotting at the main function
if __name__ == "__main__":

    eps_list = [0.1, 0.2, 0.3]
    N_list = [1, 2, 3]

    env_a, env_b = define_two_grid_worlds()

    learner_a = Q_learning_trainer(env=env_a)
    mean_stderr_datasets: List[List[Tuple[float, float]]] = []
    data_legends: List[str] = []
    for N in N_list:
        for eps in eps_list:
            learner_a.set_epsilon(eps)
            learner_a.set_n_step(N)
            mean_stderr_dataset = learner_a.run_N_step_Q_learning()
            mean_stderr_datasets.append(mean_stderr_dataset)
            data_legends.append(f"N={N}, eps={eps}")
    plot_Q_table_mean_and_stderr(datasets=mean_stderr_datasets, data_legends=data_legends, fname_suffix="GameA")


    learner_b = Q_learning_trainer(env=env_b)
    mean_stderr_datasets = []
    data_legends = []
    for N in N_list:
        for eps in eps_list:
            learner_b.set_epsilon(eps)
            learner_b.set_n_step(N)
            mean_stderr_dataset = learner_b.run_N_step_Q_learning()
            mean_stderr_datasets.append(mean_stderr_dataset)
            data_legends.append(f"N={N}, eps={eps}")
    plot_Q_table_mean_and_stderr(datasets=mean_stderr_datasets, data_legends=data_legends, fname_suffix="GameB")