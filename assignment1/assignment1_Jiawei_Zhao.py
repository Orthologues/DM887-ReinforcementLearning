#! /opt/homebrew/anaconda3/bin/python

from typing import Tuple, List, Dict, Union, Set
from gridworld import GridWorld
import numpy as np
from random import choices
from math import pow, floor
from matplotlib import pyplot as plt
from re import sub


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
    wa              o  w
    w                  w
    www     www      www
    w                  w
    w             o    w
    w                  w
    w                  w
    w       www        w
    w         w    o   w
    w         w        w
    w         w        w
    w                  w
    w                  w
    w                  w
    wwww    o         ww
    w                  w
    w  o               w
    w              g   w
    wwwwwwwwwwwwwwwwwwww
    """

# adding lots of holes on the shortest paths from the agent to the goal
GRID_B: str=\
    """
    wwwwwwwwwwwwwwwwwwww
    wa o    w          w
    w      w          w
    www  o  www      www
    w       o          w
    w         o o      w
    w          o o     w
    w           o o   w
    w       www   o    w
    w         wooo o o w
    w         w    o   w
    w        ow    o   w
    w      oooo    o   w
    w         o    o   w
    w         o    o   w
    wwww           oowww
    w          w   o   w
    w          w  oo   w
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
    change the holes to be non-terminal
    place the holes on the shortest paths to the goal
    change the immediate cost of falling into a hole to a much smaller value than 100,
    to lure the agent to choose a shorter-path for higher immediate rewards
    """
    for v in env_b.state_dict.values():
        if v['type'] == 'hole':
            v['reward'] = -5
            v['done'] = False
    return env_a, env_b

"""
Q2. [7 points] Train the learning agent using N-step Q-learning for N=1, 2, 3
in tabular form that uses an epsilon-greedy behavior policy with epsilon=0.1, 0.2, 0.3.
Repeat the whole training process ten times.
"""

eps_list = [0.1, 0.2, 0.3]
n_list = [1, 2, 3]
n_eps_pairs = [(n, eps) for n in n_list for eps in eps_list ]

class Q_learning_trainer():

    def __init__(self, env: GridWorld, epsilon, n_step, alpha=0.2, gamma=0.9, max_round=10, max_episode=500, plotting_n_step=5):
        np.random.seed(42)
        self.env = env
        self.n_step = n_step
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.max_round = max_round
        self.max_episode = max_episode # the criterion for convergence of the Q-table
        self.plotting_n_step = plotting_n_step
        self.Q_mean = np.zeros(shape=(max_round, floor(max_episode/plotting_n_step)))
        self.states_to_coords: List[Tuple[int, int]] = list(env.state_dict.keys())
        self.coords_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.states_to_coords)}
        self.viable_actions: Dict[int, Set[int]] = dict()
        # Q-table that tracks the expected future costs to be minimized, so the initial values are set to be very large
        self.Q_sa: np.ndarray[float] = np.full((self.env.state_count, self.env.action_size), 1000) + np.random.normal(0, 5, (self.env.state_count, self.env.action_size))
        self.pi = np.zeros(shape=self.env.state_count)
        self.initialize_random_policies()

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
    

    def choose_action_by_greedy_epsilon(self, state: int) -> int:
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


    def run_episode(self, initial_s: int):
        state = initial_s
        t = 0 # time_step
        T = np.inf # total time step of the episode
        while True:
            if t < T:
                # if t exceeds 1000, the agent is considered as being trapped in a circular path that never reaches a terminal state
                action = self.choose_action_by_greedy_epsilon(state) if t < 1000 else np.random.choice(list(self.viable_actions[state]))
                next_state = self.get_next_state(state, action)

                if self.env.state_dict[self.states_to_coords[next_state]]['done']:
                    T = t+1
            
            tau = t - self.n_step + 1 # represents the time step whose state's Q is being updated
            
            # update Q table using self.n_step
            if tau >= 0:
                delta = self.get_n_step_delta(state, action, next_state, tau, T)
                self.Q_sa[state, action] += self.alpha * delta
                self.pi[state] = np.argmin(self.Q_sa[state, ])
            
            if tau == T-1:
                break
            
            state = next_state
            t += 1                



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


    def reset_training(self):
        np.random.seed(42)
        # Q-table that tracks the expected future costs to be minimized, so the initial values are set to be very large
        self.Q_sa: np.ndarray[float] = np.full((self.env.state_count, self.env.action_size), 1000) + np.random.normal(0, 5, (self.env.state_count, self.env.action_size))
        self.pi = np.zeros(shape=self.env.state_count)
        self.initialize_random_policies()


    def reset_Q_array(self):
        self.Q_mean = np.zeros(shape=(self.max_round, floor(self.max_episode/self.plotting_n_step)))


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


    def update_Q(self, round: int, episode: int):
        if (episode-1) % self.plotting_n_step == 0:
            self.Q_mean[round-1, int(episode/self.plotting_n_step)] = np.mean(self.Q_sa)

    def get_n_step_delta(self, state: int, action: int, next_state: int, tau: int, T: int) -> float:
        G = 0
        s = next_state
        for i in range(tau+1, min(tau+self.n_step+1, T)):
            cost = self.get_immediate_cost(s)
            G += pow(self.gamma, i-tau-1) * cost
            eps_greedy_a = self.choose_action_by_greedy_epsilon(s)
            s = self.get_next_state(s, eps_greedy_a)

        if tau+self.n_step < T:
            # we use the cost function here, so np.min is chosen instead of np.max
            G += pow(self.gamma, self.n_step) * np.min(self.Q_sa[s, ])
        
        return G - self.Q_sa[state, action]
    

    def run_N_step_Q_learning(self) -> List[Tuple[float, float]]:
        for round in range(1, self.max_round+1):
            self.reset_training()
            for e in range(1, self.max_episode+1):
                #print(f"N={self.n_step}, eps={self.epsilon}, Episode {e}")
                self.run_episode(initial_s=0)
                self.update_Q(round, episode=e)
        
        mean_stderr_dataset: List[Tuple[float, float]] = []
        averaged_Q = np.mean(self.Q_mean, axis=0)
        stderr_Q = np.std(self.Q_mean, axis=0)/np.sqrt(self.Q_mean.shape[0])
        for (mean, stderr) in zip(averaged_Q, stderr_Q):
            mean_stderr_dataset.append((mean, stderr))

        self.reset_Q_array()

        return mean_stderr_dataset


def run_game(env, game_name, n, eps):
    learner = Q_learning_trainer(env, n_step=n, epsilon=eps)
    mean_stderr_dataset: List[Tuple[float, float]] = []    
    mean_stderr_dataset = learner.run_N_step_Q_learning()
    alt_eps = sub("\.", "-", str(eps))
    file_prefix = f"n_{n}_eps_{alt_eps}_{game_name}"
    np.savetxt(f"{file_prefix}.csv",
        mean_stderr_dataset,
        delimiter =", ",
        fmt ='% s')


# the plotting function 
def plot_Q_mean_and_stderr(datasets: List[List[Tuple[float, float]]], data_legends: List[str], fname_suffix: str, plotting_n_step=5):

    assert(len(datasets) == len(data_legends))

    # Define colormap for generating unique colors
    colors = plt.get_cmap("tab10")
    # Create the plot
    fig, ax = plt.subplots(figsize=(16, 9))
    mean_lines = []

    # Plot lines for mean values
    for i, dataset in enumerate(datasets):
        episode = [ i*plotting_n_step for i in range(0, len(dataset)) ]
        mean = [ el[0] for el in dataset ]
        color = colors(i/len(datasets))  # Get unique color from colormap
        mean_line, = ax.plot(episode, mean, color=color)
        mean_lines.append(mean_line)
        # Fill between mean +/- standard error with semi-transparent color
        mean_upper = [ el[0] + el[1] for el in dataset ]
        mean_lower = [ el[0] - el[1] for el in dataset ]
        ax.fill_between(episode, mean_upper, mean_lower, alpha=0.2, color=color)
        # Add labels and title
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Cumulative Cost")
        ax.set_title(f"Epsilon-greedy N-step Q-learning ({fname_suffix})")

    # Add legend
    ax.legend(mean_lines, data_legends, loc="upper right")

    fig.savefig(f"learning-curve-{fname_suffix}.png", dpi=150, bbox_inches='tight')    


# run training and plotting at the main function
if __name__ == "__main__":

    env_a, env_b = define_two_grid_worlds()

    for n, eps in n_eps_pairs:
        run_game(env_a, "a", n, eps)
        run_game(env_b, "b", n, eps)


    # read the training record of game A in CSV 
    learning_curves, legends = [], []
    for n, eps in n_eps_pairs:
        alt_eps = sub("\.", "-", str(eps))
        arr = np.loadtxt(f"n_{n}_eps_{alt_eps}_a.csv", delimiter=",", dtype=float)
        learning_curve: List[Tuple[float, float]] = [(el[0], el[1]) for el in arr]
        learning_curves.append(learning_curve)
        legends.append(f"N={n}, eps={eps}")
        plot_Q_mean_and_stderr(datasets=learning_curves, data_legends=legends, fname_suffix='game_a')


    # read the training record of game B in CSV 
    learning_curves, legends = [], []
    for n, eps in n_eps_pairs:
        arr = np.loadtxt(f"n_{n}_eps_{alt_eps}_b.csv", delimiter=",", dtype=float)
        learning_curve: List[Tuple[float, float]] = [(el[0], el[1]) for el in arr]
        learning_curves.append(learning_curve)
        legends.append(f"N={n}, eps={eps}")
        plot_Q_mean_and_stderr(datasets=learning_curves, data_legends=legends, fname_suffix='game_b')
    


    