#! /opt/homebrew/anaconda3/bin/python

"""
Author: Jiawei Zhao
Date: 21.02.2024

Steps: 
1. Install the following package:

https://github.com/prasenjit52282/GridWorld

2. Create a GridWorld environment no smaller than 6x6. Determine the obstacles and the reward structure yourself, potentially seeing the examples given in the package above.

3. Construct the state transition distribution in the following way:

The agent can steer the environment with 90% accuracy. That is, if it chooses the action “move right”, it moves to the right cell with 0.9 probability and to the other cells or stays within its current cell with equal probabilities.

4. Train the agent with following two model-based algorithms:

a) Value iteration
b) Policy iteration with at least three different policy improvement step counts m.

5. Run value iteration and policy iteration with at least three different choices of m on the environment for 5000 time steps. Repeat each experiment five times.

6. Draw the learning curves of all the trained algorithms in one single plot that has the time step in the x axis (which counts for how many steps the agent interacted with the environment) and the collected total reward until that time step in the y axis. Each learning curve should be averaged over the five repetitions.
"""

import numpy as np
from gridworld import GridWorld
from matplotlib import pyplot as plt
from typing import Tuple, Set, List, Dict, Union

"""
Assignment of constants to be used in MDP
"""
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
    ww         ww g  w
    wwwwwwwwwwwwwwwwww
    """
MAX_TIME_STEP: int = 5000
# define the discount factor
GAMMA = 0.9
# In reinforcement learning, "eps" typically refers to epsilon, which is a parameter used in epsilon-greedy exploration strategies
THETA = 1e-3 # the threshold to stop value iteration


"""
The class for value iteration
"""
class Value_iterator():

    def __init__(self, env: GridWorld, theta=THETA, gamma=GAMMA, max_time_step=MAX_TIME_STEP, n_repetition = 5):
        self.env = env
        self.theta = theta
        self.max_time_step = max_time_step
        self.time_step = 0
        self.n_repetition = n_repetition
        self.gamma = gamma
        # tracking the cumulative reward V_s of state transition without given a policy
        self.V_s: np.ndarray[float] = np.zeros((self.env.state_count, 1)) 
        self.viable_cells: List[Tuple[int, int]] = list(self.env.state_dict.keys())
        self.coord_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.viable_cells)}
        # tracking the collected total reward Q_sa per time step
        self.collected_R: List[float] = []

    def reset(self):
        self.V_s = np.zeros((self.env.state_count, 1))
        self.time_step = 0
        self.collected_R: List[float] = []

    def repetitive_iteration(self):
        collected_R_at_all_repetitions: List[np.ndarray[float]] = []
        for _ in range(self.n_repetition):
            self.value_iteration()
            collected_R_at_all_repetitions.append(np.array(self.collected_R))
            self.reset()
        plt.title("Value Iteration")
        plt.xlabel("Time Step")
        plt.ylabel(f"Collected Total Reward averaged by {self.n_repetition} repetitions")
        collected_R_at_all_repetitions = np.array(collected_R_at_all_repetitions)
        averaged_collected_R = np.mean(collected_R_at_all_repetitions, axis=0)
        plt.plot([step+1 for step in range(collected_R_at_all_repetitions.shape[1])], averaged_collected_R, '.b', markersize=0.5)
        # for debugging only
        #plt.show()
        plt.savefig("./figures/exer1/value_iteration.png", dpi=300)

    # 4.a. Construct the state transition distribution in the first algorithm (Value iteration):
    def value_iteration(self):
        # start the MDP process until the V_s converge
        break_flag = False
        # start the MDP process until the V_s converge
        while self.time_step < self.max_time_step:
            delta = 0  #$delta tracks the change
            # iterate through the states
            for state in range(self.env.state_count):
                coord = self.viable_cells[state]
                # skip a state if it is terminal (hole or goal)
                if self.env.state_dict[coord]['done']:
                    continue
                """
                calculate the Q-value given a state and an action (Q is used to determine how good an Action, A, taken at a particular state S is)
                matmul(P_sas, V_PREV) would have a dimension of (env.state, action_count, 1)
                R_sa has a dimension of (state_count, action_count)
                V_s would then have a dimension of (action_count, )
                """
                Q_sa: np.ndarray[Union[int, float]] = np.zeros((self.env.action_size))
                for action in self.env.action_values:
                    action_to_coords: Dict[int, Tuple[int, int]]  = {0: (coord[0]+1, coord[1]), 1: (coord[0], coord[1]+1), 2: (coord[0]-1, coord[1]), 3: (coord[0], coord[1]-1)}
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
                        extra_r = self.extra_states_reward_term(action, action_to_coords, coord)
                        r_sa = self.gamma * (self.env.P_sas[state, action, next_state] * self.V_s[next_state, 0] + extra_r) 
                        Q_sa[action] = self.env.R_sa[state, action] + r_sa
                    else:
                        Q_sa[action] = -np.inf
                # get a backup of the previous Values
                V_prev = self.V_s[state, 0]
                # sort out the action that gives the highest Q value given state S as the updated $V_s
                self.V_s[state, 0] = np.max(Q_sa)
                delta = max(delta, np.abs(self.V_s[state, 0] - V_prev))
                self.time_step += 1
                self.collected_R.append(np.sum(self.V_s))
                # exit the loop when the maximum time step is reached even if the model hasn't converged yet
                if self.time_step == self.max_time_step:
                    print(f"The Value-Iteration model did not converge in {self.max_time_step} time steps, it terminated at the cell {coord}")
                    break_flag = True
                    break
            self.collected_R.append(np.sum(self.V_s))
            if break_flag:
                break
            # exit the loop when the model converges
            if delta < self.theta:
                break
        print(f"The Value-Iteration model ended after {self.time_step} time steps, giving delta={delta}")

    # to be used at value iteration
    def extra_states_reward_term(self, action: int, action_to_coords: Dict[int, Tuple[int, int]], coord: Tuple[int, int]) -> Union[int, float]:
        extra_term = 0
        state = self.coord_to_states[coord]
        if action in action_to_coords:
            extra_coords = [action_to_coords[a] for a in self.env.action_values if a!=action]
            for extra_coord in extra_coords:
                if extra_coord in self.coord_to_states:
                    extra_state = self.coord_to_states[extra_coord]
                    extra_term += self.env.P_sas[state, action, extra_state] * self.V_s[extra_state, 0]
        return extra_term


"""
The class for policy iteration
"""
class Policy_iterator():

    def __init__(self, env: GridWorld, pi_improvement_step_count: int, gamma=GAMMA, theta=THETA, max_time_step=MAX_TIME_STEP, n_repetition = 5, test_mode = False):
        np.random.seed(42)
        self.env = env
        self.theta = theta
        self.max_time_step = max_time_step
        self.n_repetition = n_repetition
        self.pi_improvement_step_count = pi_improvement_step_count
        self.gamma = gamma
        self.time_step = 0
        self.test_mode = test_mode
        # a randomized initial set of pi
        self.pi = np.random.choice(self.env.action_values, size=self.env.state_count)
        # tracking the cumulative reward V_s of state transition given a policy
        self.V_s = np.zeros((self.env.state_count, 1))
        self.viable_cells: List[Tuple[int, int]] = list(self.env.state_dict.keys())
        self.coord_to_states: Dict[Tuple[int, int], int] = {coord: index for index, coord in enumerate(self.viable_cells)}
        # tracking the collected total reward Q_sa per time step
        self.collected_R: List[float] = []

    def reset(self):
        np.random.seed(42)
        self.V_s = np.zeros((self.env.state_count, 1))
        self.pi = np.random.choice(self.env.action_values, size=self.env.state_count)
        self.time_step = 0
        self.collected_R: List[float] = []

    def repetitive_iteration(self):
        collected_R_at_all_repetitions: List[np.ndarray[float]] = []
        for _ in range(self.n_repetition):
            self.policy_iteration()
            collected_R_at_all_repetitions.append(np.array(self.collected_R))
            self.reset()
        plt.title("Policy Iteration")
        plt.xlabel("Time Step")
        plt.ylabel(f"Collected Total Reward averaged by {self.n_repetition} repetitions and policy improvement step {self.pi_improvement_step_count}")
        collected_R_at_all_repetitions = np.array(collected_R_at_all_repetitions)
        averaged_collected_R = np.mean(collected_R_at_all_repetitions, axis=0)
        plt.plot([step+1 for step in range(collected_R_at_all_repetitions.shape[1])], averaged_collected_R, 'b.', markersize=0.5)
        # for debugging only
        #plt.show()
        plt.savefig(f"./figures/exer1/policy_iteration_m_{self.pi_improvement_step_count}.png", dpi=300)

    # perform policy evaluation until the Q function converges
    def policy_evaluation(self):
        while True:
            delta = 0
            for state in range(self.env.state_count):
                coord = self.viable_cells[state]
                # skip a state if it is terminal (hole or goal)
                if self.env.state_dict[coord]['done']:
                    continue
                V_prev = self.V_s
                self.V_s[state, 0] = self.env.R_sa[state, self.pi[state]]
                next_coord_set: Set[Tuple[int, int]] = {(coord[0]+1, coord[1]), (coord[0], coord[1]+1), (coord[0]-1, coord[1]), (coord[0], coord[1]-1)}
                for next_coord in next_coord_set:
                    if next_coord in self.coord_to_states:
                        self.V_s[state, 0] += self.gamma * self.env.P_sas[state, self.pi[state], self.coord_to_states[next_coord]] * V_prev[self.coord_to_states[next_coord], 0]
                delta = max(delta, np.abs(self.V_s[state, 0] - V_prev[state, 0]))    
            # return to a converged Value function
            if delta < self.theta:
                break
                

    # 4.b. Construct the state transition distribution in the second algorithm (Policy iteration with at least three different policy improvement step counts m):
    def policy_iteration(self):
        break_flag = False
        # start policy iteration
        while self.time_step < self.max_time_step:
            # update the value function at each time step
            self.policy_evaluation()
            pi_prev = self.pi.copy()
            for state in range(self.env.state_count):
                coord = self.viable_cells[state]
                # skip a state if it is terminal (hole or goal)
                if self.env.state_dict[coord]['done']:
                    continue
                # perform policy improvement once every time step m
                if self.time_step % self.pi_improvement_step_count==0:
                    # Implementation of policy improvement
                    Q_a = np.zeros((self.env.action_size, 1))
                    for action in self.env.action_values:
                        action_to_coords: Dict[int, Tuple[int, int]]  = {0: (coord[0]+1, coord[1]), 1: (coord[0], coord[1]+1), 2: (coord[0]-1, coord[1]), 3: (coord[0], coord[1]-1)}
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
                            extra_r = self.extra_states_reward_term(action, action_to_coords, coord)
                            r_sa = self.gamma * (self.env.P_sas[state, action, next_state] * self.V_s[next_state, 0] + extra_r) 
                            Q_a[action] = self.env.R_sa[state, action] + r_sa
                        else:
                            Q_a[action] = -np.inf
                        self.pi[state] = np.argmax(Q_a)
                        # end of policy improvement
                self.time_step += 1
                self.collected_R.append(np.sum(self.V_s))
                # exit the loop when the maximum time step is reached even if the model hasn't converged yet
                if self.time_step == self.max_time_step:
                    print(f"The Policy-Iteration model did not converge in {self.max_time_step} time steps, it terminated at the cell {coord}")
                    break_flag = True
                    break
            if break_flag:
                break
            # condition of exiting the loop: there is no longer change of pi
            if np.sum(np.abs(self.pi-pi_prev), keepdims=False)==0:
                break

        print(f"The Policy-Iteration model converged after {self.time_step} time steps")


    # to be used at policy improvement
    def extra_states_reward_term(self, action: int, action_to_coords: Dict[int, Tuple[int, int]], coord: Tuple[int, int]) -> Union[int, float]:
        extra_term = 0
        state = self.coord_to_states[coord]
        if action in action_to_coords:
            extra_coords = [action_to_coords[a] for a in self.env.action_values if a!=action]
            for extra_coord in extra_coords:
                if extra_coord in self.coord_to_states:
                    extra_state = self.coord_to_states[extra_coord]
                    extra_term += self.env.P_sas[state, action, extra_state] * self.V_s[extra_state, 0]
        return extra_term


"""
Training in both aforementioned DP algorithms
"""
if __name__ == '__main__':
    # The agent can steer the environment with 90% accuracy, so slip shall be 1-0.9=0.1
    my_env = GridWorld(MY_GRID, slip=0.1, random_state=42)

    # Algorithm 1
    print(f"Value Iteration of maximum {MAX_TIME_STEP} time steps")
    v_iterator = Value_iterator(my_env)
    v_iterator.repetitive_iteration()

    # Algorithm 2
    for m in [1, 2, 3]:
        print(f"Policy Iteration of maximum {MAX_TIME_STEP} time steps with a policy improvement step count {m}: ")
        pi_iterator = Policy_iterator(my_env, pi_improvement_step_count = m)
        pi_iterator.repetitive_iteration()
