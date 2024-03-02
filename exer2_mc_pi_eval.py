#! /opt/homebrew/anaconda3/bin/python

"""
Author: Jiawei Zhao
Date: 28.02.2024

Steps: 
1) Create a GridWorld environment of size 10x10 or bigger.

2) Initialize your value function to an arbitrarily large value, e.g. J(x) = +inf for all x if you store costs, J(x) = -inf if you store rewards.

3) Perform policy evaluation by implementing two versions of Monte Carlo simulation: a) First-visit, b) Every-visit. Define a convergence criterion, e.g. an epsilon tolerance below which the value of a state is not counted as changed, and run the algorithm until this condition is satisfied. Count how many samples you have taken and how many simulation rounds you have started.

4) Compare the sample and simulation counts for the two algorithm versions.

5) Compare the learned J to the initialized J.
"""

import numpy as np
from gridworld import GridWorld

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

# Monte-Carlo policy evaluation
def monte_carlo_pi_eval(env: GridWorld):
    # tracking the rewards, thus initialize the value function at each element of the array with a negative sign and an arbitrarily large absolute value
    J = np.full((env.state_count, 1), -np.inf)

if __name__ == '__main__':
    # The agent can steer the environment with 90% accuracy, so slip shall be 1-0.9=0.1
    my_env = GridWorld(MY_GRID, slip=0.1, random_state=42)