import matplotlib.pyplot as plt
import torch
from Q2 import LSTD_DQL_learner

"""
Author: Jiawei Zhao
Date: 19.03.2024
Assignment 2 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
"""

# use GPU for training if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
