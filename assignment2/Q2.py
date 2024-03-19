import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


"""
Author: Jiawei Zhao
Date: 19.03.2024
Assignment 2 of DM887 (Reinforcement learning)
Lecturer: Melih Kandemir
References
1. https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

class ReplayMemory(object):

    def __init__(self, capacity: int, batch_size: int):
        self.transition = namedtuple('transition', ('state', 'action', 'reward', 'state_next'))
        self.batch_size = batch_size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition to the buffer"""
        self.memory.append(self.transition(*args))

    def sample(self):
        return random.sample(self.memory, self.batch_size)

    def __len__(self):
        return len(self.memory)


class LSTD_DQL_trainer():

    def __init__(self, capacity: int, batch_size: int=32):
        self.replay_buffer= deque([], maxlen=capacity, batch_size=batch_size)