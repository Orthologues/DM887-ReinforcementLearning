from collections import namedtuple, deque
from random import sample

class ReplayMemory(object):

    """
    By default, Transitions.states would be of type torch.Tensor and shape (1, 4, 84, 84)
    """
    Transitions = namedtuple('Transitions',('states', 'action', 'reward', 'next_states', 'done'))

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, **kwargs):
        """Save a transition"""
        self.memory.append(ReplayMemory.Transitions(**kwargs))

    def sample(self, batch_size):
        return sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)