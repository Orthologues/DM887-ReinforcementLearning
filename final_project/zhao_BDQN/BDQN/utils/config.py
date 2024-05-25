from .normalizer import *

import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from typing import Union
from argparse import ArgumentParser
from .replay_memory import *
from ..network import *

class Config:
    
    DEVICE = torch.device("cuda:0") 
    TARGET_NETWORK_UPDATE_FREQ = 5*10**3
    BLR_POSTERIOR_UPDATE_FREQ = 5*10**4
    GD_UPDATE_FREQUENCY = 10 # gradient descent update frequency for the policy Q-network
    PRE_BLR_EXPLORATION_STEPS = 2 * 10**4
    DISCOUNT = 0.99
    ENV_NAME = 'ALE/Breakout-v5'
    CONV_BATCH_SIZE = 32
    STATE_WIDTH = 84
    STATE_HEIGHT = 84
    DEFAULT_OPTIMIZER_FN = lambda params: torch.optim.Adam(
    params, lr=1e-3, betas=(0.9, 0.999), eps=0.01) 
    USE_DQN = True
    REPLAY_HISTORY_LENGTH = 4
    REPLAY_BUFFER = lambda capacity: ReplayMemory(capacity)
    CONV_NETWORK = lambda input_dim: BdqnConvNet(input_dim)
    SIGMA_VARIANCE = 0.001
    NOISE_VARIANCE = 1

    def __init__(self, env_name=ENV_NAME) -> None:

        self.arg_parser = ArgumentParser()

        # boolean values defining the training process
        self.double_q = Config.USE_DQN # whether to use a target Q-network and the policy Q-network

        # defines the evaluation ATARI environment
        self.env_name = env_name
        self.__eval_env = None
        self.set_eval_env()

        # classes/methods
        self.network_fn = Config.CONV_NETWORK # Convolutional neural network for Q-learning
        self.replay_fn = Config.REPLAY_BUFFER # Replay function
        self.conv_optimizer_fn = Config.DEFAULT_OPTIMIZER_FN # Optimizer for the convolutional neural network 
        self.replay_fn = Config.REPLAY_BUFFER # Replay function

        # normalizers for the input batch of tensors tensor of the convolutional neural network
        self.state_normalizer = AtariImageNormalizer(Config.CONV_BATCH_SIZE, Config.STATE_WIDTH, Config.STATE_HEIGHT) # normalizer for the state, i.e., the input tensors of Atari games

        # normalizers for the agent exploration
        self.reward_normalizer = SignNormalizer() # normalizer for the reward

        # params for Q-network update
        self.pre_blr_update_t_steps = Config.PRE_BLR_EXPLORATION_STEPS # Number of time steps without any posterior update
        self.target_network_update_freq = Config.TARGET_NETWORK_UPDATE_FREQ        
        self.gd_update_frequency = Config.GD_UPDATE_FREQUENCY # number of time steps between gradient descent updates
        self.prior_variance = Config.SIGMA_VARIANCE 
        self.noise_variance = Config.NOISE_VARIANCE

        """
        Other parameters related to time steps
        """
        # evaluation interval
        self.eval_interval = 2e5
        self.eval_episodes = 10 # number of episodes to evaluate per $self.eval_interval time steps
        self.n_step = 1 # total number of time steps of the agent
        self.gd_step = 0 # total number of gradient descent time steps of policy Q-network

        self.blr_learn_frequency = Config.BLR_POSTERIOR_UPDATE_FREQ # the frequency of doing posterior update to the bayesian linear regression layer (BLR)


    @property
    def eval_env(self) -> Union[gym.Env, None]:
        return self.__eval_env
    

    @eval_env.setter
    def set_eval_env(self):
        self.__eval_env = gym.make(self.env_name) 
        self.state_dim = self.__eval_env.observation_space.shape
        self.action_count = self.__eval_env.action_space.n if isinstance(self.__eval_env.action_space, Discrete) else None


    def add_argument(self, *args, **kwargs):
        self.arg_parser.add_argument(*args, **kwargs)


    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.arg_parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
