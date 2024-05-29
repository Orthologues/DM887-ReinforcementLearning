from .normalizer import *

import torch
import gymnasium as gym
from gymnasium.spaces import Discrete
from argparse import ArgumentParser
from .replay_memory import *
from ..network import *

class Config:
    
    DEVICE = "cuda:0"
    """
    The frequency of a posterior update is 
    1/(Config.TARGET_NETWORK_UPDATE_INTERVAL * Config.TARGET_WEIGHT_UPDATE_INTERVAL) = 1/2000
    """
    TARGET_NETWORK_UPDATE_INTERVAL = 2500
    TARGET_WEIGHT_UPDATE_INTERVAL = 2
    GD_UPDATE_INTERVAL = 10 # gradient descent update frequency for the policy Q-network
    WARMUP_STEPS = 2 * 10**4
    THOMPSON_SAMPLING_INTERVAL = 10**3
    DISCOUNT = 0.99
    CONV_BATCH_SIZE = 32
    STATE_WIDTH = 84
    STATE_HEIGHT = 84
    DEFAULT_OPTIMIZER_FN = lambda params: torch.optim.Adam(
    params, lr=2.5e-3, betas=(0.9, 0.999), eps=0.01) 
    REPLAY_HISTORY_LENGTH = 5
    REPLAY_BUFFER = lambda capacity: ReplayMemory(capacity)
    CONV_NETWORK = lambda input_dim: BdqnConvNet(input_dim)
    SIGMA_VARIANCE = 0.001
    NOISE_VARIANCE = 1
    USE_SOFTMAX_POLICY = True
    SKIP_FRAMES = False
    FRAMES_TO_SKIP = 4
    MAX_EPISODAL_TIME_STEPS = 2500 
    MAX_BLR_BATCH_SIZE = 10**5
    MAX_TRAINING_EPISODE = 4000
    REPLAY_SIZE = 2 * 10**5
    CLIP_REWARDS = False

    def __init__(self, env_name: str, use_max_episodal_t_steps = True) -> None:

        self.arg_parser = ArgumentParser()

        # configurations wrt ATARI environment
        self.env_name = env_name
        self.__eval_env = gym.make(self.env_name) 
        self.state_dim = self.__eval_env.observation_space.shape
        self.action_count = self.__eval_env.action_space.n if isinstance(self.__eval_env.action_space, Discrete) else None
        self.skip_frames = False
        self.max_t_steps_per_episode = Config.MAX_EPISODAL_TIME_STEPS if use_max_episodal_t_steps else None
        self.device = torch.device(Config.DEVICE)

        # classes/methods
        self.network_fn = Config.CONV_NETWORK # Convolutional neural network for Q-learning
        self.replay_fn = Config.REPLAY_BUFFER # Replay function
        self.conv_optimizer_fn = Config.DEFAULT_OPTIMIZER_FN # Optimizer for the convolutional neural network 
        self.replay_memory_size = Config.REPLAY_SIZE # Replay function

        # normalizers for the input batch of tensors tensor of the convolutional neural network
        self.state_normalizer = AtariImageNormalizer(device=self.device, width=Config.STATE_WIDTH, height=Config.STATE_HEIGHT, frame_stack_size=Config.REPLAY_HISTORY_LENGTH) # normalizer for the state, i.e., the input tensors of Atari games

        # normalizers for the agent exploration
        self.reward_normalizer = SignNormalizer() # normalizer for the reward

        # params for Q-network update
        self.num_warmup_t_steps = Config.WARMUP_STEPS # Number of time steps without any posterior update or gradient descent 
        self.num_training_episodes = Config.MAX_TRAINING_EPISODE # the number of training episodes
        self.sampling_interval = Config.THOMPSON_SAMPLING_INTERVAL
        self.target_network_update_interval = Config.TARGET_NETWORK_UPDATE_INTERVAL
        self.target_weight_update_interval = Config.TARGET_WEIGHT_UPDATE_INTERVAL        
        self.gd_update_interval = Config.GD_UPDATE_INTERVAL # number of time steps between gradient descent updates
        self.prior_variance = Config.SIGMA_VARIANCE 
        self.noise_variance = Config.NOISE_VARIANCE
        self.gamma = Config.DISCOUNT
        self.max_posterior_update_batch_size = Config.MAX_BLR_BATCH_SIZE
        self.clip_rewards = Config.CLIP_REWARDS
        self.batch_size = Config.CONV_BATCH_SIZE # batch size for the DQN
        self.loss_function = torch.nn.MSELoss()

        """
        Other parameters
        """
        self.num_training_episodes_per_eval = 50
        self.num_eval_episodes = 5 # number of episodes to evaluate per $self.eval_interval time steps


    @property
    def eval_env(self) -> gym.Env:
        return self.__eval_env
    
        
    def add_argument(self, *args, **kwargs):
        self.arg_parser.add_argument(*args, **kwargs)


    def merge(self, config_dict=None):
        if config_dict is None:
            args = self.arg_parser.parse_args()
            config_dict = args.__dict__
        for key in config_dict.keys():
            setattr(self, key, config_dict[key])
