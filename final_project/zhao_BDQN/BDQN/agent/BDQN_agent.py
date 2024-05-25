from ..network import *
from ..utils import *
from torch import nn, optim
from typing import List

"""
Reference of the BDQN algorithm: 
https://github.com/kazizzad/BDQN-MxNet-Gluon/blob/master/BDQN.ipynb
The corresponding latex annotations are given at the comment for each line
"""


class BDQNAgent:

    MAX_BLR_BATCH_SIZE = 2 * 10**5
    REPLAY_SIZE = 2 * 10**6

    def __init__(self, config: Config):
        # configuration of the agent
        self.config = config

        # attributes related to the states/actions
        self.num_actions = config.action_count
        self.phi_size = BdqnConvNet.FEATURE_DIM
        
        # global attributes
        self.replay = config.replay_fn(BDQNAgent.REPLAY_SIZE)
        self.policy_network: nn.Module = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT))
        self.target_network: nn.Module = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT))

        # optimizer for the network
        self.optimizer: optim.Optimizer = config.conv_optimizer_fn(self.policy_network.parameters())

        """
        configs used in posterior update
        """
        # constants
        self.blr_batch_size = BDQNAgent.MAX_BLR_BATCH_SIZE
        self.prior_variance = config.prior_variance # variance of the prior distribution, $\sigma^2$
        self.noise_variance = config.noise_variance # variance of the noise distribution, $\sigma_\epsilon^2$
        # initialization of the mean of the Thompson-sampled weights
        self.thompson_sampled_mean = torch.normal(0, 1e-2, size=(self.num_actions, self.phi_size))
        # initialization of the mean of the policy Q-mean matrix
        self.policy_mean = torch.normal(0, 1e-2, size=(self.num_actions, self.phi_size))
        # initialization of the mean of the policy Q-covariance matrix
        self.policy_cov = torch.normal(0, 1, generator=torch.manual_seed(42), size=(self.num_actions, self.phi_size, self.phi_size))
        # initialization of the variable for the mean of the Cholesky-decompoased policy Q-covariance matrix
        self.policy_cov_decom = self.policy_cov
        # initialization of actual values at self.policy_cov_decom using Cholesky decomposition
        for i in range(self.num_actions):
            self.policy_cov[i] = torch.eye(self.phi_size)
            self.policy_cov_decom[i] = torch.cholesky( (self.policy_cov[i] + self.policy_cov[i].T)/2.0 )
        
        # initialization of the three-dimensional tensor of $\phi(x)\phi(x)^\top$
        self.phi_phi_t = torch.zeros(self.num_actions, self.phi_size, self.phi_size)
        # initialization of the two-dimensional tensor of $\phi y$
        self.phi_y = torch.zeros(self.num_actions, self.phi_size)

        # the flag to indicate whether to perform posterior update 
        self.if_posterior_update = False

        # copy the state of policy network to initialize the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # values to track time steps and the number of episodes
        self.total_t_steps = 0
        self.episodal_t_steps = 0
        self.num_episode = 0


    def preprocess_sampled_replay_batch(batch: List[ReplayMemory.Transitions]):
        pass


    def posterior_update(self):
        # reset self.phi_phi_t and self.phi_y to zero
        self.phi_phi_t *= 0
        self.phi_y *= 0

        if self.total_t_steps > self.config.pre_blr_update_t_steps:
            batch_size = Config.CONV_BATCH_SIZE
            # min: int(20000/32), max: int(200000/32)
            num_blr_repetitions = int( min(self.blr_batch_size, self.total_t_steps) / batch_size)
            
            # repeat the posterior update exploration time steps
            for _ in range(num_blr_repetitions):
                batch_of_transitions = self.replay.sample(batch_size)
                for transitions in batch_of_transitions:
                    pass


    
    

