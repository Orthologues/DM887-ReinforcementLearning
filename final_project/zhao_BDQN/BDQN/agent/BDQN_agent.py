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
        self.state_processor = config.state_normalizer
        self.use_softmax_policy = Config.USE_SOFTMAX_POLICY

        # global attributes
        self.replay = config.replay_fn(BDQNAgent.REPLAY_SIZE)

        self.policy_network: BdqnConvNet = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT))
        self.target_network: BdqnConvNet = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT))

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
        self.thompson_sampled_mean = torch.normal(0, 1e-2, generator=torch.manual_seed(42), size=(self.num_actions, self.phi_size)).detach()
        # initialization of the mean of the policy Q-mean matrix
        self.policy_mean = self.thompson_sampled_mean.clone().detach()
        # initialization of the mean of the target Q-mean matrix
        self.target_mean = self.policy_mean.clone().detach()
        
        # initialization of the mean of the policy Q-covariance matrix
        self.policy_cov = torch.normal(0, 1, generator=torch.manual_seed(42), size=(self.num_actions, self.phi_size, self.phi_size)).detach()
        # initialization of the variable for the mean of the Cholesky-decompoased policy Q-covariance matrix
        self.policy_cov_decom = self.policy_cov.clone().detach()
        # initialization of the target Q-covariance matrix
        self.target_cov = self.policy_cov.clone().detach()

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
                batch_of_transitions: List[ReplayMemory.Transitions] = self.replay.sample(batch_size)
                """
                Transitions.states shall be of shape (4, 84, 84)
                """
                batch_of_states: Tensor = torch.cat([el.states for el in batch_of_transitions], dim=0)
                batch_of_actions: Tuple[int] = (el.actions for el in batch_of_transitions)
                batch_of_rewards: Tuple[float] = (el.rewards for el in batch_of_transitions)
                batch_of_next_states: Tensor = torch.cat([el.next_states for el in batch_of_transitions], dim=0)
                batch_of_done_flags: Tuple[bool] = (el.done for el in batch_of_transitions)

                with torch.no_grad():
                    # compute the Q-values of the next states
                    policy_state_phi, expected_q_target = self.extract_state_phi(batch_of_states, batch_of_actions, batch_of_rewards, batch_of_next_states, batch_of_done_flags)


    def thompson_sample(self):
        for i in range(self.num_actions):
            # gene
            sample = tensor(torch.normal(0, 1, size=(self.phi_size, 1)))
            """
            self.policy_mean[i].shape = (self.phi_size,)
            self.policy_cov_decom[i].shape = (self.phi_size, self.phi_size)
            """
            self.thompson_sampled_mean[i] = self.policy_mean[i] + torch.matmul(self.policy_cov_decom[i], sample).squeeze(-1)

        self.policy_mean = self.thompson_sampled_mean.clone().detach()
    

    """
    
    """
    def extract_state_phi(self, batch_of_states: Tensor, batch_of_actions: Tuple[int], batch_of_rewards: Tuple[float], batch_of_next_states: Tensor, batch_of_done_flags: Tuple[bool]):
        
        with torch.no_grad():
            """
            shape of batch_of_states_phi: (32, 512)
            shape of batch_of_next_states_phi: (32, 512)
            """
            with torch.no_grad():
                batch_of_states_phi = self.policy_network(batch_of_states)
                batch_of_next_states_phi_policy = self.policy_network(batch_of_next_states)
                batch_of_next_states_phi_target = self.target_network(batch_of_next_states)
                # calculate the expected Q-value of the next states using the target Q mean values
                """
                shape of $batch_of_q_next: (32, self.num_actions)
                shape of $batch_of_prob_actions: (32, self.num_actions)
                shape of $batch_of_target_q_next: (32, self.num_actions)
                shape of $expected_batch_q_next: (32,)
                """
                batch_of_q_next = torch.matmul(batch_of_next_states_phi_policy, self.policy_mean.T).detach()
                # use the softmax function to infer the probablity of each action to calculate the expected Q_next
                batch_of_prob_actions = nn.functional.softmax(batch_of_q_next, dim=1)
                
                batch_of_target_q_next = torch.matmul(batch_of_next_states_phi_target, self.target_mean.T).detach()
                expected_batch_q_next = torch.sum(batch_of_prob_actions * batch_of_target_q_next, dim=1)
            
            return batch_of_states_phi, expected_batch_q_next




    """
    @param atari_state: the preprocessed current state of the environment (torch.Tensor)
    desired atari_state.shape: (4, 84, 84)
    """
    def select_action(self, atari_states: Tensor) -> int:
        # reshape $atari_state to (1, 4, 84, 84)
        atari_states = atari_states.unsqueeze(0)
        # output shape of $action_vector: (self.num_actions) 
        q_actions = torch.matmul(self.thompson_sampled_mean, self.policy_network(atari_states).T).squeeze(-1)
        prob_actions = nn.functional.softmax(q_actions, dim=0)

        action: int = torch.multinomial(prob_actions, 1)[0].item() if self.use_softmax_policy else torch.argmax(q_actions).item()
        return action