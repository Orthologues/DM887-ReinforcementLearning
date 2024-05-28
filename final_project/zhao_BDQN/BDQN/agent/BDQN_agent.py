from ..network import *
from ..utils import *
from torch import nn, optim
from typing import List
from pandas import DataFrame
from statistics import mean, stdev
import os


"""
Reference of the BDQN algorithm: 
https://github.com/kazizzad/BDQN-MxNet-Gluon/blob/master/BDQN.ipynb
The corresponding latex annotations are given at the comment for each line
"""


class BDQNAgent:

    def __init__(self, config: Config):
        # configuration of the agent
        self.config = config
        self.env = config.eval_env
        
        # attributes related to the states/actions
        self.num_actions = config.action_count
        self.phi_size = BdqnConvNet.FEATURE_DIM
        self.use_softmax_policy = Config.USE_SOFTMAX_POLICY

        # global attributes
        self.replay = config.replay_fn(self.config.replay_memory_size)
        self.policy_network: BdqnConvNet = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT)).to(self.config.device)
        self.target_network: BdqnConvNet = config.network_fn((Config.CONV_BATCH_SIZE, Config.REPLAY_HISTORY_LENGTH, config.STATE_WIDTH, Config.STATE_HEIGHT)).to(self.config.device)

        # optimizer for the network
        self.optimizer: optim.Optimizer = config.conv_optimizer_fn(self.policy_network.parameters())

        """
        configs used in posterior update
        """
        # constants
        self.posterior_update_batch_size = config.max_posterior_update_batch_size
        self.prior_variance = config.prior_variance # variance of the prior distribution, $\sigma^2$
        self.noise_variance = config.noise_variance # variance of the noise distribution, $\sigma_\epsilon^2$
        
        # initialization of the mean of the Thompson-sampled weights
        self.thompson_sampled_mean = torch.normal(0, 1e-2, generator=torch.manual_seed(42), size=(self.num_actions, self.phi_size)).to(self.config.device).detach()
        # initialization of the mean of the policy Q-mean matrix
        self.policy_mean = self.thompson_sampled_mean.clone().to(self.config.device).detach()
        # initialization of the mean of the target Q-mean matrix
        self.target_mean = self.policy_mean.clone().to(self.config.device).detach()
        
        # initialization of the mean of the policy Q-covariance matrix
        self.policy_cov = torch.normal(0, 1, generator=torch.manual_seed(42), size=(self.num_actions, self.phi_size, self.phi_size)).to(self.config.device).detach()
        # initialization of the variable for the mean of the Cholesky-decompoased policy Q-covariance matrix
        self.policy_cov_decom = self.policy_cov.clone().to(self.config.device).detach()
        # initialization of the target Q-covariance matrix
        self.target_cov = self.policy_cov.clone().to(self.config.device).detach()

        # initialization of actual values at self.policy_cov_decom using Cholesky decomposition
        for i in range(self.num_actions):
            self.policy_cov[i] = torch.eye(self.phi_size).to(self.config.device).detach()
            self.policy_cov_decom[i] = torch.linalg.cholesky( (self.policy_cov[i] + self.policy_cov[i].T)/2.0 ).to(self.config.device).detach()
        
        # initialization of the three-dimensional tensor of $\phi(x)\phi(x)^\top$
        self.phi_phi_t = torch.zeros(self.num_actions, self.phi_size, self.phi_size).to(self.config.device).detach()
        # initialization of the two-dimensional tensor of $\phi y$
        self.phi_Qtarget = torch.zeros(self.num_actions, self.phi_size).to(self.config.device).detach()

        # copy the state of policy network to initialize the target network
        self.target_network.load_state_dict(self.policy_network.state_dict())

        # values to track time steps and the number of episodes
        self.total_t_steps = 0
        self.total_gd_t_steps = 0 # total number of gradient descent time steps
        self.episodal_t_steps = 0
        self.target_network_update_count_per_target_weight_update = 0

        # episodal attributes during episodal interactions between the agent and the environment 
        self.episodal_clipped_reward = 0
        self.episodal_reward = 0
        self.num_skipped_frames = 4 if config.skip_frames else None

        # attributes related to the evaluation or recording
        self.training_pandas_df: DataFrame = DataFrame(columns=["training_episode_idx", "num_total_time_steps", "num_gd_time_steps", "episodal_reward", "episodal_t_steps"])
        self.eval_pandas_df: DataFrame = DataFrame(columns=["training_episode_idx", "num_gd_time_steps", "eval_reward_mean", "eval_reward_std", "num_eval_episodes"])


    def posterior_update(self) -> None:
        # reset self.phi_phi_t and self.phi_Qtarget to zero
        self.phi_phi_t *= 0
        self.phi_Qtarget *= 0
        # min: int(20000/32), max: int(200000/32)
        num_blr_repetitions = int( min(self.posterior_update_batch_size, self.total_t_steps) / self.config.batch_size)
        
        # repeat the posterior update exploration time steps
        for _ in range(num_blr_repetitions):
            batch_of_transitions: List[ReplayMemory.Transitions] = self.replay.sample(self.config.batch_size)
            """
            Transitions.states shall be of shape (4, 84, 84)
            """
            batch_of_states: torch.Tensor = torch.cat([el.states.unsqueeze(0) for el in batch_of_transitions], dim=0).to(self.config.device)
            batch_of_action: Tuple[int] = tuple([el.action for el in batch_of_transitions])
            batch_of_reward: Tuple[float] = tuple([el.reward for el in batch_of_transitions])
            batch_of_next_states: torch.Tensor = torch.cat([el.next_states.unsqueeze(0) for el in batch_of_transitions], dim=0).to(self.config.device)
            batch_of_done_flags: Tuple[bool] = tuple([el.done for el in batch_of_transitions])
            with torch.no_grad():
                """
                compute the Q-values of the next states
                shape of $batch_of_next_states_phi: (32, 512)
                shape of $expected_q_target: (32, )
                """
                policy_state_phi, expected_q_target = self.extract_state_phi(batch_of_states, batch_of_reward, batch_of_next_states, batch_of_done_flags)
            
                            # can improve by not using a loop
            for i in range(self.config.batch_size):
                action = batch_of_action[i]
                self.phi_phi_t[action] += torch.matmul(policy_state_phi[i].unsqueeze(0).T, policy_state_phi[i].unsqueeze(0)).to(self.config.device).detach()
                self.phi_Qtarget[action] += policy_state_phi[i] * expected_q_target[i].item()
            
            for i in range(self.num_actions):
                """
                size of $inv: (512, 512)
                """
                inv = torch.inverse( self.phi_phi_t[i]/self.noise_variance + 1/self.prior_variance * torch.eye(self.phi_size).to(self.config.device).detach() ).to(self.config.device).detach()
                self.policy_mean[i] = torch.matmul(inv, self.phi_Qtarget[i]).to(self.config.device).detach() / self.noise_variance
                self.policy_cov[i] = self.prior_variance * inv
                try:
                    self.policy_cov_decom[i] = torch.linalg.cholesky((self.policy_cov[i]+self.policy_cov[i].T)/2).to(self.config.device).detach()
                except RuntimeError:
                    pass


    def thompson_sample(self) -> None:
        for i in range(self.num_actions):
            # gene
            sample = torch.normal(0, 1, size=(self.phi_size, 1)).to(self.config.device).detach()
            """
            self.policy_mean[i].shape = (self.phi_size,)
            self.policy_cov_decom[i].shape = (self.phi_size, self.phi_size)
            """
            self.thompson_sampled_mean[i] = self.policy_mean[i] + torch.matmul(self.policy_cov_decom[i], sample).to(self.config.device).detach().squeeze(-1)
    

    """
    @param batch_of_states: torch.Tensor
        desired batch_of_states.shape: (32, 4, 84, 84)
    @param batch_of_rewards: Tuple[float]
        desired batch_of_rewards.__len__(): 32
    @param batch_of_next_states: torch.Tensor
        desired batch_of_next_states.shape: (32, 4, 84, 84)
    @param batch_of_done_flags: Tuple[bool]
        desired batch_of_done_flags.__len__(): 32
    """
    def extract_state_phi(self, batch_of_states: torch.Tensor, batch_of_reward: Tuple[float], batch_of_next_states: torch.Tensor, batch_of_done_flags: Tuple[bool]) -> Tuple[Tensor, Tensor]:
        
        batch_of_masks = tuple([1 if el == True else 0 for el in batch_of_done_flags])

        """
        shape of $batch_of_states_phi: (32, 512)
        shape of $batch_of_next_states_phi: (32, 512)
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
            shape of $batch_of_expected_q_target_next: (32,)
            """
            batch_of_q_next = torch.matmul(batch_of_next_states_phi_policy, self.thompson_sampled_mean.T)
            # use the softmax function to infer the probablity of each action to calculate the expected Q_next
            batch_of_prob_actions = nn.functional.softmax(batch_of_q_next, dim=1).to(self.config.device).detach()
            
            batch_of_target_q_next = torch.matmul(batch_of_next_states_phi_target, self.target_mean.T)
            batch_of_expected_q_target_next = \
                tensor(batch_of_reward).to(self.config.device).detach() + \
                self.config.gamma * (torch.ones((Config.CONV_BATCH_SIZE)).to(self.config.device).detach()) - \
                tensor(batch_of_masks).to(self.config.device).detach() * torch.sum(batch_of_prob_actions * batch_of_target_q_next, dim=1)
            
        return batch_of_states_phi, batch_of_expected_q_target_next


    """
    @param atari_state: the preprocessed current state of the environment (torch.Tensor)
    desired atari_state.shape: (4, 84, 84)
    """
    def select_action(self, atari_states: torch.Tensor) -> int:
        # reshape $atari_state to (1, 4, 84, 84)
        atari_states = atari_states.to(self.config.device).detach().unsqueeze(0)
        # output shape of $action_vector: (self.num_actions) 
        q_actions = torch.matmul(self.thompson_sampled_mean, self.policy_network(atari_states).T).to(self.config.device).detach().squeeze(-1)
        prob_actions = nn.functional.softmax(q_actions, dim=0).to(self.config.device).detach()

        action: int = torch.multinomial(prob_actions, 1)[0].item() if self.use_softmax_policy else torch.argmax(q_actions).item()
        return int(action)
    

    """
    @param action: int (range(0, self.num_actions)) 
    @param states: torch.Tensor
        desired states.shape: (4, 84, 84)
    """
    def act_without_frame_skipping(self, action: int, states: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
        
        next_frame, reward, terminated, truncated, _ = self.config.eval_env.step(action)
        done = terminated or truncated
        
        clipped_reward = self.config.reward_normalizer(reward)
        self.episodal_reward += reward
        self.episodal_clipped_reward += clipped_reward
        
        new_states = self.config.state_normalizer(states, next_frame)

        return new_states, reward, clipped_reward, done


    """
    @param action: int (range(0, self.num_actions)) 
    @param state: torch.Tensor
        desired states.shape: (4, 84, 84)
    """
    def act_with_frame_skipping(self, action: int, states: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
        
        rewards = 0
        clipped_rewards = 0
        
        for _ in range(self.num_skipped_frames - 1):
            _, reward, terminated, truncated, _ = self.config.eval_env.step(action)
            done = terminated or truncated
            rewards += reward    
            clipped_rewards += self.config.reward_normalizer(reward)
            if done:
                return states, rewards, clipped_rewards, done
        
        # add the non-skipped frame into $new_states
        next_frame, reward, terminated, truncated, _ = self.config.eval_env.step(action)
        done = terminated or truncated
        
        rewards += reward
        clipped_rewards += self.config.reward_normalizer(reward)

        self.episodal_reward += rewards 
        self.episodal_clipped_reward += clipped_rewards
        
        new_states = self.config.state_normalizer(states, next_frame)

        return new_states, rewards, clipped_rewards, done
    
    """
    @param action: int (range(0, self.num_actions)) 
    @param state: torch.Tensor
        desired states.shape: (4, 84, 84)
    """
    def act_in_env(self, action: int, state: torch.Tensor) -> Tuple[torch.Tensor, int, int, bool]:
        return self.act_with_frame_skipping(action, state) if self.config.skip_frames else self.act_without_frame_skipping(action, state)
    

    """
    Initialize the stacked frames, i.e., "states"
    @param first_frame: numpy.ndarray
        desired first_frame.shape = (210, 160, 3)
    @rval desired Tensor.shape = (4, 84, 84)
    """
    def init_episodal_states(self, x) -> torch.Tensor:
        x = tensor(x).to(device=self.config.device, dtype=torch.float32).detach().permute(2, 0, 1)
        x *= 1.0/255
        x = 0.2989 * x[0, :, :] + 0.5870 * x[1, :, :] + 0.1140 * x[2, :, :]
        # Add two dimensions to the start, the input frame has a shape at (1, 1, 210, 160) afterwards 
        x = x.unsqueeze(0)  
        x = x.unsqueeze(0)
        # input param "x" of nn.functional.interpolate must be of shape (batch_size, n_channels, width, height), it would have a shape at (1, 1, 84, 84) afterwards by default
        x: Tensor = nn.functional.interpolate(x, size=(self.config.STATE_WIDTH, self.config.STATE_HEIGHT), mode='bilinear', align_corners=False)
        # reshape x to (1, 84, 84)
        x = x.to(self.config.device).detach().squeeze(0)
        # output tensor will be of shape (batch_size, width, height)
        return concat_tensors([x.clone() for _ in range(self.config.REPLAY_HISTORY_LENGTH)], dim=0).to(self.config.device).detach()


    """
    The method to update the weights and biases of the policy DQN using the optimizer
    """
    def optimize_policy_network(self):
        batch_of_transitions: List[ReplayMemory.Transitions] = self.replay.sample(self.config.batch_size)
        """
        Transitions.states and Transitions.next_states shall be of shape (4, 84, 84)
        """
        batch_of_states: torch.Tensor = torch.cat([el.states.unsqueeze(0) for el in batch_of_transitions], dim=0).to(self.config.device)
        batch_of_next_states: torch.Tensor = torch.cat([el.next_states.unsqueeze(0) for el in batch_of_transitions], dim=0).to(self.config.device)
        batch_of_done_flags: Tuple[bool] = tuple([el.done for el in batch_of_transitions])
        batch_of_rewards: Tuple[int] = tuple([el.reward for el in batch_of_transitions])
        batch_of_actions: Tuple[int] = tuple([el.action for el in batch_of_transitions])
   
        # argmax_action_by_Q_policy.shape = (32, 1)
        argmax_action_by_Q_policy = torch.argmax(torch.matmul(self.policy_network(batch_of_next_states), self.thompson_sampled_mean.T), dim = 1).to(device=self.config.device, dtype=torch.int64).detach().unsqueeze(-1)
        # Q_target_next.shape = (32, self.num_actions)
        Q_target_next = torch.matmul(self.target_network(batch_of_next_states), self.target_mean.T).to(self.config.device)
        # Q_target_next_max.shape = (32, 1)
        # Q_target_next_expected.shape = (32, 1)
        Q_target_next_max = Q_target_next.gather(dim=1, index=argmax_action_by_Q_policy) * torch.tensor(tuple([1-flag for flag in batch_of_done_flags]), device=self.config.device).unsqueeze(-1)
        Q_target_next_expected = (torch.tensor(batch_of_rewards, device=self.config.device).unsqueeze(-1) + self.config.gamma * Q_target_next_max).to(dtype=torch.float64)
        
        # Q_policy_current.shape = (32, self.num_actions)
        Q_policy_current = torch.matmul(self.policy_network(batch_of_states), self.policy_mean.T).to(self.config.device)
        # Q_policy_observed_current.shape = (32, 1)
        Q_policy_observed_current = Q_policy_current.gather(dim=1, index=torch.tensor(batch_of_actions, device=self.config.device).unsqueeze(-1).to(dtype=torch.int64)).to(dtype=torch.float64)
        
        # $loss is a scalar tensor
        loss: torch.Tensor = self.config.loss_function(Q_policy_observed_current, Q_target_next_expected)
        loss = loss.to(device=self.config.device)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    """
    Method for training one episode
    """
    def train_one_episode(self) -> None:
        
        self.episodal_t_steps = 0
        self.episodal_reward = 0
        self.episodal_clipped_reward = 0
        first_frame, _ = self.config.eval_env.reset()
        next_states = self.init_episodal_states(first_frame)


        while True: 
            if self.config.max_t_steps_per_episode is not None:
                if self.episodal_t_steps >= self.config.max_t_steps_per_episode: 
                    print(f"max t steps per episode {self.config.max_t_steps_per_episode} reached, episodal reward: {self.episodal_reward}, clipped reward: {self.episodal_clipped_reward}, episodal time steps: {self.episodal_t_steps}, total time steps: {self.total_t_steps}, total gradient descent time steps: {self.total_gd_t_steps}")
                    break

            states = next_states
            action = self.select_action(states)

            next_states, reward, clipped_reward, done = self.act_in_env(action, states)
            self.replay.push(
                states=states,
                action=action,
                reward = clipped_reward if self.config.clip_rewards else reward,
                next_states=next_states,
                done=done
            )
            self.total_t_steps += 1
            self.episodal_t_steps += 1
            if done: 
                print(f"episodal reward: {self.episodal_reward}, clipped reward: {self.episodal_clipped_reward}, episodal time steps: {self.episodal_t_steps}, total time steps: {self.total_t_steps}, total gradient descent time steps: {self.total_gd_t_steps}")
                break

            # update the network
            if self.total_t_steps > self.config.num_warmup_t_steps:
                # perform Thompson sampling when needed
                if self.total_t_steps % self.config.sampling_interval == 0:
                    self.thompson_sample()
                # perform gradient descent to update
                if self.total_t_steps % self.config.gd_update_interval == 0:
                    self.optimize_policy_network()
                    self.total_gd_t_steps +=1
            
                # update the target network when necessary
                if self.total_t_steps % self.config.target_network_update_interval == 0:
                    # update the weights and biases of the target network
                    self.target_network.load_state_dict(self.policy_network.state_dict())
                    self.target_network_update_count_per_target_weight_update += 1
                    # update the weight and covariance matrices of the target Q values
                    if self.target_network_update_count_per_target_weight_update == self.config.target_weight_update_interval:
                         self.posterior_update()
                         self.target_mean = self.policy_mean
                         self.target_cov = self.policy_cov
                         self.target_network_update_count_per_target_weight_update = 0

    """
    run the evaluation mode for $config.num_eval_episodes episodes
    """     
    def run_eval_mode(self, training_episode_idx: int, csv_prefix: str):
        
        episodal_rewards = []

        for _ in range(self.config.num_eval_episodes):
            self.episodal_t_steps = 0
            self.episodal_reward = 0
            self.episodal_clipped_reward = 0
            first_frame, _ = self.config.eval_env.reset()
            next_states = self.init_episodal_states(first_frame)
            while True: 
                if self.config.max_t_steps_per_episode is not None:
                    if self.episodal_t_steps >= self.config.max_t_steps_per_episode: 
                        episodal_rewards.append(self.episodal_reward)
                        break
                
                states = next_states
                action = self.select_action(states)

                next_states, reward, clipped_reward, done = self.act_in_env(action, states)
                self.replay.push(
                    states=states,
                    action=action,
                    reward = clipped_reward if self.config.clip_rewards else reward,
                    next_states=next_states,
                    done=done
                )
                self.episodal_t_steps += 1
                if done: 
                    episodal_rewards.append(self.episodal_reward)
                    break
        
        episodal_data_row = {
            "training_episode_idx": training_episode_idx, 
            "num_gd_time_steps": self.total_gd_t_steps,
            "eval_reward_mean": mean(episodal_rewards), 
            "eval_reward_std": stdev(episodal_rewards),
            "num_eval_episodes": self.config.num_eval_episodes
        }

        # Append the new row to the DataFrame
        self.eval_pandas_df.loc[len(self.eval_pandas_df)] = episodal_data_row
        with open(f"{csv_prefix}.csv", 'w') as f:
            self.eval_pandas_df.to_csv(f, mode='a', header=f.tell()==0, index=False)

        
    def save_model(self, training_episode_idx: int, prefix: str):

        directory = f"{prefix}/model"
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        episodal_data_row = {
            "training_episode_idx": training_episode_idx, 
            "num_total_time_steps": self.total_t_steps, 
            "num_gd_time_steps": self.total_gd_t_steps,
            "episodal_reward": self.episodal_reward, 
            "episodal_t_steps": self.episodal_t_steps,
        }

        # Append the new row to the DataFrame
        self.training_pandas_df.loc[len(self.training_pandas_df)] = episodal_data_row
        with open(f"{directory}/training_record.csv", 'w') as f:
            self.training_pandas_df.to_csv(f, mode='a', header=f.tell()==0, index=False)
        
        torch.save(self.policy_network.state_dict(), f"{directory}/policy_dqn_.pth")
        torch.save(self.target_network.state_dict(), f"{directory}/target_dqn_.pth")
        torch.save(self.policy_mean, f"{directory}/policy_E_W_mean.pth")
        torch.save(self.policy_cov, f"{directory}/policy_E_W_cov.pth")
        torch.save(self.policy_cov_decom, f"{directory}/policy_E_W_std.pth")
        torch.save(self.thompson_sampled_mean, f"{directory}/policy_E_W_sampled.pth")
        torch.save(self.target_mean, f"{directory}/target_E_W_mean.pth")
        torch.save(self.target_cov, f"{directory}/target_E_W_cov.pth")

        