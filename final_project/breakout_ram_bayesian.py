import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm

import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

def rgb_to_gray(image):
    # Convert RGB image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return gray_image

#replay buffer class
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

#'''
class DQN(nn.Module):

    def __init__(self, in_length, out_length, hidden_dim = 1024):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(in_length, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_mean = nn.Linear(hidden_dim, out_length)
        self.fc3_stddev = nn.Linear(hidden_dim, out_length)


    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mean = self.fc3_mean(x)
        stddev = torch.exp(self.fc3_stddev(x))
        return mean, stddev
#'''
class learner():

    def __init__(self, environment, device = 'cuda', BATCH_SIZE = 16, ALPHA = 0.99, LR = 0.001, num_episodes = 100, REPETITIONS = 5, eval_freq = 10,
                 plot_version = 0, memory_size = 10000, memory_min_size = 1000, UPDATE_TARGET_AFTER  = 10, TAU = 0.05, adv = False, atari = False,
                 IMG_SIZE = 50, epsilon_min = 0.01, epsilon_start = 1, epsilon_decay = 0.995, softmax = True, large = False, sample_size = 32):

        self.env = gym.make(environment)
        self.device = device
        if self.device == 'cuda':
            device = torch.device("cuda:0")
            print("Running on the GPU")
        else:
            device = torch.device("cpu")
            print("Running on the CPU")

        self.device = device

        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.epsilon = self.epsilon_start
        self.softmax = softmax
        self.BATCH_SIZE = BATCH_SIZE
        self.ALPHA = ALPHA
        self.LR = LR
        self.num_episodes = num_episodes
        self.REPETITIONS = REPETITIONS
        self.eval_freq = eval_freq
        self.plot_version = plot_version
        self.UPDATE_TARGET_AFTER = UPDATE_TARGET_AFTER
        self.TAU = TAU
        self.IMG_SIZE = IMG_SIZE

        self.sample_size = sample_size
        self.episode_durations = []
        self.losses = []
        self.GRADIENT_STEPS = 0
        self.REWARD_MEANS = []
        self.REWARDS = []
        self.REWARD_DEVIATIONS = []
        self.GRADIENT_STEPS_NUMBERS = []
        self.steps_done = 0
        self.target_update_after_counter = 0
        self.adv = adv
        self.atari = atari
        state, _ = self.env.reset()
        self.state = state
        self.n_actions = self.env.action_space.n
        self.action_types = [i for i in range(self.n_actions)]
        self.large = large
        self.update = 0

        self.n_observations = len(self.state.reshape(-1))
        self.train_Q = DQN(self.n_observations, self.n_actions, hidden_dim = 1024).to(self.device)
        self.target_Q = DQN(self.n_observations, self.n_actions, hidden_dim = 1024).to(self.device)
        self.target_Q.load_state_dict(self.train_Q.state_dict())

        #self.softmax = torch.nn.Softmax(dim = -1)
        #if self.adv:
           # self.loss_function = nn.SmoothL1Loss()
           # self.optimizer = optim.AdamW(self.train_Q.parameters(), lr=self.LR, amsgrad=True, weight_decay=0.00001)
        self.loss_function = torch.nn.MSELoss()
        #self.loss_function = torch.nn.SmoothL1Loss()
        #self.optimizer = optim.AdamW(self.train_Q.parameters(), lr=self.LR, amsgrad=True)
        self.optimizer = optim.AdamW(self.train_Q.parameters(), lr=self.LR)

        self.memory_size = memory_size
        self.memory = ReplayMemory(self.memory_size)
        self.memory_min_size = memory_min_size

    #simplePlotting
    def simple_plotting(self):
        plt.plot(np.array(self.GRADIENT_STEPS_NUMBERS), np.array(self.REWARDS), color='r')
        #plt.fill_between(np.array(self.GRADIENT_STEPS_NUMBERS),
        #                 np.array(self.REWARD_MEANS) - np.array(self.REWARD_DEVIATIONS),
        #                 np.array(self.REWARD_DEVIATIONS) + np.array(self.REWARD_MEANS), alpha=0.2, color='k')
        #plt.ylabel("Mean of the Total Episode Rewards Over Five Repetitions")
        plt.ylabel("Episode Rewards")
        plt.xlabel("Gradient-Descent Steps Taken")
        text = (f'{environment}: Bayesian')
        plt.title(text)
        plt.savefig("learning-curve_new.png")
        plt.ioff()
        plt.show()

    def preprocess_observation(self, obs):
        # return obs.reshape(-1)
        if self.atari:
            # x = obs / 255.0
            x = rgb_to_gray(obs) / 255.0
            # x = obs.permute(1, 2, 0).shape
            x = cv2.resize(x, (self.IMG_SIZE, self.IMG_SIZE))
            x = np.expand_dims(x, axis=0)
            # x = np.einsum('lij->jli', obs)
            # x[0] = x[0]/255.0
            return x
        else:
            return obs / 255.0

    def select_action(self, state):
        with torch.no_grad():
            #state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            state_tensor = state
            mean, stddev = self.train_Q(state_tensor)
            action_probs = torch.normal(mean, stddev)
            action = torch.argmax(action_probs)
            return torch.tensor([[action]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.memory_min_size:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        M, S = self.train_Q(state_batch)
        Means_state = M.gather(1, action_batch)
        Sddevs_state = S.gather(1, action_batch)

        with torch.no_grad():
            next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
            #next_state_means = torch.zeros(self.BATCH_SIZE, device=self.device)
            #next_state_sddevs = torch.zeros(self.BATCH_SIZE, device=self.device)
            for i in range(self.sample_size):
                next_state_means, next_state_sddevs = self.target_Q(non_final_next_states)
                next_state_means = next_state_means.gather(1, action_batch[non_final_mask])
                next_state_sddevs = next_state_sddevs.gather(1, action_batch[non_final_mask])
                next_state_values[non_final_mask] += torch.normal(next_state_means,next_state_sddevs).squeeze()
            next_state_values = next_state_values/self.sample_size

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.ALPHA) + reward_batch

        # Optimize the model
        loss = self.loss_function(Means_state, expected_state_action_values.unsqueeze(1))
        self.GRADIENT_STEPS += 1
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        #if self.adv:
        torch.nn.utils.clip_grad_value_(self.train_Q.parameters(), 100)
        self.optimizer.step()
        #self.update += 1

        #if self.update >= self.UPDATE_TARGET_AFTER:
        #    self.target_Q.load_state_dict(self.train_Q.state_dict())
        #    self.update = 0

    def train_one_episode_and_eval(self):
        done = False
        state, _ = self.env.reset()
        state = self.preprocess_observation(state)
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        #print(state.shape)
        total_reward = 0
        while not done:
            action = self.select_action(state)
            observation, reward, terminated, truncated, _ = self.env.step(action.item())
            total_reward = total_reward + reward
            reward = torch.tensor([reward], device=self.device)
            done = terminated or truncated

            observation = self.preprocess_observation(observation)
            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

            # Store the transition in memory
            self.memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            self.target_update_after_counter += 1
            if self.target_update_after_counter >= self.UPDATE_TARGET_AFTER:
                self.optimize_model()

            target_net_state_dict = self.target_Q.state_dict()
            policy_net_state_dict = self.train_Q.state_dict()
            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (
                            1 - self.TAU)
            self.target_Q.load_state_dict(target_net_state_dict)

            if done:
                self.REWARDS.append(total_reward)
                self.GRADIENT_STEPS_NUMBERS.append(self.GRADIENT_STEPS)
                break
    def training_run_simple(self):
        for episode in tqdm(range(self.num_episodes)):
            self.train_one_episode_and_eval()
        self.simple_plotting()
        print('Complete')

environment = "ALE/Breakout-ram-v5" # replace with "MountainCar-v0" or "Acrobot-v1" if we want the other environment ALE/Pong-v5 "CartPole-v1" "ALE/Breakout-v5" "ALE/Breakout-ram-v5" "Breakout-ram-v0"

Learner = learner(environment, device = 'cuda', BATCH_SIZE = 16, ALPHA = 0.99, LR = 0.0001, num_episodes = 200, memory_size = 10000, memory_min_size = 1000, TAU = 0.5, atari = False, sample_size = 64, UPDATE_TARGET_AFTER = 100)

Learner.training_run_simple()