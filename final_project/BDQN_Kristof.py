#import math
#from collections import namedtuple, deque
#import time
#import pickle
#import mxnet as mx
#from mxnet import nd, autograd
#from mxnet import gluon
#import torch.selfim as selfim

#from IPython import display
import matplotlib.pyplot as plt
#from __future__ import print_function
#%matplotlib inline
#import matplotlib.ticker as mtick

import os
import gymnasium as gym
import math
import random
import numpy as np
import matplotlib
from collections import namedtuple, deque
from itertools import count
from tqdm import tqdm
import logging, logging.handlers

from torch import nn
from torch import nn, Tensor, relu
import numpy as np
from torch import Tensor, nn
from typing import Tuple, List, Union, Any, Dict
import cv2
import torch
import torch.nn as nn
import torchvision as tv
import torch.nn.functional as F
import torch.optim as optim

command = 'mkdir data' # Creat a direcotry to store models and scores.
os.system(command)

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

Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'done'))



"""
The sign function returns -1 if x < 0, 0 if x==0, 1 if x > 0. nan is returned for nan inputs.
"""

class BdqnConvNet(nn.Module):
    def __init__(self, input_dim, feature_dim=512, num_filters=32):
        super().__init__()
        self.init_body(input_dim, feature_dim, num_filters)

    def init_body(self, input_dim, output_dim, num_filters) -> None:
        """
        Desired input shape: (32, 4, 84, 84)
        Output Shape 1: (32,32,20,20)
        Output Shape 2: (32,64,9,9)
        Output Shape 3: (32,64,7,7)
        Input Shape 4 (after flattening): (32,3136) # 3136=7*7*64
        Output Shape 4: (32, 512)
        """
        # the 1th layer of the feature extractor (convolutional)
        self.conv1 = self.init_layer(nn.Conv2d(in_channels=input_dim[-1], out_channels=num_filters, kernel_size=8, stride=4))
        self.bn1 = nn.BatchNorm2d(num_filters)  # batch normalization for the output of conv layer 1
        # the 2th layer of the feature extractor (convolutional)
        self.conv2 = self.init_layer(nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2))
        self.bn2 = nn.BatchNorm2d(num_filters * 2)  # batch normalization for the output of conv layer 2
        # the 3th layer of the feature extractor (convolutional)
        self.conv3 = self.init_layer((nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=3, stride=1)))
        self.bn3 = nn.BatchNorm2d(num_filters * 2)  # batch normalization for the output of conv layer 3

        # the 4th layer of the feature extractor (linear)
        self.fc4 = self.init_layer(nn.Linear(7 * 7 * 64, output_dim))

    def init_layer(self, layer: nn.Module) -> nn.Module:
        nn.init.orthogonal_(layer.weight.data)
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def forward(self, x: Tensor):
        x = relu(self.bn1(self.conv1(x)))
        x = relu(self.bn2(self.conv2(x)))
        x = relu(self.bn3(self.conv3(x)))
        x = x.flatten(start_dim=1)
        x = relu(self.fc4(x))
        return x

class BDQN_Learner():

    def __init__(self, env, render_image = False, device = "cuda:0", skip = False):

        self.skip = skip
        self.env_name = env
        self.env = gym.make(self.env_name)
        self.loss_function = torch.nn.MSELoss()
        self.num_action = self.env.action_space.n
        self.batch_size = 32  # The size of the batch to learn the Q-function
        self.image_size = 84  # Resize the raw input frame to square frame of size 80 by 80
        # Trickes
        self.replay_buffer_size = 100000  # The size of replay buffer; set it to size of your memory (.5M for 50G available memory)
        self.learning_frequency = 4  # With Freq of 1/4 step update the Q-network
        self.skip_frame = 4  # Skip 4-1 raw frames between steps
        self.internal_skip_frame = 4  # Skip 4-1 raw frames between skipped frames
        self.frame_len = 4  # Each state is formed as a concatination 4 step frames [f(t-12),f(t-8),f(t-4),f(t)]
        self.Target_update = 2000  # Update the target network each 10000 steps
        self.epsilon_min = 0.1  # Minimum level of stochasticity of policy (epsilon)-greedy
        self.annealing_end = 1000000.  # The number of step it take to linearly anneal the epsilon to it min value
        self.gamma = 0.99  # The discount factor
        self.replay_start_size = 1000  # Start to backpropagated through the network, learning starts

        # otimization
        self.max_episode = 10000  # max number of episodes#
        self.lr = 0.0025  # RMSprop learning rate
        self.gamma1 = 0.95  # RMSprop gamma1
        self.gamma2 = 0.95  # RMSprop gamma2
        self.rms_eps = 0.01  # RMSprop epsilon bias
        self.ctx = torch.device(device)  # Enables gpu if available, if not, set it to mx.cpu()
        self.lastlayer = 512  # Dimensionality of feature space
        self.f_sampling = 100  # frequency sampling E_W_ (Thompson Sampling)
        self.alpha = .01  # forgetting factor 1->forget
        self.alpha_target = 1  # forgetting factor 1->forget
        self.f_bayes_update = 1000  # frequency update E_W and Cov
        self.target_batch_size = 5000  # target update sample batch
        self.BayesBatch = 10000  # size of batch for udpating E_W and Cov
        self.target_W_update = 3
        self.lambda_W = 0.1  # update on W = lambda W_new + (1-lambda) W
        self.sigma = 0.001  # W prior variance
        self.sigma_n = 1  # noise variacne
        self.frame_counter = 0

        self.cum_clipped_reward = 0
        self.cum_reward = 0
        self.fratot_clipped_rewardme_counter = 0.  # Counts the number of steps so far
        self.annealing_count = 0.  # Counts the number of annealing steps
        self.epis_count = 1.  # Counts the number episodes so far
        self.tot_clipped_reward = []
        self.tot_reward = []
        self.frame_count_record = []
        self.moving_average_clipped = 0.
        self.moving_average = 0.
        self.flag = 0
        self.c_t = 0

        self.bat_state = torch.empty((1, self.frame_len, self.image_size, self.image_size), device=self.ctx)
        self.bat_state_next = torch.empty((1, self.frame_len, self.image_size, self.image_size), device=self.ctx)
        self.bat_reward = torch.empty((1), device=self.ctx)
        self.bat_action = torch.empty((1), device=self.ctx)
        self.bat_done = torch.empty((1), device=self.ctx)

        self.eye = torch.zeros((self.lastlayer, self.lastlayer), device=self.ctx)
        for i in range(self.lastlayer):
            self.eye[i, i] = 1

        self.E_W = torch.normal(mean=0, std=.01, size=(self.num_action, self.lastlayer)).to(self.ctx)
        self.E_W_target = torch.normal(mean=0, std=.01, size=(self.num_action, self.lastlayer)).to(self.ctx)
        self.E_W_ = torch.normal(mean=0, std=.01, size=(self.num_action, self.lastlayer)).to(self.ctx)
        self.Cov_W = torch.normal(mean=0, std=1, size=(self.num_action, self.lastlayer, self.lastlayer)).to(self.ctx) + self.eye
        self.Cov_W_decom = self.Cov_W
        for i in range(self.num_action):
            self.Cov_W[i] = self.eye
            self.Cov_W_decom[i] = torch.linalg.cholesky(((self.Cov_W[i] + self.Cov_W[i].T) / 2.)).to(self.ctx)
        self.Cov_W_target = self.Cov_W
        self.phiphiT = torch.zeros((self.num_action, self.lastlayer, self.lastlayer), device = self.ctx)
        self.phiY = torch.zeros((self.num_action, self.lastlayer), device = self.ctx)


        self.render_image = render_image  # Whether to render Frames and show the game
        self.batch_state = torch.empty((self.batch_size, self.frame_len, self.image_size, self.image_size), device=self.ctx)
        self.batch_state_next = torch.empty((self.batch_size, self.frame_len, self.image_size, self.image_size), device=self.ctx)
        self.batch_reward = torch.empty((self.batch_size), device=self.ctx)
        self.batch_action = torch.empty((self.batch_size), device=self.ctx)
        self.batch_done = torch.empty((self.batch_size), device=self.ctx)

        self.replay_memory = ReplayMemory(self.replay_buffer_size) # Initialize the replay buffer

        self.dqn_ = BdqnConvNet(input_dim = [4]).to(self.ctx)
        self.target_dqn_ = BdqnConvNet(input_dim = [4]).to(self.ctx)
        self.target_dqn_.load_state_dict(self.dqn_.state_dict())

        self.optimizer = optim.AdamW(self.dqn_.parameters(), lr=self.lr)
        self.resizer = tv.transforms.Resize((84,84))

    def rew_clipper(self, rew):
        if rew > 0.:
            return 1.
        elif rew < 0.:
            return -1.
        else:
            return 0


    def preprocess(self, raw_frame, currentState=None, initial_state=False):
        raw_frame = torch.tensor(raw_frame, dtype=torch.float32)
        raw_frame = torch.reshape(torch.mean(raw_frame, dim=2, dtype=torch.float32),
                                  shape=(1, raw_frame.shape[0], raw_frame.shape[1]))
        raw_frame = self.resizer(raw_frame)

        raw_frame = raw_frame / 255.
        if initial_state == True:
            state = raw_frame
            for _ in range(4 - 1):
                state = torch.cat((state, raw_frame), dim=0)
        else:
            state = torch.cat((currentState[1:, :, :], raw_frame), dim=0)
        return state

    def select_action(self, state):
        data = state.reshape([1, self.frame_len, self.image_size, self.image_size]).to(self.ctx)
        a = torch.matmul(self.E_W_, self.dqn_(data).T) #****************
        action = torch.argmax(a)
        return action

    def env_act_No_Skip(self, action, state):
        next_frame, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        state = self.preprocess(next_frame, state)
        self.cum_reward += reward
        return state, reward, done

    def env_act_Skip_frame(self, action, state):
        rew = 0
        for skip in range(self.skip_frame - 1):
            next_frame, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            self.cum_clipped_reward += self.rew_clipper(reward)
            rew += reward
            for internal_skip in range(self.internal_skip_frame - 1):
                _, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                self.cum_clipped_reward += self.rew_clipper(reward)
                rew += reward
        next_frame_new, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        self.cum_clipped_reward += self.rew_clipper(reward)
        rew += reward

        self.cum_reward += rew
        # Reward clipping
        reward = self.rew_clipper(rew)
        next_frame = np.maximum(next_frame_new, next_frame)
        state = self.preprocess(next_frame, state)

        return state, reward, done

    def env_act(self, action, state):
        if self.skip:
            return self.env_act_Skip_frame(action, state)
        else:
            return self.env_act_No_Skip(action, state)

    def BayesReg(self, phiphiT, phiY, alpha, batch_size):
        E_W = self.E_W
        Cov_W = self.Cov_W
        phiphiT *= (1 - alpha)  # Forgetting parameter alpha suggest how much of the moment from the past can be used, we set alpha to 1 which means do not use the past moment
        phiY *= (1 - alpha)
        for j in range(batch_size):
            transitions = self.replay_memory.sample(1)  # sample a minibatch of size one from replay buffer
            bat_state = torch.tensor(transitions[0].state, dtype='float32', device=self.ctx) / 255.
            bat_state_next = torch.tensor(transitions[0].next_state, dtype='float32', device=self.ctx) / 255.
            bat_reward = transitions[0].reward
            bat_action = transitions[0].action
            bat_done = transitions[0].done
            phiphiT[int(bat_action)] += torch.matmul(self.dqn_(bat_state).T, self.dqn_(bat_state))
            phiY[int(bat_action)] += (self.dqn_(bat_state)[0].T * (bat_reward + (1. - bat_done) * self.gamma * torch.max(torch.matmul(self.E_W_target, self.target_dqn_(bat_state_next)[0].T))))
        for i in range(self.num_action):
            inv = torch.inverse((phiphiT[i] / self.sigma_n + 1 / self.sigma * self.eye).asnumpy())
            E_W[i] = torch.tensor(np.dot(inv, phiY[i].asnumpy()) / self.sigma_n, device=self.ctx)
            Cov_W[i] = self.sigma * inv
        return phiphiT, phiY, E_W, Cov_W

        # Thompson sampling, sample model W form the posterior.

    def sample_W(self, E_W, U):
        E_W_ = self.E_W_
        for i in range(self.num_action):
            sam = torch.normal(0, 1, size=(512, 1)).to(self.ctx)
            E_W_[i] = E_W[i] + torch.matmul(U[i], sam)[:, 0]
        return E_W_

    '''
    def renderimage(self, next_frame):
        if self.render_image:
            plt.imshow(next_frame);
            plt.show()
            display.clear_output(wait=True)
            time.sleep(.1)
    '''

    def train_one_episode(self):
        self.cum_clipped_reward = 0
        self.cum_reward = 0
        next_frame, _ = self.env.reset()
        state = self.preprocess(next_frame, initial_state=True)
        t = 0.
        done = False
 
        while not done:
            #mx.nd.waitall()
            previous_state = state
            # show the frame
            if self.frame_counter > self.replay_start_size:
                self.annealing_count += 1

            action = self.select_action(state)

            state, reward, done = self.env_act(action.item(), state)

            self.replay_memory.push(previous_state * 255., action, state * 255., reward, done)
            # Thompson Sampling
            if self.frame_counter % self.f_sampling:
                self.E_W_ = self.sample_W(self.E_W, self.Cov_W_decom)

            # Train
            if self.frame_counter > self.replay_start_size:
                if self.frame_counter % self.learning_frequency == 0:
                    batch = self.replay_memory.sample(self.batch_size)
                    # update network
                    for j in range(self.batch_size):
                        self.batch_state[j] = batch[j].state / 255.
                        self.batch_state_next[j] = batch[j].next_state / 255.
                        self.batch_reward[j] = batch[j].reward
                        self.batch_action[j] = batch[j].action
                        self.batch_done[j] = batch[j].done
                    #with autograd.record():
                    argmax_Q = torch.argmax(torch.matmul(self.dqn_(self.batch_state_next), self.E_W_.T), dim = 1).to(torch.int32)
                    Q_sp_ = torch.matmul(self.target_dqn_(self.batch_state_next), self.E_W_target.T)
                    Q_sp = torch.index_select(Q_sp_,1, argmax_Q) * (1 - self.batch_done)
                    Q_s_array = torch.matmul(self.dqn_(self.batch_state), self.E_W.T)
                    #if (Q_s_array[0, 0] != Q_s_array[0, 0]): #.asscalar()
                    #    self.flag = 1
                    #    print('break')
                    #    break
                    Q_s = torch.index_select(Q_s_array, 1, self.batch_action.to(torch.int32))
                    loss = torch.mean(self.loss_function(Q_s, (self.batch_reward + self.gamma * Q_sp)))
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
            t += 1
            self.frame_counter += 1
            # Save the model, update Target model and update posterior
            if self.frame_counter > self.replay_start_size:
                if self.frame_counter % self.Target_update == 0:
                    #check_point =  self.frame_counter / (self.Target_update * 100)
                    self.target_dqn_.load_state_dict(self.dqn_.state_dict())
                    self.c_t += 1
                    if self.c_t == self.target_W_update:
                        self.phiphiT, self.phiY, self.E_W, self.Cov_W = self.BayesReg(self.phiphiT, self.phiY, self.alpha_target, self.target_batch_size)
                        self.E_W_target = self.E_W
                        self.Cov_W_target = self.Cov_W
                        self.c_t = 0
                        for ii in range(self.num_action):
                            self.Cov_W_decom[ii] = torch.tensor(
                                torch.linalg.cholesky(((self.Cov_W[ii] + self.Cov_W[ii].T) / 2.)),
                                device=self.ctx)
                    if len(self.replay_memory.memory) < 100000:
                        self.target_batch_size = len(self.replay_memory.memory)
                    else:
                        self.target_batch_size = 100000
            if done:
                if self.epis_count % 10. == 0.:
                    logging.error(
                        'BDQN:env:%s,epis[%d],durat[%d],fnum=%d, cum_cl_rew = %d, cum_rew = %d,tot_cl = %d , tot = %d' \
                        % (self.env_name, self.epis_count, t + 1, self.frame_counter, self.cum_clipped_reward, self.cum_reward,
                           self.moving_average_clipped, self.moving_average))

    def training(self):

        while self.epis_count < self.max_episode:
            self.train_one_episode()
            if self.flag:
                print('break')
                break

            self.epis_count += 1
            self.tot_clipped_reward = np.append(self.tot_clipped_reward, self.cum_clipped_reward)
            self.tot_reward = np.append(self.tot_reward, self.cum_reward)
            self.frame_count_record = np.append(self.frame_count_record, self.frame_counter)
            if self.epis_count > 100.:
                self.moving_average_clipped = np.mean(self.tot_clipped_reward[int(self.epis_count) - 1 - 100:int(self.epis_count) - 1])
                self.moving_average = np.mean(self.tot_reward[int(self.epis_count) - 1 - 100:int(self.epis_count) - 1])
        self.plotting()

    def plotting(self):
        tot_c = self.tot_clipped_reward
        tot = self.tot_reward
        fram = self.frame_count_record
        epis_count = len(fram)

        bandwidth = 1  # Moving average bandwidth
        total_clipped = np.zeros(int(epis_count) - bandwidth)
        total_rew = np.zeros(int(epis_count) - bandwidth)
        f_num = fram[0:epis_count - bandwidth]

        for i in range(int(epis_count) - bandwidth):
            total_clipped[i] = np.sum(tot_c[i:i + bandwidth]) / bandwidth
            total_rew[i] = np.sum(tot[i:i + bandwidth]) / bandwidth

        t = np.arange(int(epis_count) - bandwidth)
        belplt = plt.plot(f_num, total_rew[0:int(epis_count) - bandwidth], "b", label="BDQN")

        plt.legend()
        print('Running after %d number of episodes' % epis_count)
        plt.xlabel("Number of steps")
        plt.ylabel("Average Reward per episode")
        plt.title("%s" % (self.env_name))
        plt.show()

env = "BreakoutNoFrameskip-v4"

Learner = BDQN_Learner(env, skip = True)

Learner.training()



