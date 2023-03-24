#!/usr/bin/env python3
#
# An annotated transcription of the code from:
# "Deep Q learning is Simple with PyTorch | Full Tutorial 2020"
# by Machine Learning with Phil
# https://www.youtube.com/watch?v=wc-FxNENg9U&t=2s

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims   = fc1_dims
        self.fc2_dims   = fc2_dims
        self.n_actions  = n_actions

        # build the network
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        # configure the optimizer
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self,state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # no relu because we want a raw number
        actions = self.fc3(x)

        return actions


class Agent():
    ## Hyper Parameters
    # gamma - weighting of future rewards
    # epsilon - determine exploit vs exlore
    # lr - learning rate to pass into the network
    # input_dims - size of the state we are looking at
    # batch_size - we are going to learn from a batch of sampled actions
    # n_actions - number of actions we can take
    # max_mem_size - max size of the batch of memory
    # eps_end - lowest epsilon we will decrement to
    # eps_dec - how much to decrement epsilon
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100000, eps_end=0.01, eps_dec=5e-4):
        self.gamma   = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_dec = eps_dec
        self.lr      = lr

        # this will make it easy to select a random action
        self.action_space = [i for i in range(n_actions)]

        self.mem_size   = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr   = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions = n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)

        # we need a replay memory
        self.state_memory     = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        # what is the current value of the next state via the current state.
        # model free (we don't need to know the model),
        # bootstrapped (construct estimates of action-value functions, use one estimate to produce a new estimate)
        # off policy (use a policy to generate actions, use epsilon to generate actions from purely greedy)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)

        # deep q only really works for discrete action spaces
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)

        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        # future value of terminal state is 0, so we need to keep track of it
        # we will pass the done flags from the environment
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward,state_, done):
        # we have a circular buffer for our memory
        index = self.mem_cntr % self.mem_size

        self.state_memory[index]     = state
        self.new_state_memory[index] = state_
        self.reward_memory[index]    = reward
        self.action_memory[index]    = action
        self.terminal_memory[index]  = done

        # increment position in memory
        self.mem_cntr += 1

    def choose_action(self, observation):
        # if we are greater than epsilon the lets be
        # greedy and pass our state to the network
        # to determine the actions with the largest
        # predicted reward.
        if np.random.random() > self.epsilon:
            # preprocess the observation to make it pytorch digestable
            state = T.tensor([observation]).to(self.Q_eval.device)
            # pass the state through the forward pass of the network
            actions = self.Q_eval.forward(state)
            # pick the action with the highest value
            action =  T.argmax(actions).item()

        # otherwise we want to take a random action
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        # we do not have enough samples
        if self.mem_cntr < self.batch_size:
            return

        # pytorch particularity
        self.Q_eval.optimizer.zero_grad()

        # need to calculate the number of actions we actually have
        max_mem = min(self.mem_cntr, self.mem_size)

        # we want to grab a batch of random transitions to train on without replacement
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        # ...? Needed for proper slicing
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        # selected chunks from the various memories based on our sampling
        state_batch     = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch    = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch  = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        # preform feed forwards to get the estimates for our loss
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]

        # if we had a target network it would be here
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0

        # current reward plus the maximum estimate of next state
        q_target = reward_batch + self.gamma * T.max(q_next, dim=1)[0]

        # how far off is our estimate and
        # let's backprogate the loss and adjust the network
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        # each time we learn we decrement epsilon
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
