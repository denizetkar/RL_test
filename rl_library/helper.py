import random
from collections import namedtuple

import numpy as np
import torch


class ReplayMemory(object):

    def __init__(self, capacity, field_names=('state', 'action', 'reward', 'next_state')):
        self._Transition = namedtuple('Transition', field_names)
        self._capacity = capacity
        self.memory = []
        self._position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self._capacity:
            self.memory.append(None)
        self.memory[self._position] = self._Transition(*args)
        self._position = (self._position + 1) % self._capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def clear(self):
        self.memory.clear()
        self._position = 0


def discount_rewards(reward, gamma):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(reward, dtype=np.float64)
    running_add = 0.0
    for t in reversed(range(0, len(reward))):
        running_add = running_add * gamma + reward[t]
        discounted_r[t] = running_add
    return discounted_r


def advantage_function(reward, state_value, gamma):
    advantage_f = np.zeros(len(reward) - 1)
    for t in range(0, len(advantage_f)):
        advantage_f[t] = reward[t] + gamma * state_value[t+1].item() - state_value[t].item()
    return advantage_f


def zip_ith_index(iterable, i):
    return list(map(lambda x: x[i], iterable))


def td_lambda_returns(rewards, state_values, gamma, gae_lambda):
    gae = torch.tensor(0.0, device=rewards.device)
    delta = rewards[:-1] + gamma * state_values[1:] - state_values[:-1]
    td_lambda_targets = torch.zeros(rewards.size(0) - 1, device=rewards.device)
    for t in reversed(range(rewards.size(0) - 1)):
        gae = delta[t] + gamma * gae_lambda * gae
        td_lambda_targets[t] = gae + state_values[t]
    return td_lambda_targets
