import random
from collections import namedtuple
import numpy as np


class ReplayMemory(object):

    def __init__(self, capacity, field_names=('state', 'action', 'reward', 'next_state')):
        self.Transition = namedtuple('Transition', field_names)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


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
