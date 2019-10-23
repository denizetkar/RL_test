import math
import random
import warnings
import weakref
from collections import namedtuple
from functools import wraps

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


def get_ith_index(iterable, i):
    return list(map(lambda x: x[i], iterable))


def td_lambda_returns(rewards, state_values, gamma, gae_lambda):
    gae = torch.tensor(0.0, device=rewards.device)
    delta = rewards[:-1] + gamma * state_values[1:] - state_values[:-1]
    td_lambda_targets = delta
    if gae_lambda > 0:
        for t in reversed(range(rewards.size(0) - 1)):
            gae = delta[t] + gamma * gae_lambda * gae
            td_lambda_targets[t] = gae + state_values[t]
    return td_lambda_targets


class CosineLogAnnealingLR:

    def __init__(self, optimizer, T_max, eta_min=0, log_order=1.0, last_epoch=-1):
        self.T_max = T_max
        self.eta_min = eta_min
        # IF log_order == 0 THEN the scheduling is same as torch.optim.lr_scheduler.CosineAnnealingLR
        self.log_order = max(log_order, -2.5)
        self.T_logT_max = T_max * math.log2(T_max + 2)**log_order

        if not isinstance(optimizer, torch.optim.Adam.__bases__):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
            last_epoch = 0
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def with_counter(method):
            if getattr(method, '_with_counter', False):
                # `optimizer.step()` has already been replaced, return.
                return method

            # Keep a weak reference to the optimizer instance to prevent
            # cyclic references.
            instance_ref = weakref.ref(method.__self__)
            # Get the unbound method for the same purpose.
            func = method.__func__
            cls = instance_ref().__class__
            del method

            @wraps(func)
            def wrapper(*args, **kwargs):
                instance = instance_ref()
                instance._step_count += 1
                wrapped = func.__get__(instance, cls)
                return wrapped(*args, **kwargs)

            # Note that the returned function here is no longer a bound method,
            # so attributes like `__func__` and `__self__` no longer exist.
            wrapper._with_counter = True
            return wrapper

        self.optimizer.step = with_counter(self.optimizer.step)
        self.optimizer._step_count = 0
        self._step_count = 0
        self.step(last_epoch)

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_scale(self, epoch):
        return (1 - math.cos(math.pi * (-epoch + self.T_max) * math.log2(
            (-epoch + self.T_max) + 2) ** self.log_order / self.T_logT_max)) / 2

    def get_lr(self):
        return [self.eta_min + (base_lr - self.eta_min) * self.get_scale(self.last_epoch)
                for base_lr in self.base_lrs]

    def step(self, epoch=None):
        # Raise a warning if old pattern is detected
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            if not hasattr(self.optimizer.step, "_with_counter"):
                warnings.warn("Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                              "initialization. Please, make sure to call `optimizer.step()` before "
                              "`lr_scheduler.step()`. See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)

            # Just check if there were two first lr_scheduler.step() calls before optimizer.step()
            elif self.optimizer._step_count < 1:
                warnings.warn("Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                              "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                              "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                              "will result in PyTorch skipping the first value of the learning rate schedule."
                              "See more details at "
                              "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate", UserWarning)
        self._step_count += 1

        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
