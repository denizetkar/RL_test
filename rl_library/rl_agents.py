import copy

import tensorflow as tf
import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F


class QLearningDiscStateDiscActionAgent:
    def __init__(self, s_size, a_size, io_dataset):
        self.iterator = io_dataset.make_initializable_iterator()
        self.state_action_holder, self.true_Q_val = self.iterator.get_next()
        self.Q_table = tf.Variable(tf.zeros(shape=(s_size, a_size), dtype=tf.float64))
        self.Q_val = tf.gather_nd(self.Q_table, self.state_action_holder)

        self.state_holder = tf.placeholder(tf.int32, shape=())
        self.Q_actions_table = self.Q_table[self.state_holder, :]
        self.best_action = tf.argmax(self.Q_actions_table)

        self.loss = tf.losses.mean_squared_error(self.true_Q_val, self.Q_val)
        self.learning_rate = tf.placeholder(tf.float64, shape=())
        self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


class PolicyGradientContStateDiscActionAgent:
    def __init__(self, s_dim, a_size, hidden_layer_size):
        self.state_holder = tf.placeholder(tf.float64, shape=(None, s_dim))
        self.action_holder = tf.placeholder(tf.int32, shape=(None,))
        self.lt_reward_holder = tf.placeholder(tf.float64, shape=(None,))
        self.hidden_layer = tf.keras.layers.Dense(
            hidden_layer_size,
            activation=tf.keras.activations.selu,
            kernel_initializer=tf.keras.initializers.lecun_normal()).apply(self.state_holder)
        self.action_probs = tf.keras.layers.Dense(
            a_size,
            activation=tf.keras.activations.softmax,
            kernel_initializer=tf.keras.initializers.lecun_normal()).apply(self.hidden_layer)
        self.indexes = tf.range(0, tf.shape(self.action_probs)[0]) * tf.shape(self.action_probs)[1] + self.action_holder
        self.policy_outputs = tf.gather_nd(tf.reshape(self.action_probs, (-1,)), tf.expand_dims(self.indexes, axis=1))

        self.loss = -tf.reduce_mean(tf.log(self.policy_outputs) * self.lt_reward_holder)
        self.learning_rate = tf.placeholder(tf.float64, shape=())
        self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)


class PolicyGradientContStateDiscActionAgentTorch(nn.Module):
    def __init__(self, s_dim, a_size, hidden_layer_size):
        super().__init__()
        self.hidden_layer = nn.Linear(s_dim, hidden_layer_size)
        self.action_layer = nn.Linear(hidden_layer_size, a_size)

    def forward(self, state_input):
        x = self.hidden_layer(state_input)
        x = F.selu(x)
        x = self.action_layer(x)
        x = F.log_softmax(x, dim=-1)
        return x


class ActorCriticContStateDiscActionAgentTorch(nn.Module):
    def __init__(self, s_dim, a_size, hidden_layer_size):
        super().__init__()
        self.hidden_layer = nn.Linear(s_dim, hidden_layer_size)
        self.action_layer = nn.Linear(hidden_layer_size, a_size)
        self.value_layer = nn.Linear(hidden_layer_size, 1)

    def forward(self, state_input):
        x = self.hidden_layer(state_input)
        x = F.leaky_relu(x)
        action_scores = self.action_layer(x)
        log_a_probs = F.log_softmax(action_scores, dim=-1)
        state_values = self.value_layer(x)
        return log_a_probs, state_values


class PPOContStateDiscActionAgentTorch(nn.Module):
    def __init__(self, s_dim, a_size, hidden_layers):
        super().__init__()
        hidden_layer_sizes, dropout_rates, use_batch_layers = zip(*hidden_layers)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(s_dim, hidden_layer_sizes[0])] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                                                         for i in range(len(hidden_layers) - 1)])
        self.bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_layer_size) if use_batch_layers[layer_num] else nn.Identity()
             for layer_num, hidden_layer_size in enumerate(hidden_layer_sizes)])
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for dropout in dropout_rates])
        # define final layers (action/value)
        self.action_layer = nn.Linear(hidden_layer_sizes[-1], a_size)
        self.value_layer = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, states):
        x = states
        try:
            for hidden_layer, bn_layer, dropout in zip(self.hidden_layers, self.bn_layers, self.dropouts):
                x = F.leaky_relu(hidden_layer(x))
                x = dropout(bn_layer(x))
        except ValueError:
            # Error is most likely due to giving 1 sample to batchnorm layer
            return None, None
        action_scores = self.action_layer(x)
        log_a_probs = F.log_softmax(action_scores, dim=-1)
        state_values = self.value_layer(x)
        return log_a_probs, state_values


class PPOExpContStateDiscActionAgentTorch(nn.Module):
    def __init__(self, s_dim, a_size, hidden_layers):
        super().__init__()
        hidden_layer_sizes, dropout_rates, use_batch_layers = zip(*hidden_layers)
        self.dropouts = nn.ModuleList([nn.Dropout(dropout) for dropout in dropout_rates])
        # define action (actor) network layers
        self.action_hidden_layers = nn.ModuleList(
            [nn.Linear(s_dim, hidden_layer_sizes[0])] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
                                                         for i in range(len(hidden_layers) - 1)])
        self.action_bn_layers = nn.ModuleList(
            [nn.BatchNorm1d(hidden_layer_size) if use_batch_layers[layer_num] else nn.Identity()
             for layer_num, hidden_layer_size in enumerate(hidden_layer_sizes)])
        self.action_layer = nn.Linear(hidden_layer_sizes[-1], a_size)
        # define value (critic) network layers
        self.value_hidden_layers = copy.deepcopy(self.action_hidden_layers)
        self.value_bn_layers = copy.deepcopy(self.action_bn_layers)
        self.value_layer = nn.Linear(hidden_layer_sizes[-1], 1)

    def forward(self, states, actor=True):
        x = states
        if actor:
            hidden_layers = self.action_hidden_layers
            bn_layers = self.action_bn_layers
            last_layer = self.action_layer
            def last_op(tensor): return F.log_softmax(tensor, dim=-1)
        else:  # critic
            hidden_layers = self.value_hidden_layers
            bn_layers = self.value_bn_layers
            last_layer = self.value_layer
            def last_op(tensor): return tensor
        try:
            for hidden_layer, bn_layer, dropout in zip(hidden_layers, bn_layers, self.dropouts):
                x = F.selu(hidden_layer(x))
                x = dropout(bn_layer(x))
        except ValueError:
            # Error is most likely due to giving 1 sample to batchnorm layer
            return None
        x = last_layer(x)
        return last_op(x)

    def actor_losses(self, old_states, old_actions, old_log_probs, old_advantages, eps_clip):
        log_probs = self(old_states, True)
        if log_probs is None:
            return None, None
        log_action_probs = log_probs.gather(1, old_actions.view(-1, 1)).squeeze(1)
        dist_entropy = Categorical(log_probs).entropy()

        # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(log_action_probs - old_log_probs)

        # Finding Surrogate Loss:
        surr1 = ratios * old_advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * old_advantages
        policy_loss = -torch.min(surr1, surr2)
        entropy_loss = -dist_entropy

        return policy_loss, entropy_loss

    def critic_losses(self, old_states, old_lt_rewards, old_state_values, eps_clip):
        state_values = self(old_states, False)
        if state_values is None:
            return None
        state_values = state_values.squeeze(1)

        # Finding value loss:
        state_value_clipped = state_values + torch.clamp(state_values - old_state_values,
                                                         -eps_clip, eps_clip)
        v_loss1 = F.smooth_l1_loss(state_values, old_lt_rewards)
        v_loss2 = F.smooth_l1_loss(state_value_clipped, old_lt_rewards)
        value_loss = torch.max(v_loss1, v_loss2)

        return value_loss
