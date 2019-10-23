import tensorflow as tf
import torch.nn as nn
import torch.nn.functional as F

from rl_library import helper


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
        hidden_layer_sizes = helper.get_ith_index(hidden_layers, 0)
        dropout_rates = helper.get_ith_index(hidden_layers, 1)
        use_batch_layers = helper.get_ith_index(hidden_layers, 2)
        first_hidden_layer = nn.Linear(s_dim, hidden_layer_sizes[0])
        self.hidden_layers = nn.ModuleList(
            [first_hidden_layer] + [nn.Linear(hidden_layer_sizes[i], hidden_layer_sizes[i + 1])
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
