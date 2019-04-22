import tensorflow as tf


class QLearningAgent:
    def __init__(self, s_size, a_size, io_dataset):
        self.iterator = io_dataset.make_initializable_iterator()
        self.state_action_holder, self.true_Q_val = self.iterator.get_next()
        self.Q_table = tf.Variable(tf.zeros(shape=(s_size, a_size)), dtype=tf.float32)
        self.Q_val = tf.gather_nd(self.Q_table, self.state_action_holder)

        self.state_holder = tf.placeholder(tf.int32, shape=())
        self.Q_actions_table = self.Q_table[self.state_holder, :]
        self.best_action = tf.argmax(self.Q_actions_table)

        self.loss = tf.losses.mean_squared_error(self.true_Q_val, self.Q_val)
        self.learning_rate = tf.placeholder(tf.float32, shape=())
        self.optimize = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
