import gym
import numpy as np
import tensorflow as tf
from rl_library import rl_agents

NUM_OF_EPISODES = 5000
MAX_STEP_PER_EPISODE = 100
GAMMA = 0.95
EPSILON = 0.1


def discount_rewards(reward):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(reward, dtype=np.float64)
    running_add = 0.0
    for t in reversed(range(0, len(reward))):
        running_add = running_add * GAMMA + reward[t]
        discounted_r[t] = running_add
    return discounted_r


def episode_generator():
    s, episode_hist = ENV.reset(), []
    while True:
        if np.random.uniform() <= EPSILON:
            a = np.random.choice(ENV.action_space.n)
        else:
            a = sess.run(q_agent.best_action, feed_dict={q_agent.state_holder: s})
        s1, r, d, _ = ENV.step(a)
        episode_hist.append((s, a, r))
        if d is True:
            episode_hist.append((None, None, 0))
            break
        s = s1
    lt_rewards = discount_rewards(list(zip(*episode_hist))[2])
    # Delete the last (s, a, r) tuple
    episode_hist.pop()
    lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)
    # Only use the first visit of each (s, a) tuple
    visited_state_actions = set()
    visited_episode_hist = []
    for i, (s, a, _) in enumerate(episode_hist):
        if (s, a) not in visited_state_actions:
            visited_state_actions.add((s, a))
            visited_episode_hist.append(((s, a), lt_rewards[i]))
    yield from visited_episode_hist


# GAME_NAME = 'EasyBlackJack-v0'
GAME_NAME = 'Taxi-v2'
ENV = gym.make(GAME_NAME)
ENV._max_episode_steps = MAX_STEP_PER_EPISODE
with tf.device('/cpu:0'):
    episode_dataset = tf.data.Dataset.from_generator(
        episode_generator,
        output_types=(tf.int32, tf.float64),
        output_shapes=((2,), ())).batch(batch_size=MAX_STEP_PER_EPISODE).prefetch(128)
    q_agent = rl_agents.QLearningDiscStateDiscActionAgent(ENV.observation_space.n, ENV.action_space.n, episode_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for episode in range(1, NUM_OF_EPISODES+1):
        LR = episode**(-0.6)
        sess.run(q_agent.iterator.initializer)
        try:
            while True:
                _ = sess.run(q_agent.optimize, feed_dict={q_agent.learning_rate: LR})
        except tf.errors.OutOfRangeError:
            pass
    Q_table = sess.run(q_agent.Q_table)

try:
    Q_old_table = np.loadtxt("tabular_models/" + GAME_NAME + ".csv", delimiter=';')
    Q_table = (Q_table + Q_old_table)/2.0
except IOError:
    pass
np.savetxt("tabular_models/" + GAME_NAME + ".csv", Q_table, delimiter=';')
