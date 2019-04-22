import gym
import numpy as np
import tensorflow as tf
import rl_agents

NUM_OF_EPISODES = 100
MAX_STEP_PER_EPISODE = 1000000
GAMMA = 0.95
EPSILON = 0.1


def discount_rewards(reward):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(reward, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, reward.size)):
        running_add = running_add * GAMMA + reward[t]
        discounted_r[t] = running_add
    return discounted_r


def episode_generator():
    s = env.reset()
    episode_hist = []
    reward_hist = []
    while True:
        if np.random.uniform() <= EPSILON:
            a = np.random.choice(env.action_space.n)
        else:
            a = sess.run(q_agent.best_action, feed_dict={q_agent.state_holder: s})
        s1, r, d, _ = env.step(a)
        episode_hist.append((s, a))
        reward_hist.append(r)
        if d is True:
            break
        s = s1
    lt_rewards = discount_rewards(np.array(reward_hist))
    episode_hist = np.array(episode_hist)
    yield from zip(episode_hist, lt_rewards)


game_name = 'Taxi-v2'
env = gym.make(game_name)
env._max_episode_steps = MAX_STEP_PER_EPISODE
episode_dataset = tf.data.Dataset.from_generator(
    episode_generator,
    output_types=(tf.int32, tf.float32),
    output_shapes=((2,), ())).batch(batch_size=1)
q_agent = rl_agents.QLearningAgent(env.observation_space.n, env.action_space.n, episode_dataset)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for episode in range(1, NUM_OF_EPISODES+1):
        LR = episode**(-0.6)
        sess.run(q_agent.iterator.initializer)
        _ = sess.run(q_agent.optimize, feed_dict={q_agent.learning_rate: LR})
    Q_table = sess.run(q_agent.Q_table)
np.savetxt(game_name + ".csv", Q_table, delimiter=';')

s = env.reset()
env.render()
while True:
    a = np.argmax(Q_table[s, :])
    s1, r, d, _ = env.step(a)
    env.render()
    if d is True:
        break
    s = s1
