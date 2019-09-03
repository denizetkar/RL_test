import gym
import numpy as np
import tensorflow as tf
from rl_library import rl_agents, helper
import matplotlib.pyplot as plt

NUM_OF_EPISODES = 1000
MAX_STEP_PER_EPISODE = 1000
GAMMA = 1.0


GAME_NAME = 'CartPole-v1'
ENV = gym.make(GAME_NAME)
ENV._max_episode_steps = MAX_STEP_PER_EPISODE
with tf.device('/cpu:0'):
    q_agent = rl_agents.PolicyGradientContStateDiscActionAgent(
        ENV.observation_space.shape[0],
        ENV.action_space.n,
        8*ENV.observation_space.shape[0])
    # Add ops to save and restore all the variables.
    saver = tf.train.Saver(tf.trainable_variables())
episode_reward_list = []

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        # Restore variables from disk.
        saver.restore(sess, "tf_models/" + GAME_NAME + ".ckpt")
    except ValueError:
        pass

    for episode in range(1, NUM_OF_EPISODES+1):
        LR, episodic_reward, s, episode_hist = 0.01 * episode**(-0.3), 0.0, ENV.reset(), []
        # ENV.render()
        while True:
            a_probs = sess.run(q_agent.action_probs, feed_dict={q_agent.state_holder: np.expand_dims(s, axis=0)})[0]
            a = np.random.choice(ENV.action_space.n, p=a_probs)
            s1, r, d, _ = ENV.step(a)
            # ENV.render()
            episodic_reward += r
            episode_hist.append((s, a, r))
            if d is True:
                episode_hist.append((None, None, 0))
                break
            s = s1

        lt_rewards = helper.discount_rewards(list(zip(*episode_hist))[2], GAMMA)
        # Delete the last (s, a, r) tuple
        episode_hist.pop()
        lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)
        # Only use the first visit of each (s, a) tuple
        visited_state_actions = set()
        visited_states, visited_actions, visited_lt_rewards = [], [], []
        for i, (s, a, _) in enumerate(episode_hist):
            if (tuple(s), a) not in visited_state_actions:
                visited_state_actions.add((tuple(s), a))
                visited_states.append(s)
                visited_actions.append(a)
                visited_lt_rewards.append(lt_rewards[i])

        feed_dict = {q_agent.state_holder: np.array(visited_states),
                     q_agent.action_holder: np.array(visited_actions),
                     q_agent.lt_reward_holder: np.array(visited_lt_rewards),
                     q_agent.learning_rate: LR}
        _ = sess.run(q_agent.optimize, feed_dict=feed_dict)

        episode_reward_list.append(episodic_reward)
    saver.save(sess, "tf_models/" + GAME_NAME + ".ckpt", write_meta_graph=False, write_state=False)

plt.plot(list(range(1, len(episode_reward_list)+1)), episode_reward_list)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
