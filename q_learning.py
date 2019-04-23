import gym
import numpy as np
# Register my gym environments
import rl_library

NUM_OF_EPISODES = 100000
STEP_PER_EPISODE_MAX = 1000
STEP_PER_EPISODE_MIN = 100
GAMMA = 0.95
EPSILON = 0.1


def discount_rewards(reward):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(reward, dtype=np.float64)
    running_add = 0.0
    for t in reversed(range(0, reward.size)):
        running_add = running_add * GAMMA + reward[t]
        discounted_r[t] = running_add
    return discounted_r

# GAME_NAME = 'EasyBlackJack-v0'
GAME_NAME = 'Taxi-v2'
ENV = gym.make(GAME_NAME)
Q_table = np.zeros(shape=(ENV.observation_space.n, ENV.action_space.n), dtype=np.float64)
# reward_sum = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float64)
# reward_count = np.zeros(shape=(env.observation_space.n, env.action_space.n), dtype=np.float64)
for episode in range(1, NUM_OF_EPISODES+1):
    LR = episode**(-0.6)
    STEP_PER_EPISODE = int(STEP_PER_EPISODE_MIN + (
            (STEP_PER_EPISODE_MAX - STEP_PER_EPISODE_MIN)/NUM_OF_EPISODES)*(episode - 1))
    ENV._max_episode_steps = STEP_PER_EPISODE
    s = ENV.reset()
    episode_hist = []
    reward_hist = []
    while True:
        if np.random.uniform() <= EPSILON:
            a = np.random.choice(ENV.action_space.n)
        else:
            a = np.argmax(Q_table[s, :])
        s1, r, d, _ = ENV.step(a)
        episode_hist.append((s, a))
        reward_hist.append(r)
        if d is True:
            episode_hist.append(None)
            reward_hist.append(0)
            break
        s = s1
    lt_rewards = discount_rewards(np.array(reward_hist))
    # Delete the last (s,a), r
    episode_hist.pop()
    lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)
    # Only use the first visit of each (s, a) tuple
    visited_state_actions = set()
    for i, s_a in enumerate(episode_hist):
        if s_a not in visited_state_actions:
            visited_state_actions.add(s_a)
            s, a = s_a
            # reward_sum[s, a] += lt_rewards[i]
            # reward_count[s, a] += 1.0
            # Q_table[s, a] = reward_sum[s, a]/reward_count[s, a]
            Q_table[s, a] = Q_table[s, a] + LR*(lt_rewards[i] - Q_table[s, a])

try:
    Q_old_table = np.loadtxt(GAME_NAME + ".csv", delimiter=';')
    Q_table = (Q_table + Q_old_table)/2.0
except IOError:
    pass
np.savetxt(GAME_NAME + ".csv", Q_table, delimiter=';')

if False:
    s = ENV.reset()
    ENV.render()
    while True:
        a = np.argmax(Q_table[s, :])
        s1, r, d, _ = ENV.step(a)
        ENV.render()
        if d is True:
            break
        s = s1