import gym
import numpy as np

NUM_OF_EPISODES = 100000
MAX_STEP_PER_EPISODE = 100
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


game_name = 'Taxi-v2'
env = gym.make(game_name)
env._max_episode_steps = MAX_STEP_PER_EPISODE
Q_table = np.zeros(shape=(env.observation_space.n, env.action_space.n))
for episode in range(1, NUM_OF_EPISODES+1):
    LR = 1.0/episode
    s = env.reset()
    episode_hist = []
    reward_hist = []
    while True:
        if np.random.uniform() <= EPSILON:
            a = np.random.choice(env.action_space.n)
        else:
            a = np.argmax(Q_table[s, :])
        s1, r, d, _ = env.step(a)
        episode_hist.append((s, a))
        reward_hist.append(r)
        if d is True:
            episode_hist.append(None)
            reward_hist.append(0)
            break
        s = s1
    lt_rewards = discount_rewards(np.array(reward_hist))
    # Delete the last ((s, a), r) tuple
    episode_hist.pop()
    lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)
    # Only use the first visit of each (s, a) tuple
    visited_state_actions = set()
    for i, s_a in enumerate(episode_hist):
        if s_a not in visited_state_actions:
            visited_state_actions.add(s_a)
            s, a = s_a
            Q_table[s, a] = Q_table[s, a] + LR*(lt_rewards[i] - Q_table[s, a])

try:
    Q_old_table = np.loadtxt(game_name + ".csv", delimiter=';')
    Q_table = (Q_table + Q_old_table)/2.0
except IOError:
    pass
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
