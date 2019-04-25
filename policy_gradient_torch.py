import gym
import torch
import numpy as np
from rl_library import rl_agents
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

NUM_OF_EPISODES = 5000
MAX_STEP_PER_EPISODE = 1000
GAMMA = 1.0


def discount_rewards(reward):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(reward, dtype=np.float64)
    running_add = 0.0
    for t in reversed(range(0, reward.size)):
        running_add = running_add * GAMMA + reward[t]
        discounted_r[t] = running_add
    return discounted_r


GAME_NAME = 'CartPole-v1'
ENV = gym.make(GAME_NAME)
ENV._max_episode_steps = MAX_STEP_PER_EPISODE
q_agent = rl_agents.PolicyGradientContStateDiscActionAgentTorch(
    ENV.observation_space.shape[0], ENV.action_space.n,
    8*ENV.observation_space.shape[0])
try:
    q_agent.load_state_dict(torch.load("torch_models/" + GAME_NAME + ".model"))
except FileNotFoundError:
    pass
episode_reward_list = []
for episode in range(1, NUM_OF_EPISODES+1):
    LR = 0.01 * episode**(-0.3)
    s = ENV.reset()
    # ENV.render()
    episode_hist = []
    reward_hist = []
    episodic_reward = 0.0
    while True:
        log_a_probs = q_agent.forward(torch.from_numpy(np.expand_dims(s, axis=0)))[0]
        a = np.random.choice(ENV.action_space.n, p=log_a_probs.exp().data.numpy())
        s1, r, d, _ = ENV.step(a)
        # ENV.render()
        episodic_reward += r
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
    visited_states, visited_actions, visited_lt_rewards = [], [], []
    for i, (s, a) in enumerate(episode_hist):
        if (tuple(s), a) not in visited_state_actions:
            visited_state_actions.add((tuple(s), a))
            visited_states.append(s)
            visited_actions.append(a)
            visited_lt_rewards.append(lt_rewards[i])

    q_agent.zero_grad()
    log_action_probs = q_agent.forward(torch.tensor(visited_states))
    ids = torch.tensor(visited_actions).long()
    log_policy_outputs = log_action_probs.gather(1, ids.view(-1, 1))
    loss = -(log_policy_outputs.squeeze(1) * torch.tensor(visited_lt_rewards)).mean()
    loss.backward()
    # Update the weights
    for f in q_agent.parameters():
        f.data.sub_(f.grad.data * LR)
    episode_reward_list.append(episodic_reward)
torch.save(q_agent.state_dict(), "torch_models/" + GAME_NAME + ".model")
plt.plot(list(range(1, len(episode_reward_list)+1)), episode_reward_list)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
