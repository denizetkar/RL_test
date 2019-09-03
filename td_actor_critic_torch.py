import gym
import torch
import torch.nn.functional as F
import numpy as np
from rl_library import rl_agents, helper
import matplotlib.pyplot as plt
torch.set_default_dtype(torch.float64)

NUM_OF_EPISODES = 2000
MAX_STEP_PER_EPISODE = 1000
GAMMA = 0.99


GAME_NAME = 'CartPole-v1'
ENV = gym.make(GAME_NAME)
ENV._max_episode_steps = MAX_STEP_PER_EPISODE
q_agent = rl_agents.ActorCriticContStateDiscActionAgentTorch(
    ENV.observation_space.shape[0],
    ENV.action_space.n,
    8*ENV.observation_space.shape[0])
try:
    q_agent.load_state_dict(torch.load("torch_models/" + GAME_NAME + ".ac_td_model"))
except FileNotFoundError:
    pass
episode_reward_list = []

for episode in range(1, NUM_OF_EPISODES+1):
    LR, episodic_reward, s, episode_hist = 0.5 * episode**(-0.5), 0.0, ENV.reset(), []
    # ENV.render()
    step = 0
    while True:
        log_a_prob, v = q_agent(torch.as_tensor(s))
        a = np.random.choice(ENV.action_space.n, p=log_a_prob.exp().data.numpy())
        s1, r, d, _ = ENV.step(a)
        # ENV.render()
        episodic_reward += r
        episode_hist.append((s, a, r, v, log_a_prob))
        step += 1
        if d is True:
            if step >= MAX_STEP_PER_EPISODE:
                _, v = q_agent(torch.as_tensor(s1))
                episode_hist.append((None, None, v.item(), v, None))
            else:
                episode_hist.append((None, None, 0, torch.tensor(0.0), None))
            break
        s = s1

    lt_rewards = helper.discount_rewards(list(zip(*episode_hist))[2], GAMMA)
    advantage_f = helper.advantage_function(list(zip(*episode_hist))[2], list(zip(*episode_hist))[3], GAMMA)
    # Delete the last (s, a, r, v, log_a_prob) tuple
    episode_hist.pop()
    lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)

    # DO NOT only use the first visit of each (s, a) tuple
    _, visited_actions, _, visited_values, visited_log_a_probs = zip(*episode_hist)
    visited_lt_rewards, visited_advantage_f = lt_rewards, advantage_f

    visited_log_a_probs_torch = torch.stack(visited_log_a_probs)
    ids = torch.as_tensor(visited_actions).long()
    log_policy_outputs = visited_log_a_probs_torch.gather(1, ids.view(-1, 1))
    visited_lt_rewards_torch = torch.as_tensor(visited_lt_rewards)
    visited_advantage_f_torch = torch.as_tensor(visited_advantage_f)
    visited_values_torch = torch.cat(visited_values)
    policy_loss = -(log_policy_outputs.squeeze(1) *
                    visited_advantage_f_torch).mean()
    value_loss = F.smooth_l1_loss(visited_lt_rewards_torch, visited_values_torch)
    loss = policy_loss + value_loss

    q_agent.zero_grad()
    loss.backward()
    # Update the weights
    for f in q_agent.parameters():
        f.data.sub_(f.grad.data * LR)

    episode_reward_list.append(episodic_reward)

torch.save(q_agent.state_dict(), "torch_models/" + GAME_NAME + ".ac_td_model")
plt.plot(list(range(1, len(episode_reward_list)+1)), episode_reward_list)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
