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
EXPERIENCE_REUSE_FACTOR = 2


GAME_NAME = 'CartPole-v1'
ENV = gym.make(GAME_NAME)
ENV._max_episode_steps = MAX_STEP_PER_EPISODE
q_agent = rl_agents.ActorCriticContStateDiscActionAgentTorch(
    ENV.observation_space.shape[0],
    ENV.action_space.n,
    8*ENV.observation_space.shape[0])
try:
    q_agent.load_state_dict(torch.load("torch_models/" + GAME_NAME + ".acer_model"))
except FileNotFoundError:
    pass
episode_reward_list = []
transition_memory = helper.ReplayMemory(EXPERIENCE_REUSE_FACTOR * MAX_STEP_PER_EPISODE,
                                        ('state', 'action', 'reward', 'next_state', 'log_a_prob'))

for episode in range(1, NUM_OF_EPISODES+1):
    LR, episodic_reward, s, episode_hist = 0.05 * episode**(-0.5), 0.0, ENV.reset(), []
    # ENV.render()
    step = 0
    while True:
        log_a_prob, v = q_agent(torch.as_tensor(s))
        a = np.random.choice(ENV.action_space.n, p=log_a_prob.exp().data.numpy())
        s1, r, d, _ = ENV.step(a)
        # ENV.render()
        episodic_reward += r
        episode_hist.append((s, a, r, s1, v, log_a_prob))
        step += 1
        if d is True:
            if step >= MAX_STEP_PER_EPISODE:
                _, v = q_agent(torch.as_tensor(s1))
                episode_hist.append((None, None, v.item(), None, None, None))
            else:
                episode_hist.append((None, None, 0, None, None, None))
            break
        s = s1

    lt_rewards = helper.discount_rewards(list(zip(*episode_hist))[2], GAMMA)
    # Delete the last (s, a, r) tuple
    episode_hist.pop()
    lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)

    # DO NOT only use the first visit of each (s, a) tuple
    visited_transitions = []
    for i, (s, a, r, s1, v, log_a_prob) in enumerate(episode_hist):
        visited_transitions.append([a, lt_rewards[i], v, log_a_prob])
        transition_memory.push(s, a, r, s1, log_a_prob.detach())

    for i in range(EXPERIENCE_REUSE_FACTOR):
        if i == 0:
            # ON-POLICY
            training_actions, training_lt_rewards, \
                training_values, training_log_a_probs = zip(*visited_transitions)
            training_values = torch.cat(training_values)
            training_log_a_probs = torch.stack(training_log_a_probs)

            ids = torch.as_tensor(training_actions).long()
            log_policy_outputs = training_log_a_probs.gather(1, ids.view(-1, 1))
            training_lt_rewards = torch.as_tensor(training_lt_rewards)
            policy_loss = -(log_policy_outputs.squeeze(1) *
                            (training_lt_rewards - training_values.detach())).mean()
            value_loss = F.smooth_l1_loss(training_values, training_lt_rewards)
            # Prevent overuse of the data
            if len(transition_memory) <= (EXPERIENCE_REUSE_FACTOR - 1) * len(visited_transitions):
                break
        else:
            # OFF-POLICY
            training_samples = transition_memory.sample(len(visited_transitions))
            training_states, training_actions, training_rewards, \
                training_next_states, behavior_log_a_probs = zip(*training_samples)
            training_log_a_probs, training_values = q_agent(torch.as_tensor(training_states))
            _, training_next_values = q_agent(torch.as_tensor(training_next_states))

            training_values, training_next_values = training_values.squeeze(1), training_next_values.squeeze(1)
            behavior_log_a_probs = torch.stack(behavior_log_a_probs)
            training_value_targets = torch.as_tensor(training_rewards) + GAMMA * training_next_values.detach()

            ids = torch.as_tensor(training_actions).long()
            log_policy_outputs = training_log_a_probs.gather(1, ids.view(-1, 1)).squeeze(1)
            log_behavior_policy_outputs = behavior_log_a_probs.gather(1, ids.view(-1, 1)).squeeze(1)
            rho = (log_policy_outputs.exp() / log_behavior_policy_outputs.exp()).detach()
            policy_loss = -(rho * log_policy_outputs *
                            (training_value_targets - training_values.detach())).mean()
            value_loss = F.smooth_l1_loss(training_values, rho * training_value_targets)

        loss = policy_loss + value_loss

        q_agent.zero_grad()
        loss.backward()
        # Update the weights
        for f in q_agent.parameters():
            f.data.sub_(f.grad.data * LR)

    episode_reward_list.append(episodic_reward)

torch.save(q_agent.state_dict(), "torch_models/" + GAME_NAME + ".acer_model")
plt.plot(list(range(1, len(episode_reward_list)+1)), episode_reward_list)
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
