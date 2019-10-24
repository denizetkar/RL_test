import random
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from rl_library import rl_agents, helper
torch.set_default_dtype(torch.float32)


def main():
    ############## Hyperparameters ##############
    env_name = 'CartPole-v1'
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = False
    max_total_step = 100000     # min number of steps to take during training
    episode_timesteps = 1000    # max time steps in one episode
    hidden_layers = [(64, 0.0, False), (64, 0.0, False)]  # list of (hidden_layer_size, dropout_rate, use_batch_layer)
    buffer_timestep = 2000      # min time steps in a training buffer
    batch_timestep = 1000       # time steps in a single update batch
    lr = 1e-2
    gamma = 0.99                # discount factor
    gae_lambda = 0.95           # lambda value for td(lambda) returns
    k_epochs = 5                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #############################################

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    env._max_episode_steps = episode_timesteps
    rl_agent = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, hidden_layers).to(device)
    rl_agent_old = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, hidden_layers).to(device)
    try:
        rl_agent.load_state_dict(torch.load("torch_models/" + env_name + ".ppo_model"))
    except FileNotFoundError:
        pass
    rl_agent_old.load_state_dict(rl_agent.state_dict())
    rl_agent_old.eval()
    optimizer = torch.optim.Adam(rl_agent.parameters(), lr=lr)
    lr_scheduler = helper.CosineLogAnnealingLR(
        optimizer, (max_total_step - 1) // buffer_timestep + 1, eta_min=0.0, log_order=0)

    episode_reward_list = []
    transition_memory = helper.ReplayMemory(buffer_timestep + episode_timesteps,
                                            ('state', 'action', 'lt_reward', 'log_prob', 'state_value'))
    buffer_step = total_step = 0
    # Start training loop
    while True:
        episodic_reward, state, episode_hist = 0.0, torch.as_tensor(env.reset(), dtype=torch.float, device=device), []
        if render:
            env.render()
        episode_step = 0
        # Start episode loop
        while True:
            log_prob, state_value = rl_agent_old(state.unsqueeze(0))
            log_prob, state_value = log_prob.detach().squeeze(0), state_value.detach().squeeze(0)
            dist = Categorical(logits=log_prob)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.as_tensor(next_state, dtype=torch.float, device=device)
            if render:
                env.render()
            episodic_reward += reward
            episode_hist.append((state, action, torch.tensor(reward, dtype=torch.float, device=device),
                                 dist.log_prob(action).detach(), state_value.squeeze(0).detach()))
            episode_step, buffer_step, total_step = episode_step + 1, buffer_step + 1, total_step + 1
            if done:
                if episode_step >= episode_timesteps:
                    _, state_value = rl_agent_old(next_state.unsqueeze(0))
                    state_value = state_value.detach().squeeze(0)
                    episode_hist.append(
                        (None, None, torch.tensor(np.nan, device=device), None, state_value.squeeze(0)))
                else:
                    episode_hist.append(
                        (None, None, torch.tensor(np.nan, device=device), None, torch.tensor(0.0, device=device)))
                break
            state = next_state
        episode_reward_list.append(episodic_reward)

        lt_rewards = helper.td_lambda_returns(
            torch.stack(helper.get_ith_index(episode_hist, 2)),
            torch.stack(helper.get_ith_index(episode_hist, 4)), gamma, gae_lambda)
        # Delete the last (s, a, r, log_prob, v) tuple
        episode_hist.pop()
        # DO NOT only use the first visit of each (s, a) tuple
        for i, (state, action, _, log_prob, state_value) in enumerate(episode_hist):
            transition_memory.push(state, action, lt_rewards[i], log_prob, state_value)

        if buffer_step >= buffer_timestep:
            old_states, old_actions, old_lt_rewards, old_log_probs, old_state_values = zip(*transition_memory.memory)
            buffer_size = len(transition_memory)
            transition_memory.clear()
            # convert lists to tensors
            old_states = torch.stack(old_states)
            old_actions = torch.as_tensor(old_actions, device=device)
            old_lt_rewards = torch.as_tensor(old_lt_rewards, device=device)
            old_log_probs = torch.as_tensor(old_log_probs, device=device)
            old_state_values = torch.as_tensor(old_state_values, device=device)
            # Normalizing the long term rewards
            old_lt_rewards = (old_lt_rewards - old_lt_rewards.mean()) / (old_lt_rewards.std() + 1e-8)
            old_advantages = old_lt_rewards - old_state_values
            # normalize 'old_advantages' for this buffer
            old_advantages = (old_advantages - old_advantages.mean()) / (old_advantages.std() + 1e-8)

            # Optimize policy for K epochs:
            for _ in range(k_epochs):
                shuffled_indexes = torch.randperm(buffer_size, device=device)
                # Perform update for each batch with 'batch_timestep' steps
                for batch_start_index in range(0, buffer_size, batch_timestep):
                    batch_end_index = min(batch_start_index + batch_timestep, buffer_size)
                    batch_indexes = shuffled_indexes[batch_start_index:batch_end_index]

                    # Evaluating old actions and values :
                    log_probs, state_values = rl_agent(old_states[batch_indexes])
                    if log_probs is None:
                        break
                    log_action_probs = log_probs.gather(1, old_actions[batch_indexes].view(-1, 1)).squeeze(1)
                    state_values = state_values.squeeze(1)
                    dist_entropy = Categorical(log_probs).entropy()

                    # Finding the ratio (pi_theta / pi_theta__old):
                    ratios = torch.exp(log_action_probs - old_log_probs[batch_indexes])

                    # Finding Surrogate Loss:
                    surr1 = ratios * old_advantages[batch_indexes]
                    surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * old_advantages[batch_indexes]
                    policy_loss = -torch.min(surr1, surr2)
                    entropy_loss = -dist_entropy
                    # Finding value loss:
                    state_value_clipped = old_state_values[batch_indexes] + torch.clamp(
                        state_values - old_state_values[batch_indexes], -eps_clip, eps_clip)
                    v_loss1 = F.smooth_l1_loss(state_values, old_lt_rewards[batch_indexes], reduction='none')
                    v_loss2 = F.smooth_l1_loss(state_value_clipped, old_lt_rewards[batch_indexes], reduction='none')
                    value_loss = torch.max(v_loss1, v_loss2)
                    # Total loss:
                    loss = policy_loss + 0.5 * value_loss + 0.001 * entropy_loss

                    # take gradient step
                    optimizer.zero_grad()
                    loss.mean().backward()
                    optimizer.step()
            lr_scheduler.step()
            # Copy new weights into old policy:
            rl_agent_old.load_state_dict(rl_agent.state_dict())

            if total_step >= max_total_step:
                break
            buffer_step = 0

    torch.save(rl_agent.state_dict(), "torch_models/" + env_name + ".ppo_model")
    plt.plot(list(range(1, len(episode_reward_list) + 1)), episode_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    print('Total time (sec): %f' % (time.time() - start, ))
