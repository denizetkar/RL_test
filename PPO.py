import time

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from rl_library import rl_agents, helper
torch.set_default_dtype(torch.float64)


def main():
    # MODEL CHANGES
    # TODO: Use 2 separate critics, 1 actor                     (3)
    # TODO: Use bagging ensemble for value function estimation  (3)
    # LOSS FUNCTION CHANGES
    # TODO: Use clipped value loss                              (2)
    # TODO: Add intrinsic reward, forward/inverse dynamic loss  (5)
    # TRAINING CHANGES
    # TODO: Train each critic (value loss) for 1 batch before   (4)
    #  training actor (policy loss + entropy loss) for 1 batch
    # TODO: Train forward/inverse loss with critics before      (5)
    #  training actor (policy loss + entropy loss) for 1 batch
    # TODO: Experiment with co-training actor and other losses  (6)
    #  for better sampling efficiency
    ############## Hyperparameters ##############
    env_name = 'CartPole-v1'
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = False
    max_batches = 100           # max training batches (each with at least 'batch_timestep' steps)
    episode_timesteps = 300     # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    batch_timestep = 2000       # update policy every n timesteps
    lr = 1e-3
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    gae_lambda = 0.9375         # lambda value for td(lambda) returns
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    env._max_episode_steps = episode_timesteps
    rl_agent = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, n_latent_var).to(device)
    rl_agent_old = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, n_latent_var).to(device)
    try:
        rl_agent.load_state_dict(torch.load("torch_models/" + env_name + ".ppo_model"))
    except FileNotFoundError:
        pass
    rl_agent_old.load_state_dict(rl_agent.state_dict())
    optimizer = torch.optim.Adam(rl_agent.parameters(),
                                 lr=lr, betas=betas)

    episode_reward_list = []
    transition_memory = helper.ReplayMemory(batch_timestep + episode_timesteps,
                                            ('state', 'action', 'lt_reward', 'log_prob', 'state_value'))
    batch_step = total_step = 0
    max_total_step = max_batches * batch_timestep
    # Start training loop
    while True:
        episodic_reward, state, episode_hist = 0.0, torch.as_tensor(env.reset(), device=device), []
        if render:
            env.render()
        episode_step = 0
        # Start episode loop
        while True:
            log_prob, state_value = rl_agent_old(state)
            log_prob, state_value = log_prob.detach(), state_value.detach()
            dist = Categorical(logits=log_prob)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.as_tensor(next_state, device=device)
            if render:
                env.render()
            episodic_reward += reward
            episode_hist.append((state, action, torch.tensor(reward, device=device),
                                 dist.log_prob(action).detach(), state_value.squeeze(0).detach()))
            episode_step, batch_step, total_step = episode_step + 1, batch_step + 1, total_step + 1
            if done:
                if episode_step >= episode_timesteps:
                    _, state_value = rl_agent_old(next_state)
                    episode_hist.append(
                        (None, None, torch.tensor(np.nan, device=device), None, state_value.squeeze(0).detach()))
                else:
                    episode_hist.append(
                        (None, None, torch.tensor(np.nan, device=device), None, torch.tensor(0.0, device=device)))
                break
            state = next_state
        episode_reward_list.append(episodic_reward)

        lt_rewards = helper.td_lambda_returns(
            torch.stack(helper.zip_ith_index(episode_hist, 2)),
            torch.stack(helper.zip_ith_index(episode_hist, 4)), gamma, gae_lambda)
        # Delete the last (s, a, r, log_prob, v) tuple
        episode_hist.pop()
        # Normalizing the long term rewards
        lt_rewards = (lt_rewards - lt_rewards.mean()) / (lt_rewards.std() + 1e-7)
        # DO NOT only use the first visit of each (s, a) tuple
        for i, (state, action, _, log_prob, state_value) in enumerate(episode_hist):
            transition_memory.push(state, action, lt_rewards[i], log_prob, state_value)

        if batch_step >= batch_timestep:
            batch_step = 0
            old_states, old_actions, old_lt_rewards, old_log_probs, old_state_values = zip(*transition_memory.memory)
            transition_memory.clear()
            # convert lists to tensors
            old_states = torch.stack(old_states)
            old_actions = torch.as_tensor(old_actions, device=device)
            old_lt_rewards = torch.as_tensor(old_lt_rewards, device=device)
            old_log_probs = torch.as_tensor(old_log_probs, device=device)
            old_state_values = torch.as_tensor(old_state_values, device=device)
            old_advantages = old_lt_rewards - old_state_values

            # Optimize policy for K epochs:
            for _ in range(k_epochs):
                # Evaluating old actions and values :
                log_probs, state_values = rl_agent(old_states)
                log_action_probs = log_probs.gather(1, old_actions.view(-1, 1)).squeeze(1)
                state_values = state_values.squeeze(1)
                dist_entropy = Categorical(log_probs).entropy()

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(log_action_probs - old_log_probs)

                # Finding Surrogate Loss:
                surr1 = ratios * old_advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * old_advantages
                loss = -torch.min(surr1, surr2) \
                    + 0.5 * F.smooth_l1_loss(state_values, old_lt_rewards) - 0.01 * dist_entropy

                # take gradient step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            # Copy new weights into old policy:
            rl_agent_old.load_state_dict(rl_agent.state_dict())

            if total_step >= max_total_step:
                break

    torch.save(rl_agent.state_dict(), "torch_models/" + env_name + ".ppo_model")
    plt.plot(list(range(1, len(episode_reward_list) + 1)), episode_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    print('Total time (sec): %f' % (time.time() - start, ))
