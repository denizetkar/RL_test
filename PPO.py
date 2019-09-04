import gym
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from rl_library import rl_agents, helper
from torch.distributions import Categorical
torch.set_default_dtype(torch.float64)


def main():
    ############## Hyperparameters ##############
    env_name = 'CartPole-v1'
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    render = False
    max_episodes = 2000         # max training episodes
    max_timesteps = 300         # max timesteps in one episode
    n_latent_var = 64           # number of variables in hidden layer
    update_timestep = 2000      # update policy every n timesteps
    lr = 0.002
    betas = (0.9, 0.999)
    gamma = 0.99                # discount factor
    k_epochs = 4                # update policy for K epochs
    eps_clip = 0.2              # clip parameter for PPO
    random_seed = None
    #############################################

    if random_seed is not None:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    env._max_episode_steps = max_timesteps
    rl_agent = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, n_latent_var)
    rl_agent_old = rl_agents.PPOContStateDiscActionAgentTorch(
        state_dim, action_dim, n_latent_var)
    try:
        rl_agent.load_state_dict(torch.load("torch_models/" + env_name + ".ppo_model"))
    except FileNotFoundError:
        pass
    rl_agent_old.load_state_dict(rl_agent.state_dict())
    optimizer = torch.optim.Adam(rl_agent.parameters(),
                                 lr=lr, betas=betas)

    episode_reward_list = []
    transition_memory = helper.ReplayMemory(update_timestep + max_timesteps,
                                            ('state', 'action', 'lt_reward', 'log_prob'))
    total_step = 0
    for episode in range(1, max_episodes + 1):
        episodic_reward, state, episode_hist = 0.0, env.reset(), []
        if render:
            env.render()
        episode_step = 0
        while True:
            log_prob, state_value = rl_agent_old(torch.as_tensor(state))
            dist = Categorical(logits=log_prob)
            action = dist.sample()
            next_state, reward, done, _ = env.step(action.item())
            if render:
                env.render()
            episodic_reward += reward
            episode_hist.append((state, action, reward, dist.log_prob(action)))
            episode_step += 1
            total_step += 1
            if done:
                if episode_step >= max_timesteps:
                    _, state_value = rl_agent_old(torch.as_tensor(next_state))
                    episode_hist.append((None, None, state_value.item(), None))
                else:
                    episode_hist.append((None, None, 0, None))
                break
            state = next_state
        episode_reward_list.append(episodic_reward)

        lt_rewards = helper.discount_rewards(list(zip(*episode_hist))[2], gamma)
        # Delete the last (s, a, r, log_prob) tuple
        episode_hist.pop()
        lt_rewards = np.delete(lt_rewards, lt_rewards.size - 1)
        # Normalizing the long term rewards
        lt_rewards = torch.as_tensor(lt_rewards)
        lt_rewards = (lt_rewards - lt_rewards.mean()) / (lt_rewards.std() + 1e-5)
        # DO NOT only use the first visit of each (s, a) tuple
        for i, (state, action, _, log_prob) in enumerate(episode_hist):
            transition_memory.push(state, action, lt_rewards[i], log_prob.detach())

        if total_step >= update_timestep or episode == max_episodes:
            total_step = 0
            old_states, old_actions, old_lt_rewards, old_log_probs = zip(*transition_memory.memory)
            transition_memory.clear()
            # convert lists to tensors
            old_states = torch.as_tensor(old_states)
            old_actions = torch.as_tensor(old_actions)
            old_lt_rewards = torch.as_tensor(old_lt_rewards)
            old_log_probs = torch.stack(old_log_probs)

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
                advantages = old_lt_rewards - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages
                loss = -torch.min(surr1, surr2) \
                    + 0.5 * F.smooth_l1_loss(state_values, old_lt_rewards) - 0.01 * dist_entropy

                # take gradient step
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()

            # Copy new weights into old policy:
            rl_agent_old.load_state_dict(rl_agent.state_dict())

    torch.save(rl_agent.state_dict(), "torch_models/" + env_name + ".ppo_model")
    plt.plot(list(range(1, len(episode_reward_list) + 1)), episode_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.show()


if __name__ == '__main__':
    main()
