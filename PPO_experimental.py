import math
import os
import pickle
import random
import time

import gym
import hyperopt
import matplotlib.pyplot as plt
import numpy as np
import torch
from hyperopt import hp
from hyperopt.fmin import generate_trials_to_calculate
from torch.distributions import Categorical

from rl_library import rl_agents, helper
torch.set_default_dtype(torch.float32)


def ppo_exp_evaluation(
        env, state_dim, action_dim, render, max_total_step, episode_timesteps, hidden_layers, buffer_timestep,
        buffer_to_batch_ratio, lr, gamma, gae_lambda, k_epochs, eps_clip, training_alternation, lr_decay_order,
        random_seed=None, device=torch.device('cpu'), initial_model=None):
    batch_timestep = max(buffer_timestep // buffer_to_batch_ratio, 1)
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    env._max_episode_steps = episode_timesteps
    rl_agent = rl_agents.PPOExpContStateDiscActionAgentTorch(state_dim, action_dim, hidden_layers).to(device)
    rl_agent_old = rl_agents.PPOExpContStateDiscActionAgentTorch(state_dim, action_dim, hidden_layers).to(device)
    if initial_model is not None:
        rl_agent.load_state_dict(initial_model.state_dict())
    rl_agent_old.load_state_dict(rl_agent.state_dict())
    rl_agent_old.eval()
    optimizer = torch.optim.Adam(rl_agent.parameters(), lr=lr)
    t_max = (max_total_step - 1) // buffer_timestep + 1
    should_train_critic = True
    if training_alternation:
        t_max = (t_max - 1) // 2 + 1
        should_train_actor = False
    else:
        should_train_actor = True
    lr_scheduler = helper.CosineLogAnnealingLR(optimizer, t_max, eta_min=0.0, log_order=lr_decay_order)

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
            log_prob, state_value = rl_agent_old(state.unsqueeze(0), True), rl_agent_old(state.unsqueeze(0), False)
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
            if done or buffer_step >= buffer_timestep:
                if episode_step >= episode_timesteps or buffer_step >= buffer_timestep:
                    state_value = rl_agent_old(next_state.unsqueeze(0), False)
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

            if total_step >= max_total_step:
                should_train_actor = should_train_critic = True
            # Optimize policy for K epochs:
            for _ in range(k_epochs):
                shuffled_indexes = torch.randperm(buffer_size, device=device)
                # Perform update for each batch with 'batch_timestep' steps
                for batch_start_index in range(0, buffer_size, batch_timestep):
                    batch_end_index = min(batch_start_index + batch_timestep, buffer_size)
                    batch_indexes = shuffled_indexes[batch_start_index:batch_end_index]
                    loss = torch.tensor(0.0, device=device)
                    if should_train_actor:
                        policy_loss, entropy_loss = rl_agent.actor_losses(old_states[batch_indexes],
                                                                          old_actions[batch_indexes],
                                                                          old_log_probs[batch_indexes],
                                                                          old_advantages[batch_indexes], eps_clip)
                        if policy_loss is None:
                            break
                        loss += policy_loss.mean() + 0.001 * entropy_loss.mean()
                    if should_train_critic:
                        value_loss = rl_agent.critic_losses(old_states[batch_indexes], old_lt_rewards[batch_indexes],
                                                            old_state_values[batch_indexes], eps_clip)
                        if value_loss is None:
                            break
                        loss += (1.0 - 0.5 * should_train_actor) * value_loss.mean()

                    # take gradient step
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # Copy new weights into old policy:
            rl_agent_old.load_state_dict(rl_agent.state_dict())

            if total_step >= max_total_step:
                break
            buffer_step = 0
            should_train_actor, should_train_critic = should_train_critic, should_train_actor
            if training_alternation:
                if should_train_critic:
                    lr_scheduler.step()
            else:
                lr_scheduler.step()

    return np.mean(episode_reward_list[-((max_total_step - 1) // episode_timesteps + 1):]), rl_agent.to(
        'cpu'), episode_reward_list


def main():
    # LOSS FUNCTION CHANGES
    # TODO: Add intrinsic reward, forward/inverse dynamic loss  (5)
    # TRAINING CHANGES
    # TODO: Train forward/inverse loss with critics before      (5)
    #  training actor (policy loss + entropy loss) for 1 batch
    # TODO: Experiment with co-training actor and other losses  (6)
    #  for better sampling efficiency
    ############## Hyperparameters ##############
    # creating environment
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    hidden_layers = [(64, 0.0, False), (64, 0.0, False)]
    #############################################

    model_path = os.path.join('torch_models', 'best_checkpoint.ppo_exp')
    trials_path = os.path.join('torch_models', 'last_trials.ppo_exp')

    random_seed = 11
    prior_params = dict(
        env=env,
        state_dim=state_dim,
        action_dim=action_dim,
        render=False,
        max_total_step=100000,      # min number of steps to take during training
        episode_timesteps=1000,     # max time steps in one episode
        hidden_layers=hidden_layers,  # list of (hidden_layer_size, dropout_rate, use_batch_layer)
        gamma=0.99,                 # discount factor
        gae_lambda=0.95,            # lambda value for td(lambda) returns
        eps_clip=0.2,               # clip parameter for PPO
        training_alternation=True,  # whether to train actor and critic together or alternatively (policy iteration)
        random_seed=random_seed,
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device=torch.device('cpu')
    )

    integer_param_names = ['buffer_timestep', 'buffer_to_batch_ratio', 'k_epochs']
    indexed_param_values = dict(buffer_to_batch_ratio=[2, 4, 5, 10])
    model_evaluator = helper.ModelEvaluator(ppo_exp_evaluation, prior_params, integer_param_names,
                                            indexed_param_values, invert_loss=True)
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
            model_evaluator.load_state_dict(checkpoint)
    # For quantized uniform parameters: low <- (actual_low - q/2) AND high <- (actual_high + q/2)
    btbr_len = len(indexed_param_values['buffer_to_batch_ratio'])
    space = {
        'buffer_timestep': hp.quniform('buffer_timestep', 750, 10250, 500),                      # (1000, 10000)
        'buffer_to_batch_ratio': hp.quniform('buffer_to_batch_ratio', -0.5, btbr_len - 0.5, 1),  # (0, btbr_len - 1)
        'lr': hp.loguniform('lr', math.log(1e-4), math.log(1e-1)),                               # (1e-4, 1e-1)
        'k_epochs': hp.quniform('k_epochs', 0.5, 10.5, 1),                                       # (1, 10)
        'lr_decay_order': hp.quniform('lr_decay_order', -2.5, 10.5, 1)                           # (-2, 10)
    }
    if model_evaluator.best_params is None:
        points_to_evaluate = [
            {
                # default parameters to evaluate
                'buffer_timestep': 2000,     # min time steps in a training buffer
                'buffer_to_batch_ratio': 0,  # ratio of time steps in a single update batch (index)
                'lr': 5e-3,
                'k_epochs': 4,               # update policy for K epochs
                'lr_decay_order': 0
            },
            {
                # best loss: -522.31
                'buffer_timestep': 2000,
                'buffer_to_batch_ratio': 2,
                'k_epochs': 3,
                'lr': 0.002313285464437867,
                'lr_decay_order': 1
            },
            {
                # best loss: -680.0
                'buffer_timestep': 1000,
                'buffer_to_batch_ratio': 2,
                'k_epochs': 7,
                'lr': 0.010826087507883215,
                'lr_decay_order': 7
            }]
    else:
        points_to_evaluate = [{name: value for name, value in model_evaluator.best_params.items()
                               if name not in prior_params}]
    trials = generate_trials_to_calculate(points_to_evaluate)
    best_params = hyperopt.fmin(model_evaluator, space, algo=hyperopt.atpe.suggest, max_evals=100, trials=trials)
    print('Best parameters: %s' % best_params)

    # Save trials for diagnosis
    with open(trials_path, 'wb') as f:
        pickle.dump(trials.results, f)
    # Save state of the model evaluator
    with open(model_path, 'wb') as f:
        pickle.dump(model_evaluator.state_dict(), f)
    episode_reward_list = model_evaluator.best_other_metrics
    plt.plot(list(range(1, len(episode_reward_list) + 1)), episode_reward_list)
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.show()


if __name__ == '__main__':
    start = time.time()
    main()
    print('Total time (sec): %f' % (time.time() - start, ))
