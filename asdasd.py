import tensorflow as tf
import numpy as np
import gym
from rl_library import rl_envs, plotting
import torch

if False:
    breakpoint;
    x = torch.tensor(0.0, requires_grad=True)
    a = torch.tensor(1.0)
    b = torch.tensor(-2.0)
    c = torch.tensor(1.0)
    LR = 0.01
    for iter in range(500):
        y = a*x.pow(2)+b*x+c
        y.backward()
        x.data.sub_(x.grad.data * LR)
        x.grad.data *= 0.0

if False:
    breakpoint;
    game_name = 'Taxi-v2'
    env = gym.make(game_name)
    Q_old_table = np.loadtxt("tabular_models/" + game_name + ".csv", delimiter=';')
    s = env.reset()
    env.render()
    while True:
        a = np.argmax(Q_old_table[s, :])
        s1, r, d, _ = env.step(a)
        env.render()
        if d is True:
            break
        s = s1

if False:
    breakpoint;
    game_name = 'EasyBlackJack-v0'
    Q_old_table = np.loadtxt(game_name + ".csv", delimiter=';')
    V = {}
    for i in range(Q_old_table.shape[0]):
        sum_hand, dealer, usable_ace = rl_envs.EasyBlackJackEnv.decode(i)
        if 11 <= sum_hand <= 21 and 1 <= dealer <= 11:
            V[(sum_hand, dealer, usable_ace)] = np.max(Q_old_table[i, :])
    plotting.plot_value_function(V, "Optimal Value Function")

if True:
    breakpoint;
    x = torch.zeros(1, requires_grad=True)
    x.data.fill_(2.0)
    y = x ** 2
    z = x ** 3
    a = y.detach() - z
    a.backward(retain_graph=True)
    print(x.grad)
