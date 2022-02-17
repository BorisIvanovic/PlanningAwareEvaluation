import os
from numpy.random import rand
import torch
import torch.autograd.functional as AF
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from tqdm import trange, tqdm
from typing import Dict, Optional
from pred_metric.environment import Environment, index_dict


def qual_compare_trajectories(env: Environment,
                              learned_theta: np.ndarray, 
                              num_random: int = 2,
                              num_tests=2,
                              random_seed=1):
    np.random.seed(random_seed)
    gt_expert_x, gt_expert_u, gt_extra_info = env.gen_expert_data(num_tests, env.gt_theta, test=True, seed=random_seed, model_name='gt')

    np.random.seed(random_seed)
    learned_expert_x, learned_expert_u, learned_extra_info = env.gen_expert_data(num_tests, learned_theta, test=True, seed=random_seed, model_name='learned')

    rand_data = list()
    for rand_num in range(num_random):
        np.random.seed()
        random_theta = np.random.randn(env.theta_dim)
        np.random.seed(random_seed)
        try:
            rand_data.append(env.gen_expert_data(num_tests, random_theta, test=True, seed=random_seed, model_name='random'))
        except cp.DCPError as e:
            continue

    n = env.state_dim
    m = env.control_dim
    gt_theta_pt = torch.tensor(env.gt_theta)

    gt_expert_x, gt_expert_u = gt_expert_x.reshape(num_tests, -1, n), gt_expert_u.reshape(num_tests, -1, m)
    learned_expert_x, learned_expert_u = learned_expert_x.reshape(num_tests, -1, n), learned_expert_u.reshape(num_tests, -1, m)

    for i in range(num_tests):
        fig, ax = plt.subplots()

        gt_T = env.num_timesteps
        lrn_T = env.num_timesteps
        if gt_extra_info is not None and 'ep_lengths' in gt_extra_info:
            gt_T = gt_extra_info['ep_lengths'][i]
            lrn_T = learned_extra_info['ep_lengths'][i]

        ax.scatter([0], [0], c='blue', label='Goal Pos')
        if gt_extra_info is not None and 'intermediate_poses' in gt_extra_info:
            intermediate_pos = index_dict(gt_extra_info, i)['intermediate_poses']
            ax.scatter([intermediate_pos[0]], [intermediate_pos[1]], c='red', label='Intermediate Pos')

        for num, (rand_x, rand_u, rand_extra_info) in enumerate(rand_data):
            rnd_T = env.num_timesteps
            if rand_extra_info is not None and 'ep_lengths' in rand_extra_info:
                rnd_T = rand_extra_info['ep_lengths'][i]

            random_expert_x = rand_x.reshape(num_tests, -1, n)
            ax.plot(random_expert_x[i, :rnd_T+1, 0], random_expert_x[i, :rnd_T+1, 1], c='k', label='Random' if num == 0 else None)
            print(f'Random {num} cost {-env.gt_reward_fn(rand_x[i], rand_u[i], gt_theta_pt, index_dict(rand_extra_info, i))}')

        ax.plot(learned_expert_x[i, :lrn_T+1, 0], learned_expert_x[i, :lrn_T+1, 1], label='Learned')
        print(f'Learned cost {-env.gt_reward_fn(learned_expert_x[i].flatten(), learned_expert_u[i].flatten(), gt_theta_pt, index_dict(learned_extra_info, i))}')

        ax.plot(gt_expert_x[i, :gt_T+1, 0], gt_expert_x[i, :gt_T+1, 1], label='GT')
        print(f'GT cost {-env.gt_reward_fn(gt_expert_x[i].flatten(), gt_expert_u[i].flatten(), gt_theta_pt, index_dict(gt_extra_info, i))}')

        ax.legend(loc='best')
        plt.show()

        fig, axes = plt.subplots(nrows=2, sharex=True)
        for num, (rand_x, rand_u, rand_extra_info) in enumerate(rand_data):
            random_expert_u = rand_u.reshape(num_tests, -1, m)
            rnd_T = env.num_timesteps
            if rand_extra_info is not None and 'ep_lengths' in rand_extra_info:
                rnd_T = rand_extra_info['ep_lengths'][i]

            axes[0].plot(random_expert_u[i, :rnd_T, 0], c='k', label='Random' if num == 0 else None)
        axes[0].plot(learned_expert_u[i, :lrn_T, 0], label='Learned')
        axes[0].plot(gt_expert_u[i, :gt_T, 0], label='GT')
        axes[0].legend(loc='best')

        for num, (rand_x, rand_u, rand_extra_info) in enumerate(rand_data):
            random_expert_u = rand_u.reshape(num_tests, -1, m)
            rnd_T = env.num_timesteps
            if rand_extra_info is not None and 'ep_lengths' in rand_extra_info:
                rnd_T = rand_extra_info['ep_lengths'][i]

            axes[1].plot(random_expert_u[i, :rnd_T, 1], c='k', label='Random' if num == 0 else None)
        axes[1].plot(learned_expert_u[i, :lrn_T, 1], label='Learned')
        axes[1].plot(gt_expert_u[i, :gt_T, 1], label='GT')
        axes[1].legend(loc='best')
        plt.show()
