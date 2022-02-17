import os
os.environ['GYM_CONFIG_CLASS'] = 'CollisionConfig'

# If your new config class is not in config.py, set this:
os.environ['GYM_CONFIG_PATH'] = 'pred_metric/environment/gca/config.py'

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from pred_metric.environment import GCAEnvironment, utils
from pred_metric.cioc import CIOC
from pred_metric.visualization import *

np.random.seed(123)


def main():
    env = GCAEnvironment(env_name='collision_avoidance',
                         state_dim=7, 
                         control_dim=2, 
                         num_timesteps=50, 
                         dt=0.1, 
                         gt_theta=np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32))

    expert_x, expert_u, extra_info = env.gen_expert_data(num_samples=64)

    # ioc_alg = CIOC(env)
    # learned_theta, theta_r, training_info = ioc_alg.fit(expert_x, expert_u, extra_info=extra_info)
    
    # print('Learned Theta:', learned_theta)
    # print('Slack Variable:', theta_r)

    # plot_training_loss_curve(training_info)
    # plot_training_slack_vars(training_info)
    # plot_training_thetas_nd(training_info, env.gt_theta)
    # plt.show()

    learned_theta = torch.tensor([1.214, 4.188, 0.366, 0.352]) # Mobile (noncoop) obstacles, but with one-step predictions

    plot_predictions_and_sensitivities(env, learned_theta, expert_x, extra_info, 3, 8)
    plot_predictions_and_sensitivities(env, learned_theta, expert_x, extra_info, 47, 8)


if __name__ == '__main__':
    main()
