import os
os.environ['GYM_CONFIG_CLASS'] = 'CollisionConfig'

# If your new config class is not in config.py, set this:
os.environ['GYM_CONFIG_PATH'] = '../pred_metric/environment/gca/config.py'

import numpy as np
from tqdm import trange

from pred_metric.environment import GCAEnvironment, utils
from pred_metric.visualization import *


def test_gca_similarity():
    env = GCAEnvironment(env_name='collision_avoidance',
                         state_dim=5, 
                         control_dim=2, 
                         num_timesteps=50, 
                         dt=0.1, 
                         gt_theta=np.array([1.0, 1.0, 1.0], dtype=np.float32))

    expert_x, expert_u, extra_info = env.gen_expert_data(num_samples=64)

    for ep in range(32):
        x, u = expert_x[ep], expert_u[ep]

        x_proper, u_proper = env._ensure_length(x, u, utils.index_dict(extra_info, ep))
        x_reshaped, u_reshaped = x_proper.reshape(-1, env.state_dim), u_proper.reshape(-1, env.control_dim)

        for t in range(extra_info['ep_lengths'][ep]-1):
            pred_next = env.dynamics_fn(x_reshaped[t], u_reshaped[t])
            gca_next = x_reshaped[t+1]

            assert np.allclose(pred_next, gca_next)
