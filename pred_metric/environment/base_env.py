import os
import gym
import torch
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt

from typing import Callable
from tqdm import trange
from gym import spaces


class Environment(gym.Env):
    
    metadata = {'render.modes': ['human']}

    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray) -> None:
        super().__init__()
        self.env_name = env_name

        self.state_dim = state_dim
        self.control_dim = control_dim

        # These should be updated if there are more specific limits that exist for the specific environment.
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.control_dim, ))
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.state_dim, ))

        self.num_timesteps = num_timesteps
        self.dt = dt

        self.gt_theta = gt_theta
        self.theta_dim = gt_theta.shape[0]

    # OpenAI Gym Functions
    def step(self, action):
        raise NotImplementedError('Subclass and implement this function!')

    def reset(self):
        raise NotImplementedError('Subclass and implement this function!')

    def render(self, mode='human'):
        pass

    def close(self) -> None:
        pass
    
    # General Math Functions/Utilities
    def gt_reward_fn(self, x, u, theta, extra_info):
        raise NotImplementedError('Subclass and implement this function!')

    def dynamics_fn(self, x_t, u_t):
        raise NotImplementedError('Subclass and implement this function!')

    def hessian(self, x, u, theta, extra_info):
        raise NotImplementedError('Subclass and implement this function!')

    def gen_expert_data(self, num_samples, theta, test, plot_trajs):
        raise NotImplementedError('Subclass and implement this function!')

    def log_likelihood(self, x, u, extra_info, theta, mu_r, lam_r):
        raise NotImplementedError('Subclass and implement this function!')

    def plot_loss_landscape(self, fig_path, expert_x, expert_u, extra_info,
                            theta_lims, theta_r):
        raise NotImplementedError('Subclass and implement this function!')

    def plot_reward_landscape(self, fig_path, expert_x, expert_u, extra_info,
                              theta_lims):
        raise NotImplementedError('Subclass and implement this function!')
