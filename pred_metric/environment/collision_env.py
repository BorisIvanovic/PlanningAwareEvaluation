import os
from numpy.core.numeric import allclose
import torch
import torch.autograd.functional as AF
import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, Dict, Optional, Tuple
from tqdm import trange, tqdm
from pathlib import Path
from functools import partial
from pred_metric.environment import UnicycleEnvironment
from pred_metric.environment import utils

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from gym import spaces
from gym.wrappers import TimeLimit


class CollisionAvoidanceEnvironment(UnicycleEnvironment):
    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray,
                 init_state: Optional[np.ndarray] = None, avoid_pos: Optional[np.ndarray] = None) -> None:
        super().__init__(env_name, state_dim, control_dim, num_timesteps, dt, gt_theta)
        self.init_state = init_state
        self.avoid_pos = avoid_pos

        # x  y  cos(heading)  sin(heading)  speed x_obs y_obs
        bounds = np.array([np.inf, np.inf, 1., 1., np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-bounds, high=bounds)
        self.obs_dim = self.observation_space.shape[0]

    def dynamics_fn(self, x_t, u_t):
        # state x is [x, y, heading, along-track speed]
        # control u is [steering rate, along-track acceleration]
        next_x = self.nonlinear_dynamics_fn(x_t, u_t)

        # rel_to_obs = x_t[-2:] + (next_x[:2] - x_t[:2])
        rel_to_obs = next_x[..., :2] - self.obs_pos

        return np.concatenate([next_x, rel_to_obs], axis=-1)

    def calc_A(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        one = torch.tensor(1)
        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([6, 6]),
                        dtype=torch.float32)

        phi = x_t[..., 2]
        v = x_t[..., 3]
        dphi = u_t[..., 0]
        a = u_t[..., 1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = one
        F[..., 1, 1] = one
        F[..., 2, 2] = one
        F[..., 3, 3] = one

        F[..., 0, 2] = v * dcos_domega - (a / dphi) * dsin_domega + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        F[..., 0, 3] = dsin_domega

        F[..., 1, 2] = v * dsin_domega + (a / dphi) * dcos_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        F[..., 1, 3] = -dcos_domega

        F[..., 4, :] = F[..., 0, :]
        F[..., 5, :] = F[..., 1, :]

        # Using u_t for shape because it has the correct # of timesteps.
        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([6, 6]),
                           dtype=torch.float32)

        F_sm[..., 0, 0] = one
        F_sm[..., 1, 1] = one
        F_sm[..., 2, 2] = one
        F_sm[..., 3, 3] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        F_sm[..., 4, :] = F_sm[..., 0, :]
        F_sm[..., 5, :] = F_sm[..., 1, :]

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def calc_B(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([6, 2]),
                        dtype=torch.float32)

        phi = x_t[..., 2]
        v = x_t[..., 3]
        dphi = u_t[..., 0]
        a = u_t[..., 1]

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        F[..., 0, 0] = ((v / dphi) * torch.cos(phi_p_omega_dt) * self.dt
                        - (v / dphi) * dsin_domega
                        - (2 * a / dphi ** 2) * torch.sin(phi_p_omega_dt) * self.dt
                        - (2 * a / dphi ** 2) * dcos_domega
                        + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt ** 2)
        F[..., 0, 1] = (1 / dphi) * dcos_domega + (1 / dphi) * torch.sin(phi_p_omega_dt) * self.dt

        F[..., 1, 0] = ((v / dphi) * dcos_domega
                        - (2 * a / dphi ** 2) * dsin_domega
                        + (2 * a / dphi ** 2) * torch.cos(phi_p_omega_dt) * self.dt
                        + (v / dphi) * torch.sin(phi_p_omega_dt) * self.dt
                        + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt ** 2)
        F[..., 1, 1] = (1 / dphi) * dsin_domega - (1 / dphi) * torch.cos(phi_p_omega_dt) * self.dt

        F[..., 2, 0] = self.dt

        F[..., 3, 1] = self.dt

        F[..., 4, :] = F[..., 0, :]
        F[..., 5, :] = F[..., 1, :]

        # Using u_t for shape because it has the correct # of timesteps.
        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([6, 2]),
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2

        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        F_sm[..., 4, :] = F_sm[..., 0, :]
        F_sm[..., 5, :] = F_sm[..., 1, :]

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def _make_observation(self, state):
        #                   x         y        cos(heading)      sin(heading)     speed    distance to obstacle (x and y)
        return np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3], state[4], state[5]])

    def step_reward_fn(self, x_t, u_t, extra_info=None):
        return -(self.gt_theta[0] * np.square(x_t[:2]).sum() + self.gt_theta[1] * np.square(u_t).sum()
                 + self.gt_theta[2] * np.exp(-0.5*np.square(x_t[-2:]).sum()/0.003))

    def gt_reward_fn(self, x, u, theta, extra_info=None):
        x_reshaped, u_reshaped = self.unvec_xu(x, u, extra_info)

        # The mean is across examples in the batch
        return -torch.mean(theta[0] * torch.square(x_reshaped[..., :2]).sum((-1, -2)) + theta[1] * torch.square(u_reshaped).sum((-1, -2))
                            + theta[2] * torch.exp(-0.5*torch.square(x_reshaped[..., -2:]).sum(-1)/0.003).sum(-1))
    
    def reset(self):
        if self.avoid_pos is None:
            init_r = np.random.uniform(low=0.4, high=0.5)
            init_theta = np.random.uniform(low=0.0, high=np.pi/2)

            self.obs_pos = np.array([init_r * np.cos(init_theta), init_r * np.sin(init_theta)])
        else:
            self.obs_pos = self.avoid_pos

        self.gym_state = np.zeros((self.state_dim, ))
        if self.init_state is None:
            init_r = np.random.uniform(low=0.6, high=1.0)
            init_theta = np.random.uniform(low=init_theta-np.pi/12, high=init_theta+np.pi/12)

            self.gym_state[0] = init_r * np.cos(init_theta)
            self.gym_state[1] = init_r * np.sin(init_theta)
            self.gym_state[2] = np.random.uniform(low=0.0, high=2*np.pi)
            self.gym_state[3] = 0.0  # Starting from rest.
        else:
            self.gym_state[:4] = self.init_state

        self.gym_state[4:] = self.gym_state[:2] - self.obs_pos

        self.curr_timestep = 0
        return self._make_observation(self.gym_state)

    def to_dynstate(self, x):
        if x.shape[-1] == self.state_dim:
            return x
        elif x.shape[-1] == self.obs_dim:
            pos_x = x[..., 0]
            pos_y = x[..., 1]
            heading = np.arctan2(x[..., 3], x[..., 2])
            speed = x[..., 4]
            x_from_obs = x[..., 5]
            y_from_obs = x[..., 6]
            return np.stack([pos_x, pos_y, heading, speed, x_from_obs, y_from_obs], axis=-1)
