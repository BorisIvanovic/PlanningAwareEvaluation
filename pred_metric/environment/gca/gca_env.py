import os
import torch
import tensorflow as tf
import torch.autograd.functional as AF
import numpy as np

from typing import List, Dict, Optional, Tuple, Union
from tqdm import tqdm
from pathlib import Path
from functools import partial
from pred_metric.environment import UnicycleEnvironment
from pred_metric.environment import utils

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

import gym
from gym import spaces
from gym.wrappers import TimeLimit

from gym_collision_avoidance.envs import Config, test_cases as tc
from gym_collision_avoidance.envs.agent import Agent
from gym_collision_avoidance.envs.dynamics.UnicycleDynamics import UnicycleDynamics
from gym_collision_avoidance.envs.sensors.OtherAgentsStatesSensor import OtherAgentsStatesSensor
from gym_collision_avoidance.envs.collision_avoidance_env import CollisionAvoidanceEnv
from gym_collision_avoidance.envs.util import wrap

class GCAEnvironment(UnicycleEnvironment):
    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray) -> None:
        super().__init__(env_name, state_dim, control_dim, num_timesteps, dt, gt_theta)
        # x  y  cos(heading)  sin(heading)   x_obs y_obs
        bounds = np.array([np.inf, np.inf, 1., 1., np.inf, np.inf, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-bounds, high=bounds)
        self.obs_dim = self.observation_space.shape[0]

        # Create single tf session for all experiments
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        tf.Session().__enter__()

        # Instantiate the environment
        env = gym.make("CollisionAvoidance-v0")
        self.gca_env: CollisionAvoidanceEnv = Monitor(TimeLimit(env, max_episode_steps=self.num_timesteps))

        # In case you want to save plots, choose the directory
        self.gca_env.set_plot_save_dir(f'logs/{env_name}/')

        # So that the first reset makes it zero
        self.gca_env.unwrapped.test_case_index = -1

    def dynamics_fn(self, x_t, u_t):
        # state x is [x, y, heading]
        # control u is [along-track speed, steering rate]
        return self.nonlinear_dynamics_fn(x_t, u_t)

    def nonlinear_dynamics_fn(self, x, u):
        x_t = x.reshape(-1, self.state_dim)
        u_t = u.reshape(-1, self.control_dim)

        x_p = x_t[..., 0]
        y_p = x_t[..., 1]
        phi = x_t[..., 2]
        x_to_obs = x_t[..., 3]
        y_to_obs = x_t[..., 4]
        v = u_t[..., 0]
        dphi = u_t[..., 1]

        phi_p_omega = wrap(phi + dphi)

        d1 = np.stack([x_p + v * np.cos(phi_p_omega) * self.dt,
                       y_p + v * np.sin(phi_p_omega) * self.dt,
                       phi_p_omega,
                       x_to_obs + v * np.cos(phi_p_omega) * self.dt,
                       y_to_obs + v * np.sin(phi_p_omega) * self.dt],
                      axis=-1)
        return d1

    def calc_A(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        one = torch.tensor(1)
        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.state_dim]),
                        dtype=torch.float32)

        phi = x_t[..., 2]
        v = u_t[..., 0]
        dphi = u_t[..., 1]

        phi_p_omega = phi + dphi

        F[..., 0, 0] = one
        F[..., 1, 1] = one
        F[..., 2, 2] = one
        F[..., 3, 3] = one
        F[..., 4, 4] = one
        F[..., 5, 5] = one
        F[..., 6, 6] = one

        F[..., 0, 2] = -v * torch.sin(phi_p_omega) * self.dt

        F[..., 1, 2] = v * torch.cos(phi_p_omega) * self.dt

        F[..., 3, 2] = F[..., 0, 2]
        F[..., 4, 2] = F[..., 1, 2]

        F[..., 5, 2] = F[..., 0, 2]
        F[..., 6, 2] = F[..., 1, 2]

        return F

    def calc_B(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        one = torch.tensor(1)
        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.control_dim]),
                        dtype=torch.float32)

        phi = x_t[..., 2]
        v = u_t[..., 0]
        dphi = u_t[..., 1]

        phi_p_omega = phi + dphi
        cos_term = torch.cos(phi_p_omega) * self.dt
        sin_term = torch.sin(phi_p_omega) * self.dt

        F[..., 0, 0] = cos_term
        F[..., 0, 1] = -v * sin_term

        F[..., 1, 0] = sin_term
        F[..., 1, 1] = v * cos_term

        F[..., 2, 1] = one

        F[..., 3, :] = F[..., 0, :]
        F[..., 4, :] = F[..., 1, :]

        F[..., 5, :] = F[..., 0, :]
        F[..., 6, :] = F[..., 1, :]

        return F

    def _make_observation(self, state):
        #                   x         y        cos(heading)      sin(heading)     distance to obstacle (x and y)  distance to predicted obstacle (x and y)
        return np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3], state[4], state[5], state[6]])

    def step_reward_fn(self, x_t, u_t, extra_info=None):
        return -(self.gt_theta[0] * np.square(x_t[:2]).sum() + self.gt_theta[1] * np.square(u_t).sum()
                 + self.gt_theta[2] * np.exp(-0.5*np.square(x_t[-2:]).sum()/0.05))

    def gt_reward_fn(self, x, u, theta, extra_info=None):
        x_reshaped, u_reshaped = self.unvec_xu(x, u, extra_info)

        # This form works and gives values like: [2.508, 4.607, 1.097], slack variable: 1e-07
        reward_terms = torch.stack([-torch.square(x_reshaped[..., :2]).sum((-1, -2)),
                                    -torch.square(u_reshaped).sum((-1, -2)),
                                    -utils.rbf(x_reshaped[..., -4:-2], scale=0.05).sum(-1),
                                    -utils.rbf(x_reshaped[..., -2:], scale=0.05).sum(-1)],
                                   dim=-1)

        # Only penalizing velocity, gives less-good values: [0.775, 2.188, 0.336], slack variable: 0.18
        # reward_terms = torch.stack([-torch.square(x_reshaped[..., :2]).sum((-1, -2)),
        #                             -torch.square(u_reshaped[..., [0]]).sum((-1, -2)),
        #                             -utils.rbf(x_reshaped[..., -2:], scale=0.05).sum(-1)],
        #                            dim=-1)

        # Only penalizing heading delta doesn't converge.
        # reward_terms = torch.stack([-torch.square(x_reshaped[..., :2]).sum((-1, -2)),
        #                             -torch.square(u_reshaped[..., [1]]).sum((-1, -2)),
        #                             -utils.rbf(x_reshaped[..., -2:], scale=0.05).sum(-1)],
        #                            dim=-1)

        # No control cost doesn't converge (gets [0.345, 0.085] @ slack 0.25 after 20 iterations), but
        # has some chance at the best NLL, maybe that's worth keeping around?
        # This one is hard to optimize... might not be worth it.
        # reward_terms = torch.stack([-torch.square(x_reshaped[..., :2]).sum((-1, -2)),
        #                             -utils.rbf(x_reshaped[..., -2:], scale=0.05).sum(-1)],
        #                            dim=-1)

        # GA3C-CADRL reward function, not fantastic either.
        # dist_to_obs = torch.linalg.norm(x_reshaped[..., -2:], dim=-1)
        # goal_reward = torch.where(torch.linalg.norm(x_reshaped[..., :2], dim=-1) <= 0.15, 1.0, 0.0).sum(-1)
        # collision_reward = torch.where(dist_to_obs <= 0.1, -0.25, 0.0).sum(-1)
        # near_reward = torch.where(torch.logical_and(0.1 < dist_to_obs, dist_to_obs <= 0.2), 
        #                           -0.1 + 0.05 * dist_to_obs/2, 
        #                           torch.tensor(0.0)).sum(-1)
        # reward_terms = torch.stack([goal_reward, collision_reward, near_reward], dim=-1)
        
        # combined_reward = (reward_terms @ torch.exp(theta))
        combined_reward = (reward_terms @ theta)
        
        # The mean is over the batch
        return torch.mean(combined_reward, dim=-1)

    def get_sensitivities(self, theta: torch.Tensor, 
                          ego_x: torch.Tensor, 
                          extra_info: Dict[str, torch.Tensor]):
        g_hat, _ = AF.jacobian(partial(self.gt_reward_fn, 
                                        theta=theta, 
                                        extra_info=extra_info), 
                                (ego_x, torch.zeros((ego_x.shape[0], self.control_dim))))

        g_hat_reshaped = g_hat.reshape((-1, self.state_dim))
        coord_sensitivies = g_hat_reshaped[:extra_info['ep_lengths'], -2:]

        sensitivity_mags = torch.linalg.norm(coord_sensitivies, dim=-1)
        return sensitivity_mags

    def get_closest_obs_delta_xy(self):
        ego_pos = self.gym_state[:2]
        distances = np.linalg.norm(ego_pos - self.obs_poses, axis=-1)
        min_idx = np.argmin(distances)
        return self.gym_state[:2] - self.obs_poses[min_idx]

    def get_closest_pred_obs_delta_xy(self, ego_vel):
        ego_pos = self.gym_state[:2]
        pred_obs_poses = self.obs_poses + self.dt * self.obs_vels
        pred_ego_pose = ego_pos + self.dt * ego_vel
        distances = np.linalg.norm(pred_ego_pose - pred_obs_poses, axis=-1)
        min_idx = np.argmin(distances)
        return pred_ego_pose - pred_obs_poses[min_idx]

    def reset(self, agent_types: List[str] = ['GA3C_CADRL', 'noncoop'], raw_obs: bool = False):
        # This reset call is just for saving plots, 
        # we'll call it again later with the new agents.
        self.gca_env.reset()
        self.gca_env.unwrapped.test_case_index += 1

        init_r = np.random.uniform(low=0.6, high=np.sqrt(2))
        init_theta = np.random.uniform(low=0.0, high=np.pi/2)
        init_pos = [init_r * np.cos(init_theta), init_r * np.sin(init_theta)]
        init_heading = np.random.uniform(low=0.0, high=2*np.pi)

        agents: List[Agent] = [
            Agent(init_pos[0], init_pos[1], 0.0, 0.0, 0.1, 0.5, init_heading, tc.policy_dict[agent_types[0]], UnicycleDynamics, [OtherAgentsStatesSensor], 0)
        ]

        for i, agent_type in enumerate(agent_types[1:]):
            # Set agent configuration (start/goal pos, radius, size, policy)
            init_obs_r = np.random.uniform(low=0.3, high=0.4)
            init_obs_theta = np.random.uniform(low=0.0, high=np.pi/2)
            init_obs_pos = [init_obs_r * np.cos(init_obs_theta), init_obs_r * np.sin(init_obs_theta)]

            init_obs_heading = np.random.uniform(low=0.0, high=2*np.pi)

            if agent_type == 'static':
                final_obs_pos = init_obs_pos
            else:
                final_obs_r = np.random.uniform(low=0.8, high=np.sqrt(2))
                final_obs_theta = np.random.uniform(low=0.0, high=np.pi/2)
                final_obs_pos = [final_obs_r * np.cos(final_obs_theta), final_obs_r * np.sin(final_obs_theta)]

            agents.append(
                Agent(init_obs_pos[0], init_obs_pos[1], final_obs_pos[0], final_obs_pos[1], 0.1, 0.5, init_obs_heading, tc.policy_dict[agent_type], UnicycleDynamics, [OtherAgentsStatesSensor], i+1)
            )

        for agent in agents:
            if hasattr(agent.policy, 'initialize_network'):
                agent.policy.initialize_network()
        
        self.gca_env.set_agents(agents)

        obs = self.gca_env.reset()
        self.obs_poses: np.ndarray = np.stack([agent.pos_global_frame for agent in agents[1:]])
        self.obs_vels: np.ndarray = np.stack([agent.vel_global_frame for agent in agents[1:]])

        self.gym_state: np.ndarray = np.zeros((self.state_dim, ), dtype=np.float32)
        self.gym_state[:3] = self.state_of(agents[0])
        self.gym_state[3:5] = self.get_closest_obs_delta_xy()
        self.gym_state[5:] = self.get_closest_pred_obs_delta_xy(agents[0].vel_global_frame)

        return obs if raw_obs else self._make_observation(self.gym_state)

    def state_of(self, agent: Agent):
        return np.array([agent.pos_global_frame[0], agent.pos_global_frame[1], agent.heading_global_frame], dtype=np.float32)

    def to_dynstate(self, x):
        if x.shape[-1] == self.state_dim:
            return x
        elif x.shape[-1] == self.obs_dim:
            pos_x = x[..., 0]
            pos_y = x[..., 1]
            heading = np.arctan2(x[..., 3], x[..., 2])
            x_from_obs = x[..., 4]
            y_from_obs = x[..., 5]
            vx_from_obs = x[..., 6]
            vy_from_obs = x[..., 7]
            return np.stack([pos_x, pos_y, heading, x_from_obs, y_from_obs, vx_from_obs, vy_from_obs], axis=-1)

    def step(self, action: np.ndarray, raw_obs: bool = False) -> Tuple[np.ndarray, float, bool, Dict]:
        # 0 is always the ego-agent for us.
        obs, reward, done, info = self.gca_env.step({0: action})

        self.obs_poses: np.ndarray = np.stack([agent.pos_global_frame for agent in self.gca_env.agents[1:]])
        self.gym_state: np.ndarray = np.zeros((self.state_dim, ), dtype=np.float32)
        self.gym_state[:3] = self.state_of(self.gca_env.agents[0])
        self.gym_state[3:5] = self.get_closest_obs_delta_xy() # While the min in here might seem annoying, from a linearized perspective it's fine.
        self.gym_state[5:] = self.get_closest_pred_obs_delta_xy(self.gca_env.agents[0].vel_global_frame)

        if raw_obs:
            return obs, reward, done, info
        else:
            return self._make_observation(self.gym_state), reward, done, info

    def gen_expert_data(self, num_samples, theta=None, test=False, plot_trajs=False, seed=None, 
                        extra_callbacks: List[BaseCallback] = [], model_name='gt'):
        """This environment's reward is, unfortunately, non-convex. Thus, we cannot use
        cvxpy to solve this (at least, not in one step), so we use SAC as our underlying
        solver.
        """
        if theta is None:
            theta = self.gt_theta

        fname = os.path.join(Path(__file__).parent, f'cached_data/{self.env_name}_{num_samples}.npz')
        if os.path.exists(fname) and not test:
            npz_data = np.load(fname)
            extra_info = {'ep_lengths': torch.tensor(npz_data['ep_lengths'], dtype=torch.int)}
            return torch.tensor(npz_data['expert_xs'], dtype=torch.float32), torch.tensor(npz_data['expert_us'], dtype=torch.float32), extra_info

        if seed is not None:
            np.random.seed(seed)

        trajs = list()
        controls = list()
        rewards = list()
        ep_length = list()
        pbar = tqdm(total=num_samples, desc='Generating Data')
        collected = 0
        while collected < num_samples:
            obs = self.reset(raw_obs=True)
            visited_states = [self.gym_state.copy()]
            enacted_controls = list()
            collected_rewards = 0.0

            steps = 0
            while steps < self.num_timesteps:
                action = self.gca_env.agents[0].policy.find_next_action(obs[0], self.gca_env.agents, 0)
                obs, reward, done, info = self.step(action, raw_obs=True)
                steps += 1

                visited_states.append(self.gym_state.copy())
                enacted_controls.append(action)
                collected_rewards += reward
                if done:
                    break

            if np.any([agent.in_collision for agent in self.gca_env.agents]):
                pbar.set_description(f'Collision in {collected}')
                continue

            visited_states_np = np.array(visited_states)
            # Adding this zero control at the end so that every state has a paired control
            # (only necessary for computing the linearized A and B matrices), and even then
            # we remove it out of the computed B tensor's last timestep.
            enacted_controls_np = np.array(enacted_controls + [np.zeros((self.control_dim, ))])

            to_add = self.num_timesteps - steps
            trajs.append(np.concatenate((visited_states_np, 
                                         np.full((to_add, self.state_dim), np.nan))))
            controls.append(np.concatenate((enacted_controls_np, 
                                            np.full((to_add, self.control_dim), np.nan))))
            rewards.append(collected_rewards)
            ep_length.append(steps)

            collected += 1
            pbar.set_description(f'Ep Length: {steps}')
            pbar.update()

        # This saves the last iteration's plots and animation.
        self.reset()

        expert_xs, expert_us = np.stack(trajs), np.stack(controls)
        expert_xs, expert_us = expert_xs.reshape(num_samples, -1), expert_us.reshape(num_samples, -1)
        if not test:
            np.savez(fname, expert_xs=expert_xs, expert_us=expert_us, ep_lengths=np.array(ep_length))

        extra_info = {'ep_lengths': np.array(ep_length, dtype=int)}
        return torch.tensor(expert_xs, dtype=torch.float32), torch.tensor(expert_us, dtype=torch.float32), extra_info
