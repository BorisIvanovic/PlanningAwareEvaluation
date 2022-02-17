import os
import torch
import torch.autograd.functional as AF
import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, Dict, List, Optional, Tuple
from tqdm import trange, tqdm
from pathlib import Path
from functools import partial
from pred_metric.environment import Environment
from pred_metric.environment import utils

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from gym import spaces
from gym.wrappers import TimeLimit


class UnicycleEnvironment(Environment):
    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray) -> None:
        super().__init__(env_name, state_dim, control_dim, num_timesteps, dt, gt_theta)

        # x  y  cos(heading)  sin(heading)  speed
        bounds = np.array([np.inf, np.inf, 1., 1., np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low=-bounds, high=bounds)
        self.obs_dim = self.observation_space.shape[0]

    def _form_J(self, A, B, extra_info, print=True):
        T = extra_info['ep_lengths'].squeeze()

        J_rows = list()
        for t2 in trange(T, desc='Building J', disable = not print):
            row_cols = list()
            for t1 in range(T+1):
                if t1 == t2:
                    row_cols.append(B[t1].T)
                elif t1 > t2:
                    row_cols.append(row_cols[t1-1] @ A[t1].T)
                else:
                    row_cols.append(torch.zeros_like(B[t1].T))
            
            J_rows.append(torch.stack(row_cols))

        # almost_J.shape is (J_rows, J_cols, block_rows, block_cols)
        almost_J = torch.stack(J_rows)
        (J_rows, J_cols, block_rows, block_cols) = almost_J.shape
        return almost_J.permute(0, 2, 1, 3).reshape(J_rows*block_rows, J_cols*block_cols)

    def _ensure_length_nd(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        if extra_info is not None:
            ep_lens = extra_info['ep_lengths']

            x_reshaped = x[..., :ep_lens+1, :]
            u_reshaped = u[..., :ep_lens+1, :]
            # Again, this is one more timesteps than there should be for u, 
            # the last is all zero, and is ignored in the creation of B later.
        
            return x_reshaped, u_reshaped
        else:
            return x, u

    def unvec_xu(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        if len(x.shape) == 1:
            x_reshaped = x.unsqueeze(0).reshape((1, -1, self.state_dim))
            u_reshaped = u.unsqueeze(0).reshape((1, -1, self.control_dim))
        else:
            x_reshaped = x.reshape((x.shape[0], -1, self.state_dim))
            u_reshaped = u.reshape((u.shape[0], -1, self.control_dim))

        return self._ensure_length_nd(x_reshaped, u_reshaped, extra_info)

    def calc_A(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        one = torch.tensor(1)
        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([4, 4]),
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

        # Using u_t for shape because it has the correct # of timesteps.
        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([4, 4]),
                           dtype=torch.float32)

        F_sm[..., 0, 0] = one
        F_sm[..., 1, 1] = one
        F_sm[..., 2, 2] = one
        F_sm[..., 3, 3] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def calc_B(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([4, 2]),
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

        # Using u_t for shape because it has the correct # of timesteps.
        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([4, 2]),
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2

        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def linear_dynamics_fn(self, x, u):
        x_t = x.reshape(-1, self.state_dim)
        u_t = u.reshape(-1, self.control_dim)

        A_t = self.calc_A(x_t, u_t)
        B_t = self.calc_B(x_t, u_t)
        linear_next = (A_t @ x_t.unsqueeze(-1) + B_t @ u_t.unsqueeze(-1)).squeeze(-1)

        with torch.no_grad():
            c_t = self.nonlinear_dynamics_fn(x_t, u_t) - linear_next

        return linear_next + c_t

    def dynamics_fn(self, x_t, u_t):
        # state x is [x, y, heading, along-track speed]
        # control u is [steering rate, along-track acceleration]
        return self.nonlinear_dynamics_fn(x_t, u_t)

    def nonlinear_dynamics_fn(self, x, u):
        x_t = x.reshape(-1, self.state_dim)
        u_t = u.reshape(-1, self.control_dim)

        x_p = x_t[..., 0]
        y_p = x_t[..., 1]
        phi = x_t[..., 2]
        v = x_t[..., 3]
        dphi = u_t[..., 0]
        a = u_t[..., 1]

        mask = np.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + (mask) * 1

        phi_p_omega_dt = phi + dphi * self.dt
        dsin_domega = (np.sin(phi_p_omega_dt) - np.sin(phi)) / dphi
        dcos_domega = (np.cos(phi_p_omega_dt) - np.cos(phi)) / dphi

        d1 = np.stack([(x_p
                           + (a / dphi) * dcos_domega
                           + v * dsin_domega
                           + (a / dphi) * np.sin(phi_p_omega_dt) * self.dt),
                          (y_p
                           - v * dcos_domega
                           + (a / dphi) * dsin_domega
                           - (a / dphi) * np.cos(phi_p_omega_dt) * self.dt),
                          phi + dphi * self.dt,
                          v + a * self.dt], axis=-1)
        d2 = np.stack([x_p + v * np.cos(phi) * self.dt + (a / 2) * np.cos(phi) * self.dt ** 2,
                          y_p + v * np.sin(phi) * self.dt + (a / 2) * np.sin(phi) * self.dt ** 2,
                          phi * np.ones_like(a),
                          v + a * self.dt], axis=-1)
        return np.where(~np.expand_dims(mask, axis=-1), d1, d2)

    # OpenAI Gym Functions
    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """By default the action is normalized to lie within [-1, 1],
        this function scales the normalized action up to metric units.

        In a dynamically-extended unicycle, actions are steering rate and along-track acceleration. 
        Here, we make the maximum acceleration 1 m/s^2 and turining rate 1 rad/s, so we will just
        return the action as-is.
        """
        return action

    def _make_observation(self, state):
        #                   x         y        cos(heading)      sin(heading)     speed
        return np.array([state[0], state[1], np.cos(state[2]), np.sin(state[2]), state[3]])

    def step_reward_fn(self, x_t, u_t, extra_info=None):
        return -(self.gt_theta[0] * np.square(x_t).sum() + self.gt_theta[1] * np.square(u_t).sum())

    def gt_reward_fn(self, x, u, theta, extra_info=None):
        x_reshaped, u_reshaped = self.unvec_xu(x, u, extra_info)

        # The mean is across examples in the batch
        return -torch.mean((theta[0] * torch.square(x_reshaped)).sum(-1) + theta[1] * torch.square(u_reshaped).sum((-1, -2)))

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        unnormed_control = self._unnormalize_action(action)
        reward = self.step_reward_fn(self.gym_state, unnormed_control)
        self.gym_state = np.squeeze(self.dynamics_fn(self.gym_state, unnormed_control))
        self.curr_timestep += 1

        done = (np.linalg.norm(self.gym_state[:2]) < 0.15).item()
        reward += 100.0 if done else 0

        info = {}
        return self._make_observation(self.gym_state), reward, done, info

    def reset(self):
        self.gym_state = np.random.rand(self.state_dim)
        self.gym_state[2:] = np.random.randn(self.state_dim - 2)

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
            return np.stack([pos_x, pos_y, heading, speed], axis=-1)

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

        # We will first come up with an RL agent that performs well on this environment, and then sample num_samples
        # trajectories from it. In essence, the well-trained RL agent is our expert here. In the real world,
        # data would be used, but this is a good check of it CIOC can recover loss function weighting given the
        # actual loss function used by the expert agent.
        with np.printoptions(precision=3, suppress=True, formatter={'float': '{: 0.3f}'.format}):
            theta_str = str(theta)

        tb_log_path = f'./{self.env_name}_logs/'
        tb_log_name = 'default_' + theta_str + (f'_{model_name}' if len(model_name) > 0 else '')

        model_dir = os.path.join(tb_log_path, tb_log_name)
        model_file = f'best_model.zip'
        model_path = os.path.join(model_dir, model_file)

        if os.path.exists(model_path):
            sac_model = SAC.load(model_path)
        else:
            rl_train_env = Monitor(TimeLimit(self.__class__(self.env_name, 
                                                            self.state_dim, 
                                                            self.control_dim, 
                                                            self.num_timesteps,
                                                            self.dt, 
                                                            theta), 
                                             max_episode_steps=self.num_timesteps))
            sac_model = SAC('MlpPolicy', rl_train_env, verbose=1, tensorboard_log=tb_log_path)

            total_timesteps = 20_000
            if np.allclose(theta, self.gt_theta):
                total_timesteps = 50_000

            rl_eval_env = Monitor(TimeLimit(self.__class__(self.env_name, 
                                                           self.state_dim, 
                                                           self.control_dim, 
                                                           self.num_timesteps, 
                                                           self.dt, 
                                                           theta), 
                                            max_episode_steps=self.num_timesteps))
            eval_callback = EvalCallback(rl_eval_env, best_model_save_path=model_dir,
                                         log_path=model_dir, eval_freq=500, n_eval_episodes=100,
                                         deterministic=True, render=False)

            sac_model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=[eval_callback] + extra_callbacks)
            sac_model = SAC.load(model_path) # Loading the best model for data collection

        if seed is not None:
            np.random.seed(seed)

        trajs = list()
        controls = list()
        rewards = list()
        ep_length = list()
        for _ in trange(num_samples, desc='Generating Data'):
            obs = self.reset()
            visited_states = [obs]
            enacted_controls = list()
            collected_rewards = 0.0

            steps = 0
            while steps < self.num_timesteps:
                action, _states = sac_model.predict(obs, deterministic=True)
                obs, reward, done, info = self.step(action)
                steps += 1

                visited_states.append(obs)
                enacted_controls.append(action)
                collected_rewards += reward
                if done:
                    break

            visited_states_np = np.array(visited_states)
            # Adding this zero control at the end so that every state has a paired control
            # (only necessary for computing the linearized A and B matrices), and even then
            # we remove it out of the computed B tensor's last timestep.
            enacted_controls_np = np.array(enacted_controls + [np.zeros((self.control_dim, ))])

            visited_states_np = self.to_dynstate(visited_states_np)

            to_add = self.num_timesteps - steps
            trajs.append(np.concatenate((visited_states_np, 
                                         np.full((to_add, self.state_dim), np.nan))))
            controls.append(np.concatenate((enacted_controls_np, 
                                            np.full((to_add, self.control_dim), np.nan))))
            rewards.append(collected_rewards)
            ep_length.append(steps)

        expert_xs, expert_us = np.stack(trajs), np.stack(controls)
        expert_xs, expert_us = expert_xs.reshape(num_samples, -1), expert_us.reshape(num_samples, -1)
        if not test:
            np.savez(fname, expert_xs=expert_xs, expert_us=expert_us, ep_lengths=np.array(ep_length))

        extra_info = {'ep_lengths': np.array(ep_length, dtype=int)}
        return torch.tensor(expert_xs, dtype=torch.float32), torch.tensor(expert_us, dtype=torch.float32), extra_info

    def _ensure_length(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        if extra_info is not None:
            ep_len = extra_info['ep_lengths']
            x_proper = x[:(ep_len+1)*self.state_dim]
            u_proper = u[:(ep_len)*self.control_dim]

            return x_proper, u_proper
        else:
            return x, u

    def _run_single_hessian(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor,
                            extra_info: Dict[str, torch.Tensor], J: Optional[torch.Tensor] = None):
        x_proper, u_proper = self._ensure_length(x, u, extra_info)

        (H_hat, _), (_, H_tilde) = AF.hessian(partial(self.gt_reward_fn, 
                                                      theta=theta[:-1], 
                                                      extra_info=extra_info), 
                                              (x_proper, u_proper),
                                              create_graph=True,
                                              vectorize=True)

        if J is None:
            J = self._form_J(self.calc_A(x, u, extra_info)[0], self.calc_B(x, u, extra_info)[0, :-1], extra_info, print=False)

        H = H_tilde + J @ H_hat @ torch.transpose(J, -1, -2)
        
        # Adding the H regularization term here
        return H - torch.exp(theta[-1]) * torch.eye(J.shape[0])

    def hessian(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, 
                extra_info: Dict[str, torch.Tensor], J: Optional[torch.Tensor] = None):
        if len(x.shape) > 1:
            # The only time one would call hessian(...) with a batch of inputs for the hard environment is for finding a feasible theta_r,
            # so we return all of them as a batch and let the feasible finding function take the determinant of all of them.
            Hs = list()

            for batch in range(x.shape[0]):
                Hs.append(self._run_single_hessian(x[batch], u[batch], theta, 
                                                   extra_info=utils.index_dict(extra_info, batch), J=J))

            return Hs
        else:
            return self._run_single_hessian(x, u, theta, extra_info, J)

    def _run_single_jacobian(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor,
                            extra_info: Dict[str, torch.Tensor]):
        return AF.jacobian(partial(self.gt_reward_fn, 
                                   theta=theta[:-1], 
                                   extra_info=extra_info), 
                            (x, u),
                            create_graph=True,
                            vectorize=True)

    def log_likelihood(self, x: torch.Tensor, u: torch.Tensor, 
                       extra_info: Dict[str, torch.Tensor], 
                       theta=None, 
                       mu_r=0, lam_r=0):
        if theta is None:
            theta = torch.cat([torch.tensor(self.gt_theta), torch.tensor([-np.inf])])
        
        batch_size = x.shape[0]

        ll_val = 0.0
        for batch in range(batch_size):
            indexed_extra_info = utils.index_dict(extra_info, batch)
            x_proper, u_proper = self._ensure_length(x[batch], u[batch], indexed_extra_info)

            g_hat, g_tilde = self._run_single_jacobian(x_proper, u_proper, theta, 
                                                       extra_info=indexed_extra_info)

            A_batch = self.calc_A(x[batch], u[batch], indexed_extra_info)
            B_batch = self.calc_B(x[batch], u[batch], indexed_extra_info)
            J = self._form_J(A_batch[0], B_batch[0, :-1], indexed_extra_info, print=False)

            g = g_tilde + J @ g_hat
            H = self.hessian(x[batch], u[batch], theta, indexed_extra_info, J)

            ll_val += 0.5 * g @ torch.linalg.solve(H, g) + 0.5 * torch.logdet(-H)
        
        if np.isinf(mu_r):
            # We've switched off the constrained variable (auglagcost.m:17)
            return ll_val/batch_size
        else:
            return ll_val/batch_size - 0.5*mu_r*(torch.exp(theta[-1])**2) + lam_r*torch.exp(theta[-1])

    def plot_loss_landscape(self,
                            fig_path: str,
                            expert_x: torch.Tensor, 
                            expert_u: torch.Tensor,
                            extra_info: Dict[str, torch.Tensor],
                            theta_lims: Optional[Dict[int, np.ndarray]] = None,
                            theta_r: float = -np.inf):
        loss_landscape = {'theta_idx': [], 'idx_val': [], 'loss': []}
        if theta_lims is None:
            theta_lims = {0: np.linspace(0, 1, num=41, dtype=np.float32),
                          1: np.linspace(0, 1, num=41, dtype=np.float32),
                          2: np.linspace(0, 1, num=41, dtype=np.float32),
                          3: np.concatenate([np.zeros((1, ), dtype=np.float32),
                                             np.logspace(-1, 1, num=41, dtype=np.float32)]),
                          4: np.concatenate([np.zeros((1, ), dtype=np.float32),
                                             np.logspace(-1, 1, num=41, dtype=np.float32)])}

        for theta_idx in trange(self.theta_dim, desc='Getting NLLs'):
            theta = torch.cat([torch.tensor(self.gt_theta), torch.tensor([theta_r])])
            for val in tqdm(theta_lims[theta_idx], leave=False, desc=f'Theta {theta_idx+1}'):
                theta[theta_idx] = val.item()

                loss = -self.log_likelihood(x=expert_x, 
                                            u=expert_u,
                                            extra_info=extra_info, 
                                            theta=theta)

                loss_landscape['theta_idx'].append(theta_idx)
                loss_landscape['idx_val'].append(val)
                loss_landscape['loss'].append(loss.item())

        loss_landscape_df = pd.DataFrame(loss_landscape)

        fig, axes = plt.subplots(nrows=self.theta_dim)
        cbar_ax = fig.add_axes([.91, .15, .03, .8])
        for i in range(self.theta_dim):
            ax = axes[i]
            mask = loss_landscape_df['theta_idx'] == i
            sns.heatmap(loss_landscape_df[mask].pivot('theta_idx', 'idx_val', 'loss'),
                        vmin=-50, vmax=50, cmap='rocket_r', 
                        cbar=(i==0), cbar_ax=None if i else cbar_ax, 
                        ax=ax)
            ax.set_xticklabels([f'{float(x.get_text()):.1f}' for x in ax.get_xticklabels()])
            ax.set_yticklabels([None])
            ax.set_xlabel(fr'$\theta_{i+1}$')
            ax.set_ylabel(None)

        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(fig_path, dpi=300)

    def plot_reward_landscape(self,
                              fig_path: str,
                              expert_x: torch.Tensor, 
                              expert_u: torch.Tensor,
                              extra_info: Dict[str, torch.Tensor],
                              theta_lims: Optional[Dict[int, np.ndarray]] = None):
        reward_landscape = {'theta_idx': [], 'idx_val': [], 'reward': []}
        if theta_lims is None:
            num_steps = 301
            theta_lims = {0: np.linspace(0, 2, num=num_steps, dtype=np.float32),
                          1: np.linspace(0, 2, num=num_steps, dtype=np.float32),
                          2: np.linspace(0, 2, num=num_steps, dtype=np.float32),
                          3: np.concatenate([np.zeros((1, ), dtype=np.float32),
                                             np.logspace(-1, 1, num=num_steps, dtype=np.float32)]),
                          4: np.concatenate([np.zeros((1, ), dtype=np.float32),
                                             np.logspace(-1, 1, num=num_steps, dtype=np.float32)])}

        for theta_idx in trange(self.theta_dim, desc='Getting Rewards'):
            theta = torch.tensor(self.gt_theta)
            for val in tqdm(theta_lims[theta_idx], leave=False, desc=f'Theta {theta_idx+1}'):
                theta[theta_idx] = val.item()

                reward = self.gt_reward_fn(expert_x, expert_u, theta, extra_info)

                reward_landscape['theta_idx'].append(theta_idx)
                reward_landscape['idx_val'].append(val)
                reward_landscape['reward'].append(reward.item())

        reward_landscape_df = pd.DataFrame(reward_landscape)

        fig, axes = plt.subplots(nrows=self.theta_dim, figsize=(8, 6))
        cbar_ax = fig.add_axes([.91, .15, .03, .8])
        for i in range(self.theta_dim):
            ax = axes[i]
            mask = reward_landscape_df['theta_idx'] == i
            sns.heatmap(reward_landscape_df[mask].pivot('theta_idx', 'idx_val', 'reward'),
                        vmin=-50, vmax=0, cmap='rocket', 
                        cbar=(i==0), cbar_ax=None if i else cbar_ax, 
                        ax=ax)
            ax.set_xticklabels([f'{float(x.get_text()):.1f}' for x in ax.get_xticklabels()])
            ax.set_yticklabels([None])
            ax.set_xlabel(fr'$\theta_{i+1}$')
            ax.set_ylabel(None)

        fig.tight_layout(rect=[0, 0, .9, 1])
        fig.savefig(fig_path, dpi=300)
