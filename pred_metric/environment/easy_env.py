import os
import torch
import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, Dict, Tuple
from tqdm import trange, tqdm
from pathlib import Path

from pred_metric.environment import Environment


class EasyEnvironment(Environment):
    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray) -> None:
        super().__init__(env_name, state_dim, control_dim, num_timesteps, dt, gt_theta)

        self.A = np.eye(self.state_dim, dtype=np.float32)
        self.A[[0, 1], [2, 3]] = self.dt

        self.B = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        self.B[[2, 3], [0, 1]] = self.dt

        self.J = self._form_J(self.A, self.B, self.num_timesteps)

    # OpenAI Gym Functions
    # This is mainly a sanity check, since this environment's reward
    # is convex we use cvxpy as our underlying problem solver.
    def _unnormalize_action(self, action: np.ndarray) -> np.ndarray:
        """By default the action is normalized to lie within [-1, 1],
        this function scales the normalized action up to metric units.

        In a double integrator, actions are x and y accelerations. 
        Here, we make the maximum acceleration 1 m/s^2, so we will just
        return the action as-is.
        """
        return action

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        unnormed_control = self._unnormalize_action(action)
        reward = self.step_reward_fn(self.gym_state, unnormed_control)
        self.gym_state = self.dynamics_fn(self.gym_state, unnormed_control)
        self.curr_timestep += 1

        done = (np.linalg.norm(self.gym_state[:2]) < 0.25).item()
        reward += 100.0 if done else 0

        if self.curr_timestep >= self.num_timesteps:
            done = True

        info = {}
        return self.gym_state, reward, done, info

    def reset(self):
        self.gym_state = np.random.rand(self.state_dim)
        self.gym_state[2:] = np.random.randn(self.state_dim - 2)

        self.curr_timestep = 0
        return self.gym_state

    def _form_J(self, A, B, T):
        J_lists = list()
        for t2 in trange(T, desc='Building J'):
            J_lists.append(list())
            for t1 in range(T+1):
                if t1 == t2:
                    J_lists[t2].append(B.T)
                elif t1 > t2:
                    J_lists[t2].append(J_lists[t2][t1-1] @ A.T)
                else:
                    J_lists[t2].append(np.zeros_like(B.T))

        return torch.tensor(np.block(J_lists))

    def step_reward_fn(self, x_t, u_t, extra_info=None):
        return self.gt_theta[0] / (np.linalg.norm(x_t).sum() + 1e-2) - self.gt_theta[1] * np.square(u_t).sum()

    def gt_reward_fn(self, x, u, theta, extra_info=None):
        final_term = theta[0] * torch.square(x[..., -4:]).sum(dim=-1)
        running_terms = (theta[0] * torch.square(x[..., :-4]).sum(dim=-1) 
                         + theta[1] * torch.square(u).sum(dim=-1))

        # The mean is across examples in the batch
        return -(final_term + running_terms).mean()

    def dynamics_fn(self, x_t, u_t):
        return self.A @ x_t + self.B @ u_t

    def cvx_running_costs(self, x_tp1, u_t):
        return cp.vstack([cp.sum_squares(x_tp1), cp.sum_squares(u_t)])

    def cvx_running_constraints(self, x_tp1, u_t):
        return [cp.norm_inf(u_t) <= 1]

    def cvx_other_costs(self, x, u):
        return np.zeros(self.theta_dim)

    def cvx_other_constraints(self, x, u):
        return []

    def gen_expert_data(self, num_samples, theta=None, test=False, plot_trajs=False, seed=None):
        if theta is None:
            theta = self.gt_theta

        fname = os.path.join(Path(__file__).parent, f'cached_data/{self.env_name}_{num_samples}.npz')
        if os.path.exists(fname) and not test:
            npz_data = np.load(fname)
            return torch.tensor(npz_data['expert_xs'], dtype=torch.float32), torch.tensor(npz_data['expert_us'], dtype=torch.float32), None

        n, m, T = self.state_dim, self.control_dim, self.num_timesteps

        if seed is not None:
            np.random.seed(seed)

        trajs = list()
        controls = list()
        for _ in trange(num_samples, desc='Generating Data'):
            x_0 = np.random.rand(n)
            # x_0[2:] = 0

            x = cp.Variable((n, T + 1))
            u = cp.Variable((m, T))

            cost = 0
            constr = []
            for t in range(T):
                cost += theta @ self.cvx_running_costs(x[:, t + 1], u[:, t])
                constr += [x[:, t + 1] == self.dynamics_fn(x[:, t], u[:, t])] + self.cvx_running_constraints(x[:, t + 1], u[:, t])
            
            # sums problem objectives and concatenates constraints.
            cost += theta @ self.cvx_other_costs(x, u)
            constr += [x[:, 0] == x_0] + self.cvx_other_constraints(x, u)

            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(solver=cp.ECOS)

            if plot_trajs:
                solved_traj = x.value
                solved_controls = u.value
                
                fig, ax = plt.subplots()
                ax.plot(solved_traj[0], solved_traj[1])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                plt.show()
                
                fig, ax = plt.subplots()
                ax.plot(np.arange(T), solved_controls[0], label=r'$a_x$')
                ax.plot(np.arange(T), solved_controls[1], label=r'$a_y$')
                ax.legend(loc='best')
                plt.show()

            trajs.append(x.value.T.flatten())
            controls.append(u.value.T.flatten())

        expert_xs, expert_us = np.stack(trajs), np.stack(controls)
        if not test:
            np.savez(fname, expert_xs=expert_xs, expert_us=expert_us)

        return torch.tensor(expert_xs, dtype=torch.float32), torch.tensor(expert_us, dtype=torch.float32), None

    def hessian(self, x, u, theta, extra_info=None):
        H_hat = (-2 * torch.eye(self.J.shape[1])).unsqueeze(0)
        H_tilde = (-2 * torch.eye(self.J.shape[0])).unsqueeze(0)

        H = theta[1] * H_tilde + self.J.unsqueeze(0) @ (theta[0] * H_hat) @ torch.transpose(self.J.unsqueeze(0), -1, -2)
        
        # Adding the H regularization term here
        return H - torch.exp(theta[-1]) * torch.eye(self.J.shape[0]).unsqueeze(0)

    def log_likelihood(self, x: torch.Tensor, u: torch.Tensor, extra_info=None, theta=None, mu_r=0, lam_r=0):
        if theta is None:
            theta = torch.cat([torch.tensor(self.gt_theta), torch.tensor([-np.inf])])
        
        g_hat = -2 * x
        g_tilde = -2 * u

        g = theta[1] * g_tilde.unsqueeze(-1) + self.J.unsqueeze(0) @ (theta[0] * g_hat.unsqueeze(-1))
        H = self.hessian(x, u, theta)

        ll_val = 0.5 * torch.transpose(g, -1, -2) @ torch.linalg.solve(H, g) + 0.5 * torch.logdet(-H)
        if np.isinf(mu_r):
            # We've switched off the constrained variable (auglagcost.m:17)
            return ll_val.mean()
        else:
            return ll_val.mean() - 0.5*mu_r*(torch.exp(theta[-1])**2) + lam_r*torch.exp(theta[-1])

    def plot_loss_landscape(self,
                            fig_path: str,
                            expert_x: torch.Tensor, 
                            expert_u: torch.Tensor,
                            extra_info=None,
                            theta_lims=None,
                            theta_r: float = -np.inf):
        loss_landscape = {f'theta{i+1}': [] for i in range(self.theta_dim)}
        loss_landscape['loss'] = []

        x_data = np.linspace(0, 2, num=101, dtype=np.float32)
        y_data = np.linspace(0, 2, num=101, dtype=np.float32)
        for theta1 in tqdm(x_data, desc='Getting NLLs'):
            for theta2 in y_data:
                if theta1 == theta2 == 0.0:
                    continue
                
                loss = -self.log_likelihood(x=expert_x, 
                                            u=expert_u,
                                            extra_info=extra_info,
                                            theta=torch.tensor([theta1, theta2, theta_r]))

                loss_landscape['theta1'].append(theta1)
                loss_landscape['theta2'].append(theta2)
                loss_landscape['loss'].append(loss.item())

        loss_landscape_df = pd.DataFrame(loss_landscape)

        fig, ax = plt.subplots()
        sns.heatmap(loss_landscape_df.pivot('theta2', 'theta1', 'loss'),
                    vmin=-50, vmax=50, cmap='rocket_r', square=True, ax=ax)
        ax.invert_yaxis()
        ax.set_xticklabels([f'{float(x.get_text()):.1f}' for x in ax.get_xticklabels()])
        ax.set_yticklabels([f'{float(y.get_text()):.1f}' for y in ax.get_yticklabels()])
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300)

    def plot_reward_landscape(self,
                              fig_path: str,
                              expert_x: torch.Tensor, 
                              expert_u: torch.Tensor,
                              extra_info=None,
                              theta_lims=None):
        reward_landscape = {f'theta{i+1}': [] for i in range(self.theta_dim)}
        reward_landscape['reward'] = []

        x_data = np.linspace(0, 2, num=201, dtype=np.float32)
        y_data = np.linspace(0, 2, num=201, dtype=np.float32)
        for theta1 in tqdm(x_data, desc='Getting Rewards'):
            for theta2 in y_data:
                reward = self.gt_reward_fn(expert_x, expert_u, 
                                           torch.tensor([theta1, theta2]))

                reward_landscape['theta1'].append(theta1)
                reward_landscape['theta2'].append(theta2)
                reward_landscape['reward'].append(reward.item())

        reward_landscape_df = pd.DataFrame(reward_landscape)

        fig, ax = plt.subplots()
        sns.heatmap(reward_landscape_df.pivot('theta2', 'theta1', 'reward'),
                    vmin=-50, vmax=0, cmap='rocket', square=True, ax=ax)
        ax.invert_yaxis()
        ax.set_xticklabels([f'{float(x.get_text()):.1f}' for x in ax.get_xticklabels()])
        ax.set_yticklabels([f'{float(y.get_text()):.1f}' for y in ax.get_yticklabels()])
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        fig.tight_layout()
        fig.savefig(fig_path, dpi=300)
