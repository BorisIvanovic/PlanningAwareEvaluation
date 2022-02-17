import os
import torch
import torch.autograd.functional as AF
import numpy as np
import cvxpy as cp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Callable, Dict, Optional
from tqdm import trange, tqdm
from pathlib import Path
from functools import partial
from pred_metric.environment import Environment


# No OpenAI gym functions because this environment's reward
# is convex we use cvxpy as our underlying problem solver.
class HardEnvironment(Environment):
    def __init__(self, env_name: str, state_dim: int, control_dim: int, num_timesteps: int, dt: float, gt_theta: np.ndarray) -> None:
        super().__init__(env_name, state_dim, control_dim, num_timesteps, dt, gt_theta)

        self.A = np.eye(self.state_dim, dtype=np.float32)
        self.A[[0, 1], [2, 3]] = self.dt

        self.B = np.zeros((self.state_dim, self.control_dim), dtype=np.float32)
        self.B[[2, 3], [0, 1]] = self.dt

        self.J = self._form_J(self.A, self.B, self.num_timesteps)

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

    def gt_reward_fn(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor,
                     extra_info: Dict[str, torch.Tensor]):
        if len(x.shape) == 1:
            x_reshaped = x.unsqueeze(0).reshape((1, -1, self.state_dim))
            u_reshaped = u.unsqueeze(0).reshape((1, -1, self.control_dim))
        else:
            x_reshaped = x.reshape((x.shape[0], -1, self.state_dim))
            u_reshaped = u.reshape((u.shape[0], -1, self.control_dim))

        running_costs = theta[:3] @ torch.stack([x_reshaped.square().sum((-1, -2)), 
                                                 u_reshaped.square().sum((-1, -2)), 
                                                 -x_reshaped[..., 1].sum(-1)])

        T = self.num_timesteps
        intermediate_poses = extra_info['intermediate_poses']
        other_costs = theta[3:] @ torch.stack([x_reshaped[:, T, :].square().sum(-1),
                                               (x_reshaped[:, T//2 - 10 : T//2 - 4, :2] - intermediate_poses.unsqueeze(1)).square().sum((-1, -2))])

        # The mean is across examples in the batch
        return -(running_costs + other_costs).mean()

    def dynamics_fn(self, x_t, u_t):
        return self.A @ x_t + self.B @ u_t

    def cvx_running_costs(self, x_tp1, u_t):
        return cp.vstack([cp.sum_squares(x_tp1), 
                          cp.sum_squares(u_t),
                          -x_tp1[1],
                          0, 
                          0])

    def cvx_running_constraints(self, x_tp1, u_t):
        return [cp.norm_inf(u_t) <= 1]

    def cvx_other_costs(self, x, u, intermediate_pos):
        T = self.num_timesteps
        return cp.vstack([0, 
                          0, 
                          0,
                          cp.sum_squares(x[:, T]),
                          sum(cp.sum_squares(x[:2, T//2 - t] - intermediate_pos) for t in range(5, 11))])

    def cvx_other_constraints(self, x, u):
        return []

    def gen_expert_data(self, num_samples, theta=None, test=False, plot_trajs=False, seed=None):
        if theta is None:
            theta = self.gt_theta

        fname = os.path.join(Path(__file__).parent, f'cached_data/{self.env_name}_{num_samples}.npz')
        if os.path.exists(fname) and not test:
            npz_data = np.load(fname)
            extra_info = {'intermediate_poses': torch.tensor(npz_data['intermediate_poses'], dtype=torch.float32)}
            return torch.tensor(npz_data['expert_xs'], dtype=torch.float32), torch.tensor(npz_data['expert_us'], dtype=torch.float32), extra_info

        n, m, T = self.state_dim, self.control_dim, self.num_timesteps

        if seed is not None:
            np.random.seed(seed)

        trajs = list()
        controls = list()
        intermediate_poses = list()
        for _ in trange(num_samples, desc='Generating Data'):
            intermediate_pos = np.random.rand(2)
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
            cost += theta @ self.cvx_other_costs(x, u, intermediate_pos)
            constr += [x[:, 0] == x_0] + self.cvx_other_constraints(x, u)

            problem = cp.Problem(cp.Minimize(cost), constr)
            problem.solve(solver=cp.ECOS)

            if plot_trajs:
                solved_traj = x.value
                solved_controls = u.value
                
                fig, ax = plt.subplots()
                ax.plot(solved_traj[0], solved_traj[1])
                ax.scatter([intermediate_pos[0]], [intermediate_pos[1]], marker='o', c='red')
                ax.scatter([x_0[0]], [x_0[1]], marker='o', c='green')
                ax.scatter([0], [0], marker='o', c='blue')
                ax.set_xlim(min(-0.1, solved_traj[0].min()), max(1.1, solved_traj[0].max()))
                ax.set_ylim(min(-0.1, solved_traj[1].min()), max(1.1, solved_traj[1].max()))
                plt.show()
                
                fig, ax = plt.subplots()
                ax.plot(np.arange(T), solved_controls[0], label=r'$a_x$')
                ax.plot(np.arange(T), solved_controls[1], label=r'$a_y$')
                ax.legend(loc='best')
                plt.show()

            trajs.append(x.value.T.flatten())
            controls.append(u.value.T.flatten())
            intermediate_poses.append(intermediate_pos)

        expert_xs, expert_us, intermediate_poses = np.stack(trajs), np.stack(controls), np.stack(intermediate_poses)
        if not test:
            np.savez(fname, expert_xs=expert_xs, expert_us=expert_us, intermediate_poses=intermediate_poses)

        extra_info = {'intermediate_poses': torch.tensor(intermediate_poses, dtype=torch.float32)}
        return torch.tensor(expert_xs, dtype=torch.float32), torch.tensor(expert_us, dtype=torch.float32), extra_info

    def _run_single_hessian(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, 
                            extra_info: Dict[str, torch.Tensor]):
        (H_hat, _), (_, H_tilde) = AF.hessian(partial(self.gt_reward_fn, 
                                                      theta=theta[:-1], 
                                                      extra_info=extra_info), 
                                              (x, u),
                                              create_graph=True)

        H = H_tilde + self.J @ H_hat @ torch.transpose(self.J, -1, -2)
        
        # Adding the H regularization term here
        return H - torch.exp(theta[-1]) * torch.eye(self.J.shape[0])

    def hessian(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, 
                extra_info: Dict[str, torch.Tensor]):
        if len(x.shape) > 1:
            # The only time one would call hessian(...) with a batch of inputs for the hard environment is for finding a feasible theta_r,
            # so we return all of them as a batch and let the feasible finding function take the determinant of all of them.
            intermediate_poses = extra_info['intermediate_poses']
            Hs = list()

            for batch in range(x.shape[0]):
                Hs.append(self._run_single_hessian(x[batch], u[batch], theta, 
                                                   extra_info={'intermediate_poses': intermediate_poses[[batch]]}))

            return torch.stack(Hs)
        else:
            return self._run_single_hessian(x, u, theta, extra_info)

    def log_likelihood(self, x: torch.Tensor, u: torch.Tensor, 
                       extra_info: Dict[str, torch.Tensor], 
                       theta=None, 
                       mu_r=0, lam_r=0):
        intermediate_poses = extra_info['intermediate_poses']
        if theta is None:
            theta = torch.cat([torch.tensor(self.gt_theta), torch.tensor([-np.inf])])
        
        batch_size = x.shape[0]

        ll_val = 0.0
        for batch in range(batch_size):
            g_hat, g_tilde = AF.jacobian(partial(self.gt_reward_fn, 
                                                 theta=theta[:-1], 
                                                 extra_info={'intermediate_poses': intermediate_poses[[batch]]}), 
                                         (x[batch], u[batch]),
                                         create_graph=True)

            g = g_tilde + self.J @ g_hat
            H = self.hessian(x[batch], u[batch], theta, {'intermediate_poses': intermediate_poses[[batch]]})

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
