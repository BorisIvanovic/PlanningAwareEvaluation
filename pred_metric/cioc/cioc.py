import os
import time
import torch
from torch import nn
from torch import optim
import numpy as np
# torch.autograd.set_detect_anomaly(True)

from tqdm import trange, tqdm
from pred_metric.environment import Environment
from pred_metric.cioc.data_logger import DataLogger
from typing import Any, Dict, Optional

class CIOC(object):
    def __init__(self, env: Environment, verbose: bool = True) -> None:
        self.env = env

        self.verbose = verbose
        self._training_thetas = None
        self._training_losses = None

        self.logger = DataLogger(col_names=['total_iter', 'outer_iter', 'lbfgs_iter', 'nll'] + [f'theta_{i}' for i in range(env.gt_theta.shape[0])] + ['theta_r'], 
                                 filepath=os.path.join(f'logs/{env.env_name}', f'cioc_log_{time.strftime("%d_%b_%Y_%H_%M_%S", time.localtime())}.csv'),
                                 data_dir=f'logs/{env.env_name}')
        self.logger.update_indices({'total_iter': 0, 'outer_iter': 0, 'lbfgs_iter': 0})

    def _auglag_find_feasible(self, x: torch.Tensor, u: torch.Tensor, theta: torch.Tensor, 
                              extra_info: Optional[Dict[str, torch.Tensor]] = None) -> float:
        print('Finding a feasible theta_r', end='', flush=True)
        theta_r = 0.4096

        while True:
            print(f'... {theta_r}', end='', flush=True)
            Hs = self.env.hessian(x, u, torch.cat([theta, torch.tensor([np.log(theta_r)], dtype=torch.float32)]), 
                                 extra_info=extra_info)
            if isinstance(Hs, list):
                negH_det = min(torch.linalg.det(-H_individual) for H_individual in Hs)
            else:
                negH_det = torch.linalg.det(-Hs).min()

            if negH_det.item() > 1e-2:
                break
            else:
                theta_r *= 2

        print(f' works. Done!')
        
        # Multiplying again by 2 to be sure.
        return theta_r*2

    def _track_theta(self, theta):
        if self.verbose:
            with torch.no_grad():
                curr_theta = np.copy(theta.numpy())
                curr_theta[-1] = np.exp(curr_theta[-1])
                self._training_thetas.append(curr_theta)

    def _train_step(self, expert_x: torch.Tensor, expert_u: torch.Tensor,
                    extra_info: Optional[Dict[str, torch.Tensor]], 
                    theta: torch.Tensor, mu_r: float, lam_r: float, 
                    optimizer: optim.Optimizer, pbar: tqdm):
        def closure():
            optimizer.zero_grad()

            loss = -self.env.log_likelihood(x=expert_x,
                                            u=expert_u,
                                            extra_info=extra_info, 
                                            theta=theta, 
                                            mu_r=mu_r,
                                            lam_r=lam_r)
            
            # Keeping track of training losses
            if self.verbose:
                self._training_losses.append(loss.item())

            with torch.no_grad():
                curr_theta = theta.numpy()
                reg_val = np.exp(curr_theta[-1])
                
                # Logging
                logger_dict = {'nll': [loss.item()], 'theta_r': [reg_val]}
                for i in range(curr_theta.shape[0]-1):
                    logger_dict[f'theta_{i}'] = [curr_theta[i]]
                self.logger.add_rows(logger_dict)
                self.logger.increment('lbfgs_iter')
                self.logger.increment('total_iter')

                # Progress Bar
                pbar.set_description(f'NLL: {loss.item():.2f}, reg: {reg_val:.2g}, theta: {curr_theta[:-1]}')
            
            loss.backward()
            # optimizer.step()
            return loss

        self.logger.update_indices({'lbfgs_iter': 0})
        optimizer.step(closure)
        # closure()
        self.logger.increment('outer_iter')

        pbar.update()

        # Keeping track of training thetas
        self._track_theta(theta)

    def fit(self, 
            expert_x: torch.Tensor, 
            expert_u: torch.Tensor,
            extra_info: Optional[Dict[str, torch.Tensor]] = None,
            init_theta: Optional[torch.Tensor] = None,
            init_lr: float = 1e-2,
            num_iters: int = 20) -> np.ndarray:
        
        if init_theta is None:
            # init_theta = torch.rand(size=(self.env.theta_dim + 1, ), dtype=torch.float32) - 3
            # init_theta[2] = 0.0
            init_theta = torch.randn(self.env.theta_dim + 1, dtype=torch.float32) * 0.01
            init_theta[0] = 1.0
            log_theta_r = np.log(self._auglag_find_feasible(expert_x, expert_u, init_theta[:-1], extra_info))
            init_theta[-1] = log_theta_r
            theta = nn.Parameter(data=init_theta)
        else:
            assert init_theta.shape[0] == self.env.theta_dim
            log_theta_r = np.log(self._auglag_find_feasible(expert_x, expert_u, init_theta, extra_info))
            theta = nn.Parameter(data=torch.cat([init_theta, torch.tensor([log_theta_r], dtype=torch.float32)]))

        # optimizer = optim.SGD([theta], lr=init_lr)
        optimizer = optim.LBFGS([theta], lr=0.9)
        # optimizer = optim.LBFGS([theta], lr=0.95)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
        
        if self.verbose:
            with torch.no_grad():
                curr_theta = np.copy(theta.numpy())
                curr_theta[-1] = np.exp(curr_theta[-1])
                self._training_thetas = [curr_theta]
                self._training_losses = list()

        theta_r = np.exp(log_theta_r)

        optimizer.zero_grad()
        loss = -self.env.log_likelihood(x=expert_x,
                                        u=expert_u,
                                        extra_info=extra_info, 
                                        theta=theta, 
                                        mu_r=np.inf)
        loss.backward()

        lam_r = 0.0
        mu_r = max(1.0e-2, min(1.0e2, 0.1*torch.mean(torch.abs(theta.grad))/abs(theta_r))).item()
        pbar = trange(num_iters, desc='Training')

        total_iterations = 0
        while abs(theta_r) > 1e-5:
            with torch.no_grad():
                prev_theta_r = theta_r.item()

            self._train_step(expert_x, expert_u, extra_info, theta, 
                             mu_r, lam_r, optimizer, pbar)
            lr_scheduler.step()

            with torch.no_grad():
                theta_r = np.exp(theta[-1].item())
                lam_r -= mu_r*theta_r
                if abs(theta_r) > 0.5*abs(prev_theta_r) and abs(theta_r) > 1e-5:
                    mu_r *= 10.0

            total_iterations += 1
            if total_iterations >= num_iters:
                print('Maximum number of augmented Lagrangian iterations reached.')
                break

        pbar.close()

        if self.verbose:
            training_info = {'loss_vals': np.array(self._training_losses),
                             'theta_vals': np.stack(self._training_thetas)}
        else:
            training_info = None

        with torch.no_grad():
            final_theta = np.copy(theta.numpy())
            final_theta[-1] = np.exp(final_theta[-1])

        return final_theta[:-1], final_theta[-1], training_info
