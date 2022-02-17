import torch
import torch.autograd.functional as AF

from functools import partial
from typing import Any, Dict, Callable


class CachingGradientComputer(object):
    def __init__(self, theta_dim: int) -> None:
        self.theta_dim: int = theta_dim

        # Jacobian
        self.memoized_g_hats: Dict[int, torch.Tensor] = dict()
        self.memoized_g_tildes: Dict[int, torch.Tensor] = dict()

        # Hessian
        self.memoized_H_hats: Dict[int, torch.Tensor] = dict()
        self.memoized_H_tildes: Dict[int, torch.Tensor] = dict()

    def cached_jacobian(self,
                        indexed_reward_function: Callable,
                        x_proper: torch.Tensor,
                        u_proper: torch.Tensor, 
                        theta: torch.Tensor, 
                        extra_info: Dict[str, Any]):

        batch_idx = extra_info['scene_idxs'].item()
        
        if batch_idx in self.memoized_g_hats:
            return self.memoized_g_hats[batch_idx] @ theta, self.memoized_g_tildes[batch_idx] @ theta

        g_hats = list()
        g_tildes = list()
        for theta_idx in range(self.theta_dim):
            g_hat, g_tilde = AF.jacobian(partial(indexed_reward_function, 
                                                 term_idx=theta_idx, 
                                                 extra_info=extra_info), 
                                         (x_proper, u_proper),
                                         create_graph=False,
                                         vectorize=True)

            g_hats.append(g_hat[0])
            g_tildes.append(g_tilde[0])

        self.memoized_g_hats[batch_idx] = torch.stack(g_hats, dim=-1)
        self.memoized_g_tildes[batch_idx] = torch.stack(g_tildes, dim=-1)

        return self.memoized_g_hats[batch_idx] @ theta, self.memoized_g_tildes[batch_idx] @ theta

    def cached_hessian(self,
                       indexed_reward_function: Callable,
                       x_proper: torch.Tensor,
                       u_proper: torch.Tensor, 
                       theta: torch.Tensor, 
                       extra_info: Dict[str, Any]):

        batch_idx = extra_info['scene_idxs'].item()
        
        if batch_idx in self.memoized_H_hats:
            return self.memoized_H_hats[batch_idx] @ theta, self.memoized_H_tildes[batch_idx] @ theta

        H_hats = list()
        H_tildes = list()
        for theta_idx in range(self.theta_dim):
            (H_hat, _), (_, H_tilde) = AF.hessian(partial(indexed_reward_function, 
                                                          term_idx=theta_idx, 
                                                          extra_info=extra_info), 
                                                  (x_proper, u_proper),
                                                  create_graph=False,
                                                  vectorize=True)

            H_hats.append(H_hat)
            H_tildes.append(H_tilde)

        self.memoized_H_hats[batch_idx] = torch.stack(H_hats, dim=-1)
        self.memoized_H_tildes[batch_idx] = torch.stack(H_tildes, dim=-1)

        return self.memoized_H_hats[batch_idx] @ theta, self.memoized_H_tildes[batch_idx] @ theta


if __name__ == '__main__':
    pass