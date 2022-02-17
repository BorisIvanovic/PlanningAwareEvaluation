import os
import dill
import torch
import numpy as np
import cvxpy as cp
import torch.autograd.functional as AF
import matplotlib.pyplot as plt

from typing import Dict, Optional, Union, Any
from tqdm import tqdm
from pathlib import Path
from functools import partial
from scipy.optimize import minimize

from pred_metric.environment import UnicycleEnvironment
from pred_metric.environment import utils
from pred_metric.visualization import plot_map
from pred_metric.gradients import CachingGradientComputer

from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion import arcline_path_utils

# Need this to properly load the dataset (pickle needs the classes 
# in Trajectron++ to inflate them with the saved data)
import sys
sys.path.append('./Trajectron-plus-plus/trajectron')
from model.trajectron import Trajectron
from model.components import GMM2D
from environment import Environment, Scene, Node

pos_dict = {'position': ['x', 'y']}
velocity_dict = {'velocity': ['x', 'y']}
unicycle_state_dict = {'position': ['x', 'y'], 'heading': ['°']}
unicycle_control_dict = {'heading': ['d°'], 'acceleration': ['x', 'y']}


class NuScenesEnvironment(UnicycleEnvironment):
    def __init__(self, env_name: str, num_timesteps: int, gt_theta: np.ndarray, data_path: str, load_nusc: bool = True) -> None:
        super().__init__(env_name=env_name, state_dim=4, control_dim=2, num_timesteps=num_timesteps, dt=0.5, gt_theta=gt_theta)

        self.data_gradients = CachingGradientComputer(self.theta_dim)
        
        if load_nusc:
            print('Loading', data_path)
            with open(data_path, 'rb') as f:
                self.env: Environment = dill.load(f, encoding='latin1')

            self.data_version: str = 'v1.0-mini' if 'mini' in data_path else 'v1.0-trainval'
            self.data_root: str = os.path.join(Path(__file__).parent, 'raw')
            self.nusc: NuScenes = NuScenes(version=self.data_version, dataroot=self.data_root)
            self.helper: PredictHelper = PredictHelper(self.nusc)

    def calc_A(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        #           0  1     2     3
        # State is [x, y, heading, v]
        #             0    1
        # Control is [v, dphi]
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        one = torch.tensor(1)
        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.state_dim]),
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

        for i in range(self.state_dim):
            F[..., i, i] = one

        F[..., 0, 2] = v * dcos_domega - (a / dphi) * dsin_domega + (a / dphi) * torch.cos(phi_p_omega_dt) * self.dt
        F[..., 0, 3] = dsin_domega

        F[..., 1, 2] = v * dsin_domega + (a / dphi) * dcos_domega + (a / dphi) * torch.sin(phi_p_omega_dt) * self.dt
        F[..., 1, 3] = -dcos_domega

        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.state_dim]),
                           dtype=torch.float32)

        for i in range(self.state_dim):
            F_sm[..., i, i] = one

        F_sm[..., 0, 2] = -v * torch.sin(phi) * self.dt - (a * torch.sin(phi) * self.dt ** 2) / 2
        F_sm[..., 0, 3] = torch.cos(phi) * self.dt

        F_sm[..., 1, 2] = v * torch.cos(phi) * self.dt + (a * torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 3] = torch.sin(phi) * self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def calc_B(self, x, u, extra_info: Optional[Dict[str, torch.Tensor]] = None):
        #           0  1     2     3
        # State is [x, y, heading, v]
        #             0    1
        # Control is [v, dphi]
        x_t, u_t = self.unvec_xu(x, u, extra_info)

        # Using u_t for shape because it has the correct # of timesteps.
        F = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.control_dim]),
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

        F_sm = torch.zeros(u_t.shape[:-1] + torch.Size([self.state_dim, self.control_dim]),
                           dtype=torch.float32)

        F_sm[..., 0, 1] = (torch.cos(phi) * self.dt ** 2) / 2
        F_sm[..., 1, 1] = (torch.sin(phi) * self.dt ** 2) / 2

        F_sm[..., 3, 1] = self.dt

        return torch.where(~mask.unsqueeze(-1).unsqueeze(-1), F, F_sm)

    def get_closest_obs(self, ego_state: np.ndarray, scene: Scene, timestep: int):
        scene_offset = np.array([scene.x_min, scene.y_min])

        timestep_np = np.array([timestep])
        present_nodes = scene.present_nodes(timestep_np, return_robot=False)

        if len(present_nodes) > 0:
            node_poses = scene_offset + np.concatenate([node.get(timestep_np, pos_dict) for node in present_nodes[timestep]],
                                                    axis=0)

            distances = np.linalg.norm(ego_state[:2] - node_poses, axis=-1)
            min_idx = np.argmin(distances)
            return node_poses[min_idx]
        else:
            return np.full_like(ego_state[:2], fill_value=np.inf)

    def get_obs_preds(self, 
                      scene: Scene, 
                      timestep: int, 
                      pred_model: Trajectron,
                      pred_horizon: int):
        timestep_np = np.array([timestep])
        present_nodes = scene.present_nodes(timestep_np, return_robot=False)
        if len(present_nodes) > 0:
            with torch.no_grad():
                obs_pred_dists, _ = pred_model.predict(scene, timestep_np, pred_horizon, output_dists=True)
            
            gmm_dict = obs_pred_dists[timestep]
            del gmm_dict[scene.robot]

            probabilities = torch.stack([gmm.pis_cat_dist.probs.squeeze()[0]
                                            for gmm in gmm_dict.values()])

            return gmm_dict, probabilities
        else:
            return dict()

    def get_observation(self, 
                        scene: Scene, 
                        t: int,
                        nusc_map: NuScenesMap, 
                        last_ego_state: np.ndarray, 
                        pred_model: Trajectron,
                        pred_horizon: int,
                        return_features: bool = True):
        scene_offset = np.array([scene.x_min, scene.y_min, 0])

        ego_state = scene.robot.get(np.array([t]), unicycle_state_dict)[0]
        ego_state_global = ego_state + scene_offset

        nearest_lane = nusc_map.get_closest_lane(x=ego_state_global[0], y=ego_state_global[1])
        lane_rec = nusc_map.get_arcline_path(nearest_lane)
        poses = np.asarray(arcline_path_utils.discretize_lane(lane_rec, resolution_meters=0.5))

        lane_x, lane_y, lane_h, seg_idx = utils.lane_frenet_features(ego_state_global, poses)

        projected_goal_state = utils.closest_lane_state(last_ego_state + scene_offset, nusc_map)
        goal_x, goal_y, goal_h = projected_goal_state

        ego_vel = scene.robot.get(np.array([t]), velocity_dict)[0]
        ego_vel_norm = np.linalg.norm(ego_vel)

        obs = np.array([ego_state_global[0], ego_state_global[1], ego_state_global[2], ego_vel_norm])
            
        obs_x, obs_y = self.get_closest_obs(obs, scene, t)
        predictions, probabilities = self.get_obs_preds(scene, t, pred_model, pred_horizon)

        features = np.array([ego_state_global[0], ego_state_global[1], ego_state_global[2], ego_vel_norm, # 3
                             lane_x, lane_y, lane_h, # 6
                             goal_x, goal_y, # 8
                             obs_x, obs_y, # 10
                             scene_offset[0], scene_offset[1], # 12
                             scene.dt]) # 13

        if return_features:
            return obs, features, predictions, probabilities
        else:
            return obs

    def get_actions(self, scene: Scene, ts: np.ndarray):
        ego_u = scene.robot.get(ts, unicycle_control_dict)
        return np.stack([ego_u[..., 0], np.linalg.norm(ego_u[..., 1:], axis=-1)], axis=-1) # delta heading, a_norm

    def cvx_step_reward_fn(self, x_t, u_t, theta, features):
        # x_t is x, y, heading, v
        # u_t is delta_heading, a
        ego_lane_lat_delta = x_t[:2] - features[4:6]
        ego_lane_heading_delta = x_t[2] - features[6]
        ego_goal_delta = x_t[:2] - features[7:9]

        # Ignoring the non-convex terms for the first (convex) pass
        return (-theta[0] * cp.sum_squares(ego_lane_lat_delta)
                -theta[1] * cp.square(ego_lane_heading_delta)
                -theta[2] * cp.sum_squares(ego_goal_delta)
                -theta[4] * cp.sum_squares(u_t))

    @staticmethod
    def pred_coll_reward_term(x, u, extra_info):
        ep_len = extra_info['ep_lengths'] + 1 # Need +1 because this is the # of controls, and state has one more.
        features = extra_info['features'][:ep_len]
        dt = features[0, -1] # dt is the same for all timesteps, so just taking it from the first one.

        scene_offset = features[..., 11:13]
        pred_horizon = extra_info['pred_horizon']
        ep_len = extra_info['ep_lengths']
        gmm_dict: Dict[int, Dict[Node, GMM2D]] = extra_info['predictions']

        # Can compute these in a faster way than double-looping since they all have the same size
        if ep_len > pred_horizon:
            cons_size_ol_ego_poses = [x[0, :ep_len - pred_horizon]]
            for ph in range(pred_horizon):
                ego_action = u[0, ph : ph + ep_len - pred_horizon]
                cons_size_ol_ego_poses.append(NuScenesEnvironment.extended_unicycle_dyn_fn(cons_size_ol_ego_poses[-1], ego_action, dt, 
                                                                                           ret_np=False, ego_pred_type=extra_info['ego_pred_type']))

            cons_size_ol_ego_poses = torch.stack(cons_size_ol_ego_poses[1:], dim=1)[..., :2]
        else:
            cons_size_ol_ego_poses = torch.tensor([])

        if extra_info['use_pred_uncertainty']:
            # TODO: Fix this!
            # Maximum log_prob of T++ (should be better than just closest and circular distance,
            # example is someone turning left into the lane left of you and the car immediately to 
            # the right of you being closer (but less dangerous because stopped)).
            max_obs_log_probs = list()
            for timestep in range(ep_len):
                if timestep >= cons_size_ol_ego_poses.shape[0]:
                    # Only go down this slow path if it's not computed above ...
                    curr_ol_ego_poses = [x[0, timestep]]
                    ego_actions = u[0, timestep : timestep + pred_horizon]
                    for ph in range(0, ego_actions.shape[0]):
                        curr_ol_ego_poses.append(NuScenesEnvironment.extended_unicycle_dyn_fn(curr_ol_ego_poses[-1], ego_actions[ph], dt,
                                                                                              ret_np=False, ego_pred_type=extra_info['ego_pred_type']))
                    for ph in range(ego_actions.shape[0], pred_horizon):
                        curr_ol_ego_poses.append(torch.full_like(curr_ol_ego_poses[-1], np.nan, requires_grad=False))
                    
                    # Only focusing on open-loop future positions (hence the [1:] on the list and [..., :2] on the stacked tensor)
                    curr_ol_ego_poses = torch.stack(curr_ol_ego_poses[1:])[..., :2]
                    effective_ph = ego_actions.shape[0]

                else:
                    # ... otherwise we can use the quickly-computed values from above.
                    curr_ol_ego_poses = cons_size_ol_ego_poses[timestep]
                    effective_ph = pred_horizon
                
                # predictions.shape is (num_agents, pred_horizon, num_GMM_components, 2 (for x, y))
                # Indexing predictions like this to handle end-of-episode cases where we do not have as much future control data.
                pred_coll_probs = torch.stack([gmm.log_prob(curr_ol_ego_poses - scene_offset[timestep]).squeeze() 
                                                for gmm in gmm_dict[timestep].values()])

                # Maximum collision prob over agents and time -> closest encounter with another agent in the next pred_horizon timesteps.
                # Negating it since we want less collision probability.
                max_obs_log_probs.append(torch.amax(pred_coll_probs[:, :effective_ph]))

            max_obs_log_probs = torch.stack(max_obs_log_probs).unsqueeze(0)
            return -torch.sigmoid(max_obs_log_probs + 1).sum(-1)
        
        else:
            min_obs_pred_dists = list()
            for timestep in range(ep_len):
                if timestep >= cons_size_ol_ego_poses.shape[0]:
                    # Only go down this slow path if it's not computed above ...
                    curr_ol_ego_poses = [x[0, timestep]]
                    ego_actions = u[0, timestep : timestep + pred_horizon]
                    for ph in range(ego_actions.shape[0]):
                        curr_ol_ego_poses.append(NuScenesEnvironment.extended_unicycle_dyn_fn(curr_ol_ego_poses[-1], ego_actions[ph], dt,
                                                                                              ret_np=False, ego_pred_type=extra_info['ego_pred_type']))
                    
                    # Only focusing on open-loop future positions (hence the [1:] on the list and [..., :2] on the stacked tensor)
                    curr_ol_ego_poses = torch.stack(curr_ol_ego_poses[1:])[..., :2].unsqueeze(0).unsqueeze(2)
                    effective_ph = ego_actions.shape[0]

                else:
                    # ... otherwise we can use the quickly-computed values from above.
                    curr_ol_ego_poses = cons_size_ol_ego_poses[timestep].unsqueeze(0).unsqueeze(2)
                    effective_ph = pred_horizon
                
                # predictions.shape is (num_agents, pred_horizon, num_GMM_components, 2 (for x, y))
                # Indexing predictions like this to handle end-of-episode cases where we do not have as much future control data.
                if 'training' in extra_info and extra_info['training']:
                    predictions = torch.stack([torch.from_numpy(node.get(np.array([extra_info['init_timestep'] + timestep + 1, extra_info['init_timestep'] + timestep + pred_horizon]), pos_dict)[:effective_ph]) + scene_offset[timestep] for node in gmm_dict[timestep]]).unsqueeze(2)
                    probs = torch.ones((predictions.shape[0], 1))
                else:
                    predictions = torch.stack([gmm.mus.squeeze()[:effective_ph] + scene_offset[timestep] for gmm in gmm_dict[timestep].values()])
                    probs = extra_info['probabilities'][timestep]

                comp_dists = torch.linalg.norm(curr_ol_ego_poses - torch.nan_to_num(predictions, nan=1e6), dim=-1) # Distance from predictions
                min_comps_dist = torch.amin(comp_dists, dim=1) # Minimum over time -> closest encounter
                expected_min_dists = torch.sum(min_comps_dist * probs, dim=-1) # Factoring in traj. probabilities -> expected closest encounter
                min_obs_pred_dist = torch.amin(expected_min_dists) # Closest we expect the ego will get to another agent in the next pred_horizon timesteps.
                min_obs_pred_dists.append(min_obs_pred_dist)

            min_obs_pred_dists = torch.stack(min_obs_pred_dists).unsqueeze(0).unsqueeze(2)
            return -utils.pt_rbf(min_obs_pred_dists, scale=2).sum(-1)

    def indexed_reward_fn(self, x_vec: torch.Tensor, u_vec: torch.Tensor, term_idx: int, extra_info: Dict[str, torch.Tensor]):
        x, u = self.unvec_xu(x_vec, u_vec, extra_info)
        
        ep_len = extra_info['ep_lengths'] + 1 # Need +1 because this is the # of controls, and state has one more.
        features = extra_info['features'][:ep_len]

        if term_idx == 0: # Frenet lateral distance to the nearest lane
            ego_lane_lat_delta = x[..., :2] - features[..., 4:6]
            return -torch.square(ego_lane_lat_delta).sum((-1, -2))

        elif term_idx == 1: # Smallest distance to the nearest lane's current heading
            ego_lane_heading_delta = utils.wrap(x[..., 2] - features[..., 6])
            return -torch.square(ego_lane_heading_delta).sum(-1)

        elif term_idx == 2: # Distance from the ego to its goal (its last position)
            ego_goal_delta = x[..., :2] - features[..., 7:9]
            return -torch.square(ego_goal_delta).sum((-1, -2))

        elif term_idx == 3: # Distance to other agents
            ego_obs_delta = x[..., :2] - features[..., 9:11]
            return -utils.pt_rbf(ego_obs_delta, scale=2).sum(-1)

        elif term_idx == 4: # Control effort
            return -torch.square(u).sum((-1, -2))

        elif term_idx == 5: # Predictive collision avoidance term
            return self.pred_coll_reward_term(x, u, extra_info)
            
        else:
            raise ValueError(f'term_idx {term_idx} is > {self.theta_dim}')

    def gt_reward_fn(self, x_vec: torch.Tensor, u_vec: torch.Tensor, theta: torch.Tensor, extra_info: Dict[str, torch.Tensor]):
        reward_terms = torch.stack([self.indexed_reward_fn(x_vec, u_vec, i, extra_info) 
                                        for i in range(self.theta_dim)], dim=-1)
        
        combined_reward = (reward_terms @ theta)
        
        # The mean is over the batch
        return torch.mean(combined_reward, dim=-1)

    def _run_single_jacobian(self, 
                             x_proper: torch.Tensor, 
                             u_proper: torch.Tensor, 
                             theta: torch.Tensor,
                             extra_info: Dict[str, torch.Tensor]):
        return self.data_gradients.cached_jacobian(self.indexed_reward_fn, 
                                                   x_proper, u_proper, 
                                                   theta=theta[:-1], 
                                                   extra_info=extra_info)

    def _run_single_hessian(self, 
                            x: torch.Tensor, 
                            u: torch.Tensor, 
                            theta: torch.Tensor,
                            extra_info: Dict[str, torch.Tensor], 
                            J: Optional[torch.Tensor] = None):
        x_proper, u_proper = self._ensure_length(x, u, extra_info)

        H_hat, H_tilde = self.data_gradients.cached_hessian(self.indexed_reward_fn, 
                                                            x_proper, u_proper, 
                                                            theta=theta[:-1], 
                                                            extra_info=extra_info)

        if J is None:
            J = self._form_J(self.calc_A(x, u, extra_info)[0], self.calc_B(x, u, extra_info)[0, :-1], extra_info, print=False)

        H = H_tilde + J @ H_hat @ torch.transpose(J, -1, -2)
        
        # Adding the H regularization term here
        return H - torch.exp(theta[-1]) * torch.eye(J.shape[0])

    def _convex_optimize(self, theta: np.ndarray, extra_info: Dict[str, Any], verbose: bool = False):
        T = extra_info['ep_lengths'].item()
        x = cp.Variable((T+1, 4))
        u = cp.Variable((T, self.control_dim))

        features = extra_info['features'].numpy()
        dt = features[0, -1] # dt is the same for all timesteps, so just taking it from the first one.

        reward = 0
        constr = [x[0, :] == features[0, :4]]
        for t in range(T):
            reward += self.cvx_step_reward_fn(x[t, :], u[t, :], theta, features[t])

        # Last timestep
        reward += self.cvx_step_reward_fn(x[T, :], np.zeros((self.control_dim, )), theta, features[T])

        problem = cp.Problem(cp.Maximize(reward), constr)
        problem.solve(solver=cp.ECOS, verbose=verbose)

        x_soln, u_soln = x.value, u.value

        # Making the velocity and controls consistent with the resulting positions/heading.
        v_xy = np.gradient(x_soln[:, :2], dt, axis=0)
        vx, vy = v_xy[:, 0], v_xy[:, 1]

        a_xy = np.gradient(v_xy, dt, axis=0)
        ax, ay = a_xy[:, 0], a_xy[:, 1]

        v_norm = np.linalg.norm(v_xy, axis=-1)
        a_norm = np.divide(ax*vx + ay*vy, v_norm, out=np.zeros_like(ax), where=(v_norm > 1.))

        x_soln[1:, 3] = v_norm[1:]
        u_soln[:, 0] = np.gradient(np.unwrap(x_soln[:-1, 2] + np.pi), dt)
        u_soln[:, 1] = a_norm[:-1]

        return x_soln, u_soln

    @staticmethod
    def _noncvx_cost_fn(xu: np.ndarray, 
                        theta: np.ndarray, 
                        x_size: int, 
                        x_dims: int, 
                        u_dims: int, 
                        features: np.ndarray, 
                        extra_info: Dict[str, Any]):
        x, u = xu[:x_size], xu[x_size:]
        x_reshaped, u_reshaped = x.reshape((-1, x_dims)), u.reshape((-1, u_dims))

        ego_lane_lat_delta = x_reshaped[..., :2] - features[..., 4:6]
        ego_lane_heading_delta = x_reshaped[..., 2] - features[..., 6]
        ego_goal_delta = x_reshaped[..., :2] - features[..., 7:9]
        ego_obs_delta = x_reshaped[..., :2] - features[..., 9:11]

        with torch.no_grad():
            prediction_reward_term = NuScenesEnvironment.pred_coll_reward_term(torch.from_numpy(x_reshaped).unsqueeze(0),
                                                                               torch.from_numpy(u_reshaped).unsqueeze(0),
                                                                               extra_info).numpy()

        reward_terms = np.stack([np.square(ego_lane_lat_delta).sum(),
                                 np.square(ego_lane_heading_delta).sum(),
                                 np.square(ego_goal_delta).sum(),
                                 utils.np_rbf(ego_obs_delta, scale=2).sum(),
                                 np.square(u_reshaped).sum(),
                                 prediction_reward_term.sum()],
                                axis=-1)

        combined_reward = (reward_terms @ theta)
        
        return np.mean(combined_reward)

    @staticmethod
    def extended_unicycle_dyn_fn(x: Union[torch.Tensor, np.ndarray],
                                 u: Union[torch.Tensor, np.ndarray], 
                                 dt: float, 
                                 ret_np: bool,
                                 ego_pred_type: str = 'motion_plan'):
        x_p = torch.as_tensor(x[..., 0])
        y_p = torch.as_tensor(x[..., 1])
        phi = torch.as_tensor(x[..., 2])
        v = torch.as_tensor(x[..., 3])
        dphi = torch.as_tensor(u[..., 0])
        a = torch.as_tensor(u[..., 1])

        if ego_pred_type == 'const_vel':
            return torch.stack([x_p + v * torch.cos(phi) * dt,
                                y_p + v * torch.sin(phi) * dt,
                                phi * torch.ones_like(a),
                                v], dim=-1)

        mask = torch.abs(dphi) <= 1e-2
        dphi = ~mask * dphi + mask * 1

        phi_p_omega_dt = utils.wrap(phi + dphi * dt)
        dsin_domega = (torch.sin(phi_p_omega_dt) - torch.sin(phi)) / dphi
        dcos_domega = (torch.cos(phi_p_omega_dt) - torch.cos(phi)) / dphi

        d1 = torch.stack([(x_p
                           + (a / dphi) * dcos_domega
                           + v * dsin_domega
                           + (a / dphi) * torch.sin(phi_p_omega_dt) * dt),
                          (y_p
                           - v * dcos_domega
                           + (a / dphi) * dsin_domega
                           - (a / dphi) * torch.cos(phi_p_omega_dt) * dt),
                          phi_p_omega_dt,
                          v + a * dt], dim=-1)
        d2 = torch.stack([x_p + v * torch.cos(phi) * dt + (a / 2) * torch.cos(phi) * dt ** 2,
                          y_p + v * torch.sin(phi) * dt + (a / 2) * torch.sin(phi) * dt ** 2,
                          phi * torch.ones_like(a),
                          v + a * dt], dim=-1)

        next_states = torch.where(~mask.unsqueeze(-1), d1, d2)
        if ret_np:
            return next_states.numpy()
        else:
            return next_states

    def _nonconvex_optimize(self, 
                            theta: np.ndarray, 
                            extra_info: Dict[str, Any], 
                            init_traj: np.ndarray, 
                            init_controls: np.ndarray, 
                            verbose: bool = False):
        ep_len = extra_info['ep_lengths'] + 1
        features = extra_info['features'][:ep_len].numpy()
        
        flattened_traj = init_traj.flatten()
        flattened_controls = init_controls.flatten()
        x0 = np.concatenate([flattened_traj, flattened_controls])

        def init_constr(xu, features):
            return xu[:4] - features[0, :4]

        def dyn_constr(xu, dyn_fn, x_size, x_dims, u_dims, dt, t, ego_pred_type):
            x_t, u_t = xu[t*x_dims : (t+1)*x_dims], xu[x_size + t*u_dims : x_size + (t+1)*u_dims]
            data_x_tp1 = xu[(t+1)*x_dims : (t+2)*x_dims]
            dyn_x_tp1 = dyn_fn(x_t, u_t, dt, ret_np=True, ego_pred_type=ego_pred_type)
            state_diff = data_x_tp1 - dyn_x_tp1
            return np.array([state_diff[0], state_diff[1], utils.wrap(state_diff[2]), state_diff[3]])

        x_size = flattened_traj.shape[0]
        x_dims = init_traj.shape[-1]
        u_dims = init_controls.shape[-1]
        T = init_traj.shape[0]

        init_constraint = {'type': 'eq', 'fun': init_constr, 'args': (features, )}
        dynamics_constraints = [{'type': 'eq', 
                                 'fun': dyn_constr, 
                                 'args': (self.extended_unicycle_dyn_fn, x_size, x_dims, u_dims, self.dt, t, extra_info['ego_pred_type'])}
                                for t in range(T-1)]
        opt_result = minimize(self._noncvx_cost_fn, x0, 
                              args=(theta, x_size, x_dims, u_dims, features, extra_info),
                              method='SLSQP',
                              constraints=[init_constraint] + dynamics_constraints,
                              options={'disp': verbose})

        soln = opt_result.x
        soln_x, soln_u = soln[:x_size], soln[x_size:]
        return soln_x.reshape((-1, x_dims)), soln_u.reshape((-1, u_dims))

    def reoptimize_with_theta(self, 
                              theta: np.ndarray, 
                              scene: Scene, 
                              extra_info: Dict[str, Any], 
                              plot: bool = False, 
                              verbose: bool = False,
                              prefix: Optional[str] = None):
        # Two-step process, first get a good initial guess by maximizing the convex parts of the reward.
        # Then, run a non-linear optimizer (e.g., scipy.optimize) to get the final solution.
        init_traj, init_controls = self._convex_optimize(theta, extra_info, verbose)
        final_traj, final_controls = self._nonconvex_optimize(theta, extra_info, init_traj, init_controls, verbose)

        if plot:
            if verbose:
                print('Plotting...', end=' ', flush=True)

            # Need +1 because this is the # of controls, and state has one more.
            ep_len = extra_info['ep_lengths'] + 1
            features = extra_info['features'][:ep_len].numpy()

            fig, ax = plot_map(dataroot=self.data_root,
                               map_name=self.helper.get_map_name_from_sample_token(scene.name), 
                               map_patch=(features[:, 0].min() - 10,
                                          features[:, 1].min() - 10, 
                                          features[:, 0].max() + 10, 
                                          features[:, 1].max() + 10))

            for _artist in ax.lines + ax.collections + ax.patches + ax.images:
                _artist.set_label(s=None)

            ax.plot(features[:, 0], features[:, 1], label='GT')
            ax.plot(init_traj[:, 0], init_traj[:, 1], label='1st Opt. Step')
            ax.plot(final_traj[:, 0], final_traj[:, 1], label='2nd Opt. Step')

            # ax.axis('equal')
            ax.grid(False)
            ax.legend(loc='best', prop={'size': 40})

            fig.tight_layout()
            fig.savefig(f'plots/nuScenes/reoptimized/{prefix}_{self.env_name}_{scene.name}.pdf', dpi=300)
            plt.close(fig)

            if verbose:
                print('Done!', flush=True)

        return final_traj, final_controls

    def gen_expert_data(self, 
                        num_samples: int, 
                        prediction_model_dir: str,
                        prediction_model_ckpt: int,
                        test: bool = False, 
                        seed: Optional[int] = None, 
                        initial_timestep: int = 8,
                        pred_horizon: int = 4, 
                        use_pred_uncertainty: bool = False,
                        ego_pred_type: str = 'motion_plan'):
        fname = os.path.join(Path(__file__).parent, f'cached_data/{self.env_name}_{num_samples}_ph_{pred_horizon}.dill')
        if os.path.exists(fname) and not test:
            with open(fname, 'rb') as f:
                saved_info = dill.load(f, encoding='latin1')

            extra_info = {'ep_lengths': torch.tensor(saved_info['ep_lengths'], dtype=torch.int),
                          'scene_idxs': torch.tensor(saved_info['scene_idxs'], dtype=torch.int),
                          'features': torch.tensor(saved_info['features'], dtype=torch.float32),
                          'predictions': saved_info['predictions'],
                          'probabilities': saved_info['probabilities'],
                          'init_timestep': saved_info['init_timestep'],
                          'pred_horizon': saved_info['pred_horizon'],
                          'use_pred_uncertainty': use_pred_uncertainty,
                          'ego_pred_type': ego_pred_type}
            return torch.tensor(saved_info['expert_xs'], dtype=torch.float32), torch.tensor(saved_info['expert_us'], dtype=torch.float32), extra_info

        pred_model, pred_hyperparams = utils.load_model(prediction_model_dir, self.env, ts=prediction_model_ckpt)

        if seed is not None:
            np.random.seed(seed)

        trajs = list()
        feats = list()
        controls = list()
        ep_length = list()
        scene_idxs = list()
        prediction_dicts = list()
        prediction_probs = list()
        pbar = tqdm(total=num_samples, desc='Generating Data')
        collected = 0
        for scene in self.env.scenes:
            skip = False
            if scene.timesteps <= initial_timestep + 1:
                # Want to have at least two state vectors (=> at least 1 action vector)
                continue

            nusc_map = NuScenesMap(dataroot=self.data_root, map_name=self.helper.get_map_name_from_sample_token(scene.name))
            last_ego_state = scene.robot.get(np.array([scene.timesteps-1]), unicycle_state_dict)[0]

            visited_states = list()
            obs_features = list()
            enacted_controls = list()
            scene_pred_dicts = dict()
            scene_pred_probs = dict()
            for timestep in range(initial_timestep, initial_timestep + self.num_timesteps):
                action = self.get_actions(scene, np.array([timestep]))[0]
                try:
                    obs, features, predictions, probabilities = self.get_observation(scene, timestep, nusc_map, last_ego_state, pred_model,
                                                                                        pred_horizon=pred_horizon)
                except ValueError:
                    skip = True
                    break

                visited_states.append(obs)
                obs_features.append(features)
                enacted_controls.append(action)
                scene_pred_dicts[timestep - initial_timestep] = predictions
                scene_pred_probs[timestep - initial_timestep] = probabilities
                if timestep >= scene.timesteps - 1:
                    break

            if skip:
                continue
            
            visited_states_np = np.array(visited_states)
            obs_features_np = np.array(obs_features)
            enacted_controls_np = np.array(enacted_controls)

            to_add = self.num_timesteps - (timestep - initial_timestep)
            trajs.append(np.concatenate((visited_states_np, 
                                         np.full((to_add, self.state_dim), np.nan))))
            feats.append(np.concatenate((obs_features_np, 
                                         np.full((to_add, obs_features_np.shape[-1]), np.nan))))
            controls.append(np.concatenate((enacted_controls_np, 
                                            np.full((to_add, self.control_dim), np.nan))))

            ep_length.append(timestep - initial_timestep)
            scene_idxs.append(collected)
            prediction_dicts.append(scene_pred_dicts)
            prediction_probs.append(scene_pred_probs)

            collected += 1
            pbar.set_description(f'Ep Length: {timestep}')
            pbar.update()
            if collected >= num_samples:
                break

        expert_xs, expert_us = np.stack(trajs), np.stack(controls)
        expert_xs, expert_us = expert_xs.reshape(expert_xs.shape[0], -1), expert_us.reshape(expert_us.shape[0], -1)
        if not test:
            saved_info = {'expert_xs': expert_xs, 
                         'expert_us': expert_us, 
                         'ep_lengths': np.array(ep_length), 
                         'scene_idxs': np.array(scene_idxs), 
                         'features': np.stack(feats),
                         'predictions': prediction_dicts,
                         'probabilities': prediction_probs,
                         'init_timestep': initial_timestep,
                         'pred_horizon': pred_horizon}
            with open(fname, 'wb') as f:
                dill.dump(saved_info, f, protocol=4)  # For compatability between Python 3.6 and 3.8.

        extra_info = {'ep_lengths': torch.tensor(ep_length, dtype=torch.int), 
                      'scene_idxs': torch.tensor(scene_idxs, dtype=torch.int), 
                      'features': torch.tensor(feats, dtype=torch.float32),
                      'predictions': prediction_dicts,
                      'probabilities': prediction_probs,
                      'init_timestep': initial_timestep,
                      'pred_horizon': pred_horizon,
                      'use_pred_uncertainty': use_pred_uncertainty,
                      'ego_pred_type': ego_pred_type}
        return torch.tensor(expert_xs, dtype=torch.float32), torch.tensor(expert_us, dtype=torch.float32), extra_info


    def get_sensitivities(self, 
                          theta: torch.Tensor, 
                          ego_x: torch.Tensor, # Just the current ego state (self.state_dim, )
                          ego_u: torch.Tensor, # Planned actions (pred_horizon, self.control_dim)
                          pred_mus: torch.Tensor,
                          pred_probs: torch.Tensor,
                          extra_info: Dict[str, torch.Tensor]) -> torch.Tensor:

        def prediction_reward_term(pred_mus_vec: torch.Tensor, pred_probs_vec: torch.Tensor, 
                                   ego_x: torch.Tensor, ego_u: torch.Tensor,
                                   theta: torch.Tensor, extra_info: Dict[str, Any]):
            pred_mus = pred_mus_vec.reshape(extra_info['pred_mus_shape'])
            pred_probs = pred_probs_vec.reshape(extra_info['pred_probs_shape'])

            # Just implemented a simplified version here, taking in theta too.        
            pred_horizon = extra_info['pred_horizon']

            # Can compute these in a faster way than double-looping since they all have the same size
            cons_size_ol_ego_poses = [ego_x]
            for ph in range(pred_horizon):
                ego_action = ego_u[ph]
                if torch.isnan(ego_action).any():
                    break

                cons_size_ol_ego_poses.append(NuScenesEnvironment.extended_unicycle_dyn_fn(cons_size_ol_ego_poses[-1], ego_action, self.dt,
                                              ret_np=False, ego_pred_type=extra_info['ego_pred_type']))

            cons_size_ol_ego_poses = torch.stack(cons_size_ol_ego_poses[1:])[..., :2]

            if extra_info['use_pred_uncertainty']:
                raise NotImplementedError()
            else:                
                curr_ol_ego_poses = cons_size_ol_ego_poses.unsqueeze(0).unsqueeze(2)
                
                # predictions.shape is (num_agents, pred_horizon, num_GMM_components, 2 (for x, y))
                # Added the nan_to_num because NaNs can creep in here when agents don't have pred_horizon-length futures.
                comp_dists = torch.linalg.norm(curr_ol_ego_poses - torch.nan_to_num(pred_mus[:, :curr_ol_ego_poses.shape[1]], nan=1e6), dim=-1) # Distance from predictions
                min_comps_dist = torch.amin(comp_dists, dim=1) # Minimum over time -> closest encounter
                expected_min_dist = torch.sum(min_comps_dist * pred_probs, dim=-1) # Factoring in traj. probabilities -> expected closest encounter

                return -theta[5] * utils.pt_rbf(expected_min_dist, scale=2)

        extra_info['pred_mus_shape'] = (1, ) + pred_mus.shape[1:]
        extra_info['pred_probs_shape'] = (1, ) + pred_probs.shape[1:]

        mu_sensitivies = torch.zeros_like(pred_mus)
        prob_sensitivities = torch.zeros_like(pred_probs)
        for pred_idx in range(pred_mus.shape[0]):
            g_mus, g_probs = AF.jacobian(partial(prediction_reward_term, 
                                                 ego_x=ego_x,
                                                 ego_u=ego_u,
                                                 theta=theta, 
                                                 extra_info=extra_info), 
                                         (pred_mus[pred_idx].flatten(), pred_probs[pred_idx].flatten()),
                                         create_graph=False,
                                         vectorize=True)

            mu_sensitivies[pred_idx] = g_mus.reshape(extra_info['pred_mus_shape'])
            prob_sensitivities[pred_idx] = g_probs.reshape(extra_info['pred_probs_shape'])

        mu_sens_mags = torch.amax(torch.linalg.norm(mu_sensitivies, dim=-1), dim=-2)
        return mu_sens_mags

    def get_det_sensitivities(self, 
                              theta: torch.Tensor, 
                              ego_x: torch.Tensor, # Just the current ego state (self.state_dim, )
                              det_mus: torch.Tensor,
                              extra_info: Dict[str, torch.Tensor]) -> torch.Tensor:

        def detection_reward_term(det_mus_vec: torch.Tensor, ego_x: torch.Tensor, 
                                  theta: torch.Tensor, extra_info: Dict[str, Any]):
            det_mus = det_mus_vec.reshape(extra_info['det_mus_shape'])

            ego_obs_delta = ego_x[..., :2] - det_mus
            return -theta[3] * utils.pt_rbf(ego_obs_delta, scale=2).sum(-1)

        extra_info['det_mus_shape'] = det_mus.shape[1:]

        det_sens_mags = torch.zeros((det_mus.shape[0], det_mus.shape[-1]))
        for det_idx in range(det_mus.shape[0]):
            g_mus = AF.jacobian(partial(detection_reward_term, 
                                        ego_x=ego_x,
                                        theta=theta, 
                                        extra_info=extra_info), 
                                det_mus[det_idx].flatten(),
                                create_graph=False,
                                vectorize=True)

            det_sens_mags[det_idx] = g_mus
        return torch.linalg.norm(det_sens_mags, dim=-1)
