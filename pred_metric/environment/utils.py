import os
import json
import torch
import numpy as np

from typing import Dict, Union, Tuple, Any, Optional
from nuscenes.map_expansion import arcline_path_utils
from nuscenes.map_expansion.map_api import NuScenesMap

import sys
sys.path.append('./Trajectron-plus-plus/trajectron')
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron


def load_model(model_dir, env, ts=100) -> Tuple[Trajectron, Dict[str, Any]]:
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)

    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')

    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams


def closest_lane_state(global_state: np.ndarray, nusc_map: NuScenesMap):
    nearest_lane = nusc_map.get_closest_lane(x=global_state[0], y=global_state[1])
    lane_rec = nusc_map.get_arcline_path(nearest_lane)
    closest_state, _ = arcline_path_utils.project_pose_to_lane(global_state, lane_rec)
    return closest_state


def lane_frenet_features(ego_state: np.ndarray, lane_states: np.ndarray):
    """Taking the equation from the "Line defined by two points" section of
    https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    as well as this answer: https://stackoverflow.com/a/6853926
    """
    x1s = lane_states[:-1, 0]
    y1s = lane_states[:-1, 1]
    h1s = lane_states[:-1, 2]

    x2s = lane_states[1:, 0]
    y2s = lane_states[1:, 1]
    h2s = lane_states[1:, 2]

    A = ego_state[0] - x1s
    B = ego_state[1] - y1s

    C = x2s - x1s
    D = y2s - y1s

    dot = A * C + B * D
    len_sq = C * C + D * D
    params = np.ma.masked_invalid(np.divide(dot, len_sq, out=np.full_like(dot, np.nan), where=np.abs(len_sq) >= 1e-3))

    if (params < 0).all():
        seg_idx = np.argmax(params)
        lane_x = x1s[seg_idx]
        lane_y = y1s[seg_idx]
        lane_h = h1s[seg_idx]
    elif (params > 1).all():
        seg_idx = np.argmin(params)
        lane_x = x2s[seg_idx]
        lane_y = y2s[seg_idx]
        lane_h = h2s[seg_idx]
    else:
        seg_idx = np.argmin(np.abs(params))
        lane_x = x1s[seg_idx] + params[seg_idx] * C[seg_idx]
        lane_y = y1s[seg_idx] + params[seg_idx] * D[seg_idx]
        lane_h = h1s[seg_idx] + params[seg_idx] * (h2s[seg_idx] - h1s[seg_idx])

    # plot_lane_frenet(lane_states, ego_state, np.array([xx, yy, hh]), seg_idx)
    return lane_x, lane_y, lane_h, seg_idx


def np_rbf(input: np.ndarray, center: Union[np.ndarray, float] = 0.0, scale: Union[np.ndarray, float] = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return np.exp(-0.5*np.square(input - center).sum(-1)/scale)


def pt_rbf(input: torch.Tensor, center: Union[torch.Tensor, float] = 0.0, scale: Union[torch.Tensor, float] = 1.0):
    """Assuming here that input is of shape (..., D), with center and scale of broadcastable shapes.
    """
    return torch.exp(-0.5*torch.square(input - center).sum(-1)/scale)


def wrap(angles: Union[torch.Tensor, np.ndarray]):
    return (angles + np.pi) % (2 * np.pi) - np.pi


def index_dict(d: Dict[str, Any], index: int, keepdim: bool = False):
    if d is None:
        return d
    
    if keepdim:
        return {k: v[[index]] for k, v in d.items()}
    else:
        return {k: (v[index] if isinstance(v, (torch.Tensor, list)) else v) for k, v in d.items()}


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    x = y = torch.linspace(-2, 2, 501)
    X, Y = torch.meshgrid(x, y)
    input = torch.stack((X, Y), dim=-1)
    reward_terms = torch.stack([pt_rbf(input, scale=0.1),
                                -pt_rbf(input, center=torch.tensor([0.4, 0.4]), scale=0.005)],
                               dim=-1)

    theta = torch.tensor([1.0, 1.0])

    combined_reward = reward_terms @ theta

    fig, ax = plt.subplots()
    Z = combined_reward # exp_reward(input, scale=0.1)
    cax = ax.contourf(X, Y, Z)
    fig.colorbar(cax)
    plt.show()

    
    # fig, ax = plt.subplots()
    # for scale in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
    #     y = exp_reward(x, scale=scale)
    #     ax.plot(x, y, label=scale)

    # ax.legend(loc='best')
    # plt.show()
