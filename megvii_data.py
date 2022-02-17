import os
import sys
import dill
import torch
import torch.autograd.functional as AF
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.transforms import Affine2D
import moviepy.editor as mpy

from tqdm import tqdm, trange
from typing import List, Any
from pathlib import Path
from functools import partial
from pred_metric.environment.nuScenes_data import analysis
from pred_metric.visualization import *
from collections import defaultdict

from nuscenes.map_expansion.map_api import NuScenesMap

import sys
sys.path.append('./Trajectron-plus-plus/trajectron')
from environment import Environment, Scene, Node

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--seed', type=int, default=12,
                    help='random seed value')

# We use vectorize=True for autograd.functional.{jacobian,hessian} calls
# and see great performance improvements. Leaving this here in case some
# debugging is needed.
# torch._C._debug_only_display_vmap_fallback_warnings(True)
# torch.autograd.set_detect_anomaly(True)


def get_det_sensitivities(theta: torch.Tensor, 
                          ego_x: torch.Tensor, # Just the current ego state (self.state_dim, )
                          det_mus: torch.Tensor,
                          extra_info: Dict[str, torch.Tensor]) -> torch.Tensor:

    def detection_reward_term(det_mus_vec: torch.Tensor, ego_x: torch.Tensor, 
                              theta: torch.Tensor, extra_info: Dict[str, Any]):
        det_mus = det_mus_vec.reshape(extra_info['det_mus_shape'])

        ego_obs_delta = ego_x[..., :2] - det_mus
        closest_obs_idx = torch.argmin(torch.linalg.norm(ego_obs_delta, dim=-1))
        return -theta[3] * utils.pt_rbf(ego_obs_delta[closest_obs_idx], scale=2).sum(-1)

    extra_info['det_mus_shape'] = (det_mus.shape[-1], )

    det_sens_mags = torch.zeros((det_mus.shape[0], ))
    for det_idx in range(det_mus.shape[0]):
        g_mus = AF.jacobian(partial(detection_reward_term, 
                                    ego_x=ego_x,
                                    theta=theta, 
                                    extra_info=extra_info), 
                            det_mus[det_idx].flatten(),
                            create_graph=False,
                            vectorize=True)

        det_sens_mags[det_idx] = torch.linalg.norm(g_mus)
    
    return det_sens_mags


def main(args, plot=True):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)    
    
    ###############
    # Evaluations #
    ###############
    learned_theta = np.array([1.13, 0.551, 0., 12.176, 1.359]) # with the new mu_r initialization (old one it was just 1.0), num_trajs 256, L-BFGS lr=0.95
    
    # with open(f'plots/megvii/dets_and_sens_combined/all_sens.csv', 'w') as f:
    #     f.write(f'ep,agent,det_sens\n')

    output_path = 'pred_metric/environment/nuScenes_data/cached_data/megvii'

    dets_map = dict()
    for ep in trange(6019):
        with open(f'pred_metric/environment/nuScenes_data/cached_data/megvii/megvii_gt_{ep}.pkl', 'rb') as f:
            gt_env: Environment = dill.load(f, encoding='latin1')

        gt_scene: Scene = gt_env.scenes[0]
        gt_node_states = np.concatenate([node.get(np.array([0]), {'position': ['x', 'y'], 'heading': ['°']}) for node in gt_scene.nodes], axis=0)

        det_sens = get_det_sensitivities(torch.from_numpy(learned_theta), 
                                         torch.from_numpy(gt_node_states[0, :2]),
                                         torch.from_numpy(gt_node_states[1:, :2]),
                                         dict())

        for idx, node in enumerate(gt_scene.nodes[1:]):
            dets_map[(gt_scene.name, node.id)] = det_sens[idx].item()

    data_dict_path = os.path.join(output_path, f'megvii_gt_dets.pkl')
    with open(data_dict_path, 'wb') as f:
        dill.dump(dets_map, f, protocol=4) # For Python 3.6 and 3.8 compatability.
    print('Saved GT Dets!')

    raise

    for ep in tqdm([3326, 3327, 3400, 3195, 14, 3403, 3423, 3399, 3395]): # trange(6019):
        with open(f'pred_metric/environment/nuScenes_data/cached_data/megvii/megvii_dets_{ep}.pkl', 'rb') as f:
            dets_env: Environment = dill.load(f, encoding='latin1')

        with open(f'pred_metric/environment/nuScenes_data/cached_data/megvii/megvii_gt_{ep}.pkl', 'rb') as f:
            gt_env: Environment = dill.load(f, encoding='latin1')

        Path(f'plots/megvii/dets_and_sens_combined').mkdir(parents=True, exist_ok=True)

        det_scene: Scene = dets_env.scenes[0]
        gt_scene: Scene = gt_env.scenes[0]

        det_node_states = np.concatenate([node.get(np.array([0]), {'position': ['x', 'y'], 'heading': ['°']}) for node in det_scene.nodes], axis=0)
        gt_node_states = np.concatenate([node.get(np.array([0]), {'position': ['x', 'y'], 'heading': ['°']}) for node in gt_scene.nodes], axis=0)

        det_sens = get_det_sensitivities(torch.from_numpy(learned_theta), 
                                         torch.from_numpy(det_node_states[0, :2]),
                                         torch.from_numpy(det_node_states[1:, :2]),
                                         dict())

        # with open(f'plots/megvii/dets_and_sens_combined/all_sens.csv', 'a') as f:
        #     for sen_idx in range(det_sens.shape[0]):
        #         f.write(f'{ep},{sen_idx},{det_sens[sen_idx].item()}\n')

        if plot:
            fig, ax = plot_map(dataroot='pred_metric/environment/nuScenes_data/raw',
                            map_name=det_scene.map_name,
                            map_patch=(gt_node_states[0, 0] - 12.5,
                                       gt_node_states[0, 1] - 12.5,
                                       gt_node_states[0, 0] + 12.5,
                                       gt_node_states[0, 1] + 12.5))

            for _artist in ax.lines + ax.collections + ax.patches + ax.images:
                _artist.set_label(s=None)

            gt_color = 'royalblue'
            dets_color = 'deeppink'
            ego_color = 'coral'
            sens_color = 'goldenrod'
            heading_ind = 'grey'

            sens_low_color = np.array(colors.to_rgb(dets_color))
            sens_high_color = np.array(colors.to_rgb(sens_color))
            
            # Sensitivity of the ego is zero, just for indexing purposes.
            scaled_det_sens = np.concatenate([np.zeros((1, )), torch.clamp(det_sens*1, max=1.0).numpy()], axis=0)
            det_sens_colors = interpolate_colors(scaled_det_sens[:, np.newaxis], sens_low_color, sens_high_color)

            point_of_rotation = np.array([det_scene.robot.length/2, det_scene.robot.width/2])
            rec = plt.Rectangle(det_node_states[0, :2]-point_of_rotation, width=det_scene.robot.length, height=det_scene.robot.width, 
                                color=ego_color, alpha=1.0,
                                transform=Affine2D().rotate_around(*(det_node_states[0, 0], det_node_states[0, 1]), det_node_states[0, 2])+ax.transData,
                                label='Ego Vehicle')
            ax.add_patch(rec)
            ax.plot([det_node_states[0, 0], det_node_states[0, 0] + det_scene.robot.length*np.cos(det_node_states[0, 2])/2],
                    [det_node_states[0, 1], det_node_states[0, 1] + det_scene.robot.length*np.sin(det_node_states[0, 2])/2], 
                    color=heading_ind,
                    lw=det_scene.robot.width*3)

            # ax.scatter(det_node_states[1:, 0], det_node_states[1:, 1], label='Detections', color='blue')
            for idx, node in enumerate(det_scene.nodes):
                if node.is_robot:
                    continue

                point_of_rotation = np.array([node.width/2, node.length/2])
                rec = plt.Rectangle(det_node_states[idx, :2]-point_of_rotation, width=node.width, height=node.length, 
                                    color=det_sens_colors[idx], alpha=0.4,
                                    transform=Affine2D().rotate_around(*(det_node_states[idx, 0], det_node_states[idx, 1]), det_node_states[idx, 2])+ax.transData,
                                    label='Detection' if idx == 1 else None)
                ax.add_patch(rec)
                ax.plot([det_node_states[idx, 0], det_node_states[idx, 0] + node.width*np.cos(det_node_states[idx, 2])/2],
                        [det_node_states[idx, 1], det_node_states[idx, 1] + node.width*np.sin(det_node_states[idx, 2])/2], 
                        color=heading_ind,
                        lw=min(node.length*3, 10))

            # ax.scatter(gt_node_states[1:, 0], gt_node_states[1:, 1], label='GT', color='green')
            for idx, node in enumerate(gt_scene.nodes):
                if node.is_robot:
                    continue

                point_of_rotation = np.array([node.width/2, node.length/2])
                rec = plt.Rectangle(gt_node_states[idx, :2]-point_of_rotation, width=node.width, height=node.length, 
                                    color=gt_color, alpha=0.4,
                                    transform=Affine2D().rotate_around(*(gt_node_states[idx, 0], gt_node_states[idx, 1]), gt_node_states[idx, 2])+ax.transData,
                                    label='GT' if idx == 1 else None)
                ax.add_patch(rec)
                ax.plot([gt_node_states[idx, 0], gt_node_states[idx, 0] + node.width*np.cos(gt_node_states[idx, 2])/2],
                        [gt_node_states[idx, 1], gt_node_states[idx, 1] + node.width*np.sin(gt_node_states[idx, 2])/2], 
                        color=heading_ind,
                        lw=min(node.length*3, 10))

            ax.legend(loc='best', prop={"size": 40})
            ax.grid(False)
            fig.tight_layout()
            fig.savefig(f'plots/megvii/dets_and_sens_combined/ep_{ep}.pdf', dpi=200)
            plt.close(fig)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
