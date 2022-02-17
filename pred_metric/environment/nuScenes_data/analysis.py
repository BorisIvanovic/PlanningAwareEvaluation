import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

from typing import List, Dict, Any, Union, Optional
from torch.tensor import Tensor
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from moviepy.video.io.bindings import mplfig_to_npimage

from pred_metric.environment import utils, NuScenesEnvironment
from pred_metric.visualization import plot_map, interpolate_colors

# Need this to properly load the dataset (pickle needs the classes 
# in Trajectron++ to inflate them with the saved data)
import sys
sys.path.append('./Trajectron-plus-plus/trajectron')
from model.components import GMM2D
from environment import Node
from evaluation.evaluation import compute_ade_pt, compute_fde_pt, compute_nll_pt

pos_dict = {'position': ['x', 'y']}
velocity_dict = {'velocity': ['x', 'y']}
unicycle_state_dict = {'position': ['x', 'y'], 'heading': ['°']}
unicycle_control_dict = {'heading': ['d°'], 'acceleration': ['x', 'y']}


def get_sensitivities_at_time(env: NuScenesEnvironment, 
                              learned_theta: np.ndarray,
                              expert_x: torch.Tensor,
                              expert_u: torch.Tensor, 
                              extra_info: Dict[str, Any],
                              ep: int, 
                              scene_t: int):
    indexed_extra_info = utils.index_dict(extra_info, ep)
    init_timestep = indexed_extra_info['init_timestep']
    pred_horizon = indexed_extra_info['pred_horizon']

    indexed_extra_info['features'] = indexed_extra_info['features'][scene_t - init_timestep]
    for k, v in indexed_extra_info.items():
        if isinstance(v, dict):
            indexed_extra_info[k] = v[scene_t - init_timestep]

    ego_x = expert_x[ep].reshape((-1, env.state_dim))
    ego_u = expert_u[ep].reshape((-1, env.control_dim))

    gmm_dict: Dict[Node, GMM2D] = indexed_extra_info['predictions']
    features = indexed_extra_info['features']
    scene_offset = features[11:13]

    prediction_gmms = gmm_dict.values()
    pred_mus = torch.stack([gmm.mus.squeeze() + scene_offset for gmm in prediction_gmms])
    pred_probs = torch.stack([gmm.pis_cat_dist.probs.squeeze()[0] for gmm in prediction_gmms])

    prediction_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        pred_mus, pred_probs,
                                                        indexed_extra_info)

    # Getting prediction_horizon + current to make plotting easier later.
    gt_mus = torch.stack(
        [torch.from_numpy(node.get(np.array([scene_t, scene_t + pred_horizon]), pos_dict)) + scene_offset 
            for node in gmm_dict]
    )

    gt_future_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        gt_mus[:, 1:].unsqueeze(-2), torch.ones((gt_mus.shape[0], 1)),
                                                        indexed_extra_info)[:, 0]

    return gt_future_sensitivities, prediction_sensitivities, (ego_x, init_timestep, pred_horizon, gt_mus, pred_mus, pred_probs)


def plot_combined_preds_and_sens(env: NuScenesEnvironment, 
                                 learned_theta: np.ndarray,
                                 expert_x: torch.Tensor,
                                 expert_u: torch.Tensor, 
                                 extra_info: Dict[str, Any],
                                 ep: int, 
                                 scene_t: int,
                                 save_individual_figs: bool = False):
    Path(f'plots/nuScenes/{env.env_name}_combined').mkdir(parents=True, exist_ok=True)
    
    # Getting prediction sensitivities
    gt_future_sensitivities, prediction_sensitivities, other_values = get_sensitivities_at_time(env, learned_theta, expert_x, expert_u, extra_info, ep, scene_t)
    ego_x, init_timestep, pred_horizon, gt_mus, pred_mus, pred_probs = other_values

    ego_color = 'coral'
    ado_color = 'royalblue'
    sens_low_color = np.array(colors.to_rgb(ado_color))
    sens_high_color = np.array(colors.to_rgb('deeppink'))
    preds_color = 'green'

    ep_len = extra_info['ep_lengths'][ep] + 1
    ego_curr_and_future = ego_x[scene_t - init_timestep : scene_t - init_timestep + pred_horizon + 1]
    ego_len = torch.isfinite(ego_curr_and_future.sum(-1)).sum()
    ego_curr_and_future = ego_curr_and_future[:ego_len]

    gt_lens = torch.isfinite(gt_mus.sum(-1)).sum(-1)

    fig, ax = plot_map(dataroot=env.data_root,
                        map_name=env.helper.get_map_name_from_sample_token(env.env.scenes[ep].name), 
                        map_patch=(ego_x[:ep_len, 0].min() - 10, 
                                    ego_x[:ep_len, 1].min() - 10, 
                                    ego_x[:ep_len, 0].max() + 10, 
                                    ego_x[:ep_len, 1].max() + 10))

    for _artist in ax.lines + ax.collections + ax.patches + ax.images:
        _artist.set_label(s=None)

    ax.scatter(ego_curr_and_future[[0], 0], ego_curr_and_future[[0], 1], label='Ego Vehicle', c=ego_color)
    ax.plot(ego_curr_and_future[:, 0], ego_curr_and_future[:, 1], label='Ego Motion Plan', c=ego_color)

    ax.plot([], [], color=preds_color, label='Predictions')

    scaled_gt_sens = torch.clamp(gt_future_sensitivities*100, max=1.0).numpy()
    gt_sens_colors = interpolate_colors(scaled_gt_sens[:, np.newaxis], sens_low_color, sens_high_color)

    ax.scatter(gt_mus[:, 0, 0], gt_mus[:, 0, 1], label='Other Agents', c=gt_sens_colors, zorder=10)
    for i in range(gt_mus.shape[0]):
        ax.plot(gt_mus[i, :gt_lens[i], 0], gt_mus[i, :gt_lens[i], 1], label='GT Future' if i == 0 else None, c=gt_sens_colors[i])

    for i in range(pred_mus.shape[0]):
        pis = pred_probs[i]
        for k in range(pred_mus.shape[2]):
            ax.plot([gt_mus[i, 0, 0].item()] + pred_mus[i, :, k, 0].tolist(), 
                    [gt_mus[i, 0, 1].item()] + pred_mus[i, :, k, 1].tolist(), 
                    c=preds_color, alpha=pis[k].item())

    ax.grid(False)
    ax.legend(loc='best', prop={'size': 40})
    if save_individual_figs:
        fig.savefig(f'plots/nuScenes/{env.env_name}_combined/ph_{pred_horizon}_ep_{ep}_t_{scene_t}.pdf', dpi=200)
    combined_fig = mplfig_to_npimage(fig)
    plt.close(fig)

    return combined_fig


def predictions_and_sensitivities(env: NuScenesEnvironment, 
                                  learned_theta: np.ndarray,
                                  expert_x: torch.Tensor,
                                  expert_u: torch.Tensor, 
                                  extra_info: Dict[str, Any],
                                  ep: int, 
                                  scene_t: int, 
                                  ret_pred_fig: bool = False,
                                  save_individual_figs: bool = False):
    Path(f'plots/nuScenes/{env.env_name}_preds_and_sens').mkdir(parents=True, exist_ok=True)

    # Getting prediction sensitivities
    gt_future_sensitivities, prediction_sensitivities, other_values = get_sensitivities_at_time(env, learned_theta, expert_x, expert_u, extra_info, ep, scene_t)
    ego_x, init_timestep, pred_horizon, gt_mus, pred_mus, pred_probs = other_values

    ego_color = 'coral'
    ado_color = 'royalblue'

    ep_len = extra_info['ep_lengths'][ep] + 1
    ego_curr_and_future = ego_x[scene_t - init_timestep : scene_t - init_timestep + pred_horizon + 1]
    ego_len = torch.isfinite(ego_curr_and_future.sum(-1)).sum()
    ego_curr_and_future = ego_curr_and_future[:ego_len]

    gt_lens = torch.isfinite(gt_mus.sum(-1)).sum(-1)

    ########################
    # Plotting Predictions #
    ########################
    if ret_pred_fig:
        fig, ax = plot_map(dataroot=env.data_root,
                           map_name=env.helper.get_map_name_from_sample_token(env.env.scenes[ep].name), 
                           map_patch=(ego_x[:ep_len, 0].min() - 50, 
                                      ego_x[:ep_len, 1].min() - 50, 
                                      ego_x[:ep_len, 0].max() + 50, 
                                      ego_x[:ep_len, 1].max() + 50))
        ax.scatter(ego_curr_and_future[[0], 0], ego_curr_and_future[[0], 1], label='Ego', c=ego_color)
        ax.plot(ego_curr_and_future[:, 0], ego_curr_and_future[:, 1], label='Ego Motion Plan', c=ego_color)

        ax.scatter(gt_mus[:, 0, 0], gt_mus[:, 0, 1], label='Ado', c=ado_color)
        for i in range(gt_mus.shape[0]):
            ax.plot(gt_mus[i, :gt_lens[i], 0], gt_mus[i, :gt_lens[i], 1], label='Ado GT Future' if i == 0 else None, c=ado_color)

        for i in range(pred_mus.shape[0]):
            pis = pred_probs[i]
            for k in range(pred_mus.shape[2]):
                ax.plot([gt_mus[i, 0, 0].item()] + pred_mus[i, :, k, 0].tolist(), 
                        [gt_mus[i, 0, 1].item()] + pred_mus[i, :, k, 1].tolist(), 
                        label='Predictions' if i == 0 and k == 0 else None, 
                        c='green', alpha=pis[k].item()*3)
                ax.plot([gt_mus[i, 0, 0].item()] + pred_mus[i, :, k, 0].tolist(), 
                        [gt_mus[i, 0, 1].item()] + pred_mus[i, :, k, 1].tolist(), 
                        label='All Preds' if i == 0 and k == 0 else None, 
                        c='k', alpha=0.1)

        ax.legend(loc='upper right')
        fig.tight_layout()
        if save_individual_figs:
            fig.savefig(f'plots/nuScenes/{env.env_name}_preds_and_sens/preds_ep_{ep}_t_{scene_t}.pdf', dpi=200)
        prediction_fig = mplfig_to_npimage(fig)
        plt.close(fig)
    
    ################################
    # Plotting Agent Sensitivities #
    ################################
    fig, ax = plot_map(dataroot=env.data_root,
                       map_name=env.helper.get_map_name_from_sample_token(env.env.scenes[ep].name), 
                       map_patch=(ego_x[:ep_len, 0].min() - 50, 
                                  ego_x[:ep_len, 1].min() - 50, 
                                  ego_x[:ep_len, 0].max() + 50, 
                                  ego_x[:ep_len, 1].max() + 50))
    ax.scatter(ego_curr_and_future[[0], 0], ego_curr_and_future[[0], 1], label='Ego', c=ego_color)
    ax.plot(ego_curr_and_future[:, 0], ego_curr_and_future[:, 1], label='Ego Motion Plan', c=ego_color)

    ax.scatter(gt_mus[:, 0, 0], gt_mus[:, 0, 1], label='Ado', c=ado_color)
    for i in range(gt_mus.shape[0]):
        ax.plot(gt_mus[i, :gt_lens[i], 0], gt_mus[i, :gt_lens[i], 1], label='Ado GT Future' if i == 0 else None, c=ado_color)

    for i in range(pred_mus.shape[0]):
        for k in range(pred_mus.shape[2]):
            ax.plot([gt_mus[i, 0, 0].item()] + pred_mus[i, :, k, 0].tolist(), 
                    [gt_mus[i, 0, 1].item()] + pred_mus[i, :, k, 1].tolist(), 
                    label='Predictions' if i == 0 and k == 0 else None, 
                    c='green', alpha=torch.clamp(gt_future_sensitivities[i]*100, max=1.0).item())

    ax.legend(loc='upper right')
    fig.tight_layout()
    if save_individual_figs:
        fig.savefig(f'plots/nuScenes/{env.env_name}_preds_and_sens/agent_sens_ep_{ep}_t_{scene_t}.pdf', dpi=200)
    agent_sens_fig = mplfig_to_npimage(fig)
    plt.close(fig)

    #####################################
    # Plotting Prediction Sensitivities #
    #####################################
    fig, ax = plot_map(dataroot=env.data_root,
                       map_name=env.helper.get_map_name_from_sample_token(env.env.scenes[ep].name), 
                       map_patch=(ego_x[:ep_len, 0].min() - 50, 
                                  ego_x[:ep_len, 1].min() - 50, 
                                  ego_x[:ep_len, 0].max() + 50, 
                                  ego_x[:ep_len, 1].max() + 50))
    ax.scatter(ego_curr_and_future[[0], 0], ego_curr_and_future[[0], 1], label='Ego', c=ego_color)
    ax.plot(ego_curr_and_future[:, 0], ego_curr_and_future[:, 1], label='Ego Motion Plan', c=ego_color)

    ax.scatter(gt_mus[:, 0, 0], gt_mus[:, 0, 1], label='Ado', c=ado_color)
    for i in range(gt_mus.shape[0]):
        ax.plot(gt_mus[i, :gt_lens[i], 0], gt_mus[i, :gt_lens[i], 1], label='Ado GT Future' if i == 0 else None, c=ado_color)

    for i in range(pred_mus.shape[0]):
        for k in range(pred_mus.shape[2]):
            ax.plot([gt_mus[i, 0, 0].item()] + pred_mus[i, :, k, 0].tolist(), 
                    [gt_mus[i, 0, 1].item()] + pred_mus[i, :, k, 1].tolist(), 
                    label='Predictions' if i == 0 and k == 0 else None, 
                    c='green', alpha=torch.clamp(prediction_sensitivities[i, k]*100, max=1.0).item())

    ax.legend(loc='upper right')
    fig.tight_layout()
    if save_individual_figs:
        fig.savefig(f'plots/nuScenes/{env.env_name}_preds_and_sens/pred_sens_ep_{ep}_t_{scene_t}.pdf', dpi=200)
    pred_sens_fig = mplfig_to_npimage(fig)
    plt.close(fig)

    return (prediction_fig if ret_pred_fig else None, 
            agent_sens_fig, 
            pred_sens_fig, 
            gt_future_sensitivities.numpy(), 
            prediction_sensitivities.numpy())


def evaluate_learned_theta(env: NuScenesEnvironment, 
                           theta: np.ndarray, 
                           extra_info: Dict[str, Any], 
                           plot: bool = False, 
                           verbose: bool = False,
                           prefix: Optional[str] = None):
    scene_idxs = extra_info['scene_idxs']
    features = extra_info['features'].numpy()
    ep_lengths = extra_info['ep_lengths']
    max_err_mags = list()

    if plot:
        # Create it in case it doesn't exist.
        Path('plots/nuScenes/reoptimized').mkdir(parents=True, exist_ok=True)

    for idx in tqdm(scene_idxs):
        if idx == 9:
            # The map (specifically its lanes) has really bad localization in this scene.
            continue
        
        ep_len = ep_lengths[idx]
        if ep_len <= 1:
            continue

        traj, controls = env.reoptimize_with_theta(theta, 
                                                   env.env.scenes[idx], utils.index_dict(extra_info, idx),
                                                   plot, verbose, f'{prefix}_{idx}')

        max_error_magnitude = traj - features[idx, :ep_len+1, :env.state_dim]
        max_error_magnitude[..., 2] = utils.wrap(max_error_magnitude[..., 2])
        max_err_mags.append(np.abs(max_error_magnitude).max(axis=0))
        print(f'{idx}: Max error mag', max_err_mags[-1])
    
    max_err_mags = np.stack(max_err_mags)
    print('Avg. Maximum Error Magnitude:', np.mean(max_err_mags, axis=0))


def eval_preds_reweighted(env: NuScenesEnvironment, 
                          learned_theta: np.ndarray,
                          expert_x: torch.Tensor, 
                          expert_u: torch.Tensor, 
                          extra_info: Dict[str, Any], 
                          ep: int,
                          make_plots: bool = False):
    Path(f'plots/nuScenes/{env.env_name}_reweighted_metrics').mkdir(parents=True, exist_ok=True)

    orig_indexed_extra_info = utils.index_dict(extra_info, ep)
    init_timestep = orig_indexed_extra_info['init_timestep']
    pred_horizon = orig_indexed_extra_info['pred_horizon']
    
    scene_sensitivities: List[torch.Tensor] = list()
    og_metrics: Dict[str, List[float]] = defaultdict(list)
    weighted_metrics: Dict[str, List[float]] = defaultdict(list)

    agents_mask: Dict[str, torch.Tensor] = dict()
    raw_metrics_data: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    weighted_metrics_data: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    time_range = range(init_timestep, init_timestep + orig_indexed_extra_info['ep_lengths'] - pred_horizon + 1)
    for scene_t in tqdm(time_range, desc='Timesteps', position=1):
        # print(f'Time {scene_t}:')
        
        # Getting prediction sensitivities
        indexed_extra_info = copy.deepcopy(orig_indexed_extra_info)
        indexed_extra_info['features'] = indexed_extra_info['features'][scene_t - init_timestep]
        for k, v in indexed_extra_info.items():
            if isinstance(v, dict):
                indexed_extra_info[k] = v[scene_t - init_timestep]

        ego_x = expert_x[ep].reshape((-1, env.state_dim))
        ego_u = expert_u[ep].reshape((-1, env.control_dim))

        gmm_dict: Dict[Node, GMM2D] = indexed_extra_info['predictions']
        features = indexed_extra_info['features']
        scene_offset = features[11:13]

        prediction_gmms = gmm_dict.values()
        pred_mus = torch.stack([gmm.mus.squeeze() + scene_offset for gmm in prediction_gmms])
        pred_probs = torch.stack([gmm.pis_cat_dist.probs.squeeze()[0] for gmm in prediction_gmms])

        prediction_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        pred_mus, pred_probs,
                                                        indexed_extra_info)

        # Getting prediction_horizon
        gt_mus = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t + 1, scene_t + pred_horizon]), pos_dict)) + scene_offset 
                for node in gmm_dict]
        )

        gt_future_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        gt_mus.unsqueeze(-2), torch.ones((gt_mus.shape[0], 1)),
                                                        indexed_extra_info)[:, 0]

        scene_sensitivities.append(torch.linalg.norm(gt_future_sensitivities))

        # Getting the history
        past_pos = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t - 7, scene_t]), pos_dict)) + scene_offset 
                for node in gmm_dict]
        )

        ####################
        # Standard Metrics #
        ####################
        gt_all_finite = ~torch.isnan(gt_mus.sum(-1)).any(-1)
        history_all_finite = ~torch.isnan(past_pos.sum(-1)).any(-1)
        mask = gt_all_finite & history_all_finite
        if not mask.any():
            # No agent satisfies the above conditions.
            continue

        # Doing this here to ensure that we mask these as well (so the agents line up with those that are actually evaluated).
        nodes_list: List[Node] = [node for node in gmm_dict.keys()]
        agents_mask['VEHICLE'] = torch.tensor([True if node.type.name == 'VEHICLE' else False for node in nodes_list])[mask]
        agents_mask['PEDESTRIAN'] = ~agents_mask['VEHICLE']  # Can get away with this because nuScenes effectively only has these two types.

        masked_pred_probs = pred_probs[mask]
        masked_pred_mus = pred_mus[mask]
        masked_gt_mus = gt_mus[mask]

        most_likely_idx = masked_pred_probs.argmax(dim=-1)
        ml_pred_mus = masked_pred_mus[range(masked_pred_mus.shape[0]), :, most_likely_idx, :]
        
        ades = compute_ade_pt(ml_pred_mus, masked_gt_mus)
        fdes = compute_fde_pt(ml_pred_mus.permute(0, 2, 1), masked_gt_mus)
        
        mean_nlls, final_nlls = list(), list()
        for idx, (node, gmm) in enumerate(gmm_dict.items()):
            if not mask[idx]:
                continue

            mean_nll, final_nll = compute_nll_pt(gmm, gt_mus[idx] - scene_offset)
            mean_nlls.append(mean_nll)
            final_nlls.append(final_nll)
        
        mean_nlls = torch.cat(mean_nlls)
        final_nlls = torch.cat(final_nlls)

        mean_lls = torch.logsumexp(-mean_nlls, dim=0) - np.log(mean_nlls.shape[0])
        final_lls = torch.logsumexp(-final_nlls, dim=0) - np.log(mean_nlls.shape[0])

        # print('\tOriginal:', ades.mean().item(), fdes.mean().item(), -mean_lls.item(), -final_lls.item())
        og_metrics['ade'].append(ades.mean().item())
        og_metrics['fde'].append(fdes.mean().item())
        og_metrics['anll'].append(-mean_lls.item())
        og_metrics['fnll'].append(-final_lls.item())

        for agent_type, agent_mask in agents_mask.items():
            raw_metrics_data[agent_type]['ade'].append(ades[agent_mask].numpy())
            raw_metrics_data[agent_type]['fde'].append(fdes[agent_mask].numpy())
            raw_metrics_data[agent_type]['anll'].append(mean_nlls[agent_mask].numpy())
            raw_metrics_data[agent_type]['fnll'].append(final_nlls[agent_mask].numpy())
            raw_metrics_data[agent_type]['sens'].append(gt_future_sensitivities[mask][agent_mask].numpy())
        
        ######################
        # Reweighted Metrics #
        ######################
        # masked_agent_weights = 1.0 + gt_future_sensitivities[mask] * 10

        # w_ades = compute_ade_pt(ml_pred_mus, masked_gt_mus) * masked_agent_weights
        # w_fdes = compute_fde_pt(ml_pred_mus.permute(0, 2, 1), masked_gt_mus) * masked_agent_weights
        
        # w_mean_nlls, w_final_nlls = list(), list()
        # for idx, (node, gmm) in enumerate(gmm_dict.items()):
        #     if not mask[idx]:
        #         continue

        #     mean_nll, final_nll = compute_nll_pt(gmm, gt_mus[idx] - scene_offset)
        #     w_mean_nlls.append(mean_nll)
        #     w_final_nlls.append(final_nll)
        
        # w_mean_nlls = torch.cat(w_mean_nlls)
        # w_final_nlls = torch.cat(w_final_nlls)

        # w_mean_lls = torch.logsumexp(-w_mean_nlls + torch.log(masked_agent_weights), dim=0) - torch.log(masked_agent_weights.sum())
        # w_final_lls = torch.logsumexp(-w_final_nlls + torch.log(masked_agent_weights), dim=0) - torch.log(masked_agent_weights.sum())

        # # print('\tWeighted:', w_ades.mean().item(), w_fdes.mean().item(), -w_mean_lls.item(), -w_final_lls.item())
        # weighted_metrics['ade'].append(w_ades.mean().item())
        # weighted_metrics['fde'].append(w_fdes.mean().item())
        # weighted_metrics['anll'].append(-w_mean_lls.item())
        # weighted_metrics['fnll'].append(-w_final_lls.item())

        # for agent_type, agent_mask in agents_mask.items():
        #     weighted_metrics_data[agent_type]['ade'].append(w_ades[agent_mask].numpy())
        #     weighted_metrics_data[agent_type]['fde'].append(w_fdes[agent_mask].numpy())
        #     weighted_metrics_data[agent_type]['anll'].append(w_mean_nlls[agent_mask].numpy())
        #     weighted_metrics_data[agent_type]['fnll'].append(w_final_nlls[agent_mask].numpy())

    if make_plots:
        # Sensitivity Magnitudes
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_range, scene_sensitivities)

        ax.set_ylabel('Planning Sensitivity')
        ax.set_xlabel('Scene Timestep')
        ax.set_xticks(list(time_range))
        ax.set_ylim((0, 0.1))

        fig.tight_layout()
        fig.savefig(f'plots/nuScenes/{env.env_name}_reweighted_metrics/ep_{ep}_sensitivities.pdf', dpi=300)
        plt.close(fig)

        # ADE
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_range, og_metrics['ade'], label='Original')
        ax.plot(time_range, weighted_metrics['ade'], label='Weighted')

        ax.set_ylabel('ADE (m)')
        ax.set_xlabel('Scene Timestep')
        ax.set_xticks(list(time_range))
        ax.legend(loc='best')

        fig.tight_layout()
        fig.savefig(f'plots/nuScenes/{env.env_name}_reweighted_metrics/ep_{ep}_ade.pdf', dpi=300)
        plt.close(fig)

        # FDE
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_range, og_metrics['fde'], label='Original')
        ax.plot(time_range, weighted_metrics['fde'], label='Weighted')

        ax.set_ylabel('FDE (m)')
        ax.set_xlabel('Scene Timestep')
        ax.set_xticks(list(time_range))
        ax.legend(loc='best')
        
        fig.tight_layout()
        fig.savefig(f'plots/nuScenes/{env.env_name}_reweighted_metrics/ep_{ep}_fde.pdf', dpi=300)
        plt.close(fig)

        # ANLL
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_range, og_metrics['anll'], label='Original')
        ax.plot(time_range, weighted_metrics['anll'], label='Weighted')

        ax.set_ylabel('ANLL (nats)')
        ax.set_xlabel('Scene Timestep')
        ax.set_xticks(list(time_range))
        ax.legend(loc='best')
        
        fig.tight_layout()
        fig.savefig(f'plots/nuScenes/{env.env_name}_reweighted_metrics/ep_{ep}_anll.pdf', dpi=300)
        plt.close(fig)

        # FNLL
        fig, ax = plt.subplots(figsize=(15, 5))
        ax.plot(time_range, og_metrics['fnll'], label='Original')
        ax.plot(time_range, weighted_metrics['fnll'], label='Weighted')

        ax.set_ylabel('FNLL (nats)')
        ax.set_xlabel('Scene Timestep')
        ax.set_xticks(list(time_range))
        ax.legend(loc='best')
        
        fig.tight_layout()
        fig.savefig(f'plots/nuScenes/{env.env_name}_reweighted_metrics/ep_{ep}_fnll.pdf', dpi=300)
        plt.close(fig)

    return raw_metrics_data#, weighted_metrics_data


def most_sensitive(env: NuScenesEnvironment, 
                   learned_theta: np.ndarray,
                   expert_x: torch.Tensor, 
                   expert_u: torch.Tensor, 
                   extra_info: Dict[str, Any], 
                   ep: int):
    orig_indexed_extra_info = utils.index_dict(extra_info, ep)
    init_timestep = orig_indexed_extra_info['init_timestep']
    pred_horizon = orig_indexed_extra_info['pred_horizon']
    
    time_range = range(init_timestep, init_timestep + orig_indexed_extra_info['ep_lengths'])

    scene_pred_sensitivities: torch.Tensor = torch.zeros((len(time_range), ))
    scene_agent_sensitivities: torch.Tensor = torch.zeros((len(time_range), ))
    for scene_t in tqdm(time_range, desc='Timesteps', position=1):
        # print(f'Time {scene_t}:')
        
        # Getting prediction sensitivities
        indexed_extra_info = copy.deepcopy(orig_indexed_extra_info)
        indexed_extra_info['features'] = indexed_extra_info['features'][scene_t - init_timestep]
        for k, v in indexed_extra_info.items():
            if isinstance(v, dict):
                indexed_extra_info[k] = v[scene_t - init_timestep]

        ego_x = expert_x[ep].reshape((-1, env.state_dim))
        ego_u = expert_u[ep].reshape((-1, env.control_dim))

        gmm_dict: Dict[Node, GMM2D] = indexed_extra_info['predictions']
        features = indexed_extra_info['features']
        scene_offset = features[11:13]

        prediction_gmms = gmm_dict.values()
        pred_mus = torch.stack([gmm.mus.squeeze() + scene_offset for gmm in prediction_gmms])
        pred_probs = torch.stack([gmm.pis_cat_dist.probs.squeeze()[0] for gmm in prediction_gmms])

        prediction_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                         ego_x[scene_t - init_timestep], 
                                                         ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                         pred_mus, pred_probs,
                                                         indexed_extra_info)

        # Getting prediction_horizon
        gt_mus = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t + 1, scene_t + pred_horizon]), pos_dict)) + scene_offset 
                for node in gmm_dict]
        ).unsqueeze(-2)

        gt_future_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        gt_mus, torch.ones((gt_mus.shape[0], 1)),
                                                        indexed_extra_info)

        scene_pred_sensitivities[scene_t - init_timestep] = prediction_sensitivities.max()
        scene_agent_sensitivities[scene_t - init_timestep] = gt_future_sensitivities.max()

    pred_max_vals, pred_max_idxs = torch.max(scene_pred_sensitivities, dim=0) if torch.numel(scene_pred_sensitivities) > 0 else (torch.tensor([np.nan]), torch.tensor([np.nan]))
    agent_max_vals, agent_max_idxs = torch.max(scene_agent_sensitivities, dim=0) if torch.numel(scene_agent_sensitivities) > 0 else (torch.tensor([np.nan]), torch.tensor([np.nan]))
    return agent_max_vals, agent_max_idxs + init_timestep, pred_max_vals, pred_max_idxs + init_timestep


def sens_and_errors(env: NuScenesEnvironment, 
                    learned_theta: np.ndarray,
                    expert_x: torch.Tensor, 
                    expert_u: torch.Tensor, 
                    extra_info: Dict[str, Any], 
                    ep: int):
    orig_indexed_extra_info = utils.index_dict(extra_info, ep)
    init_timestep = orig_indexed_extra_info['init_timestep']
    pred_horizon = orig_indexed_extra_info['pred_horizon']
    
    og_metrics: Dict[str, List[float]] = defaultdict(list)
    
    gt_sens: Dict[str, List[np.ndarray]] = defaultdict(list)
    pred_sens: Dict[str, List[np.ndarray]] = defaultdict(list)
    node_info: Dict[str, List[Node]] = defaultdict(list)

    agents_mask: Dict[str, torch.Tensor] = dict()
    raw_metrics_data: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))

    time_range = range(init_timestep, init_timestep + orig_indexed_extra_info['ep_lengths'] - pred_horizon + 1)
    for scene_t in tqdm(time_range, desc='Timesteps', position=1):
        # print(f'Time {scene_t}:')
        
        # Getting prediction sensitivities
        indexed_extra_info = copy.deepcopy(orig_indexed_extra_info)
        indexed_extra_info['features'] = indexed_extra_info['features'][scene_t - init_timestep]
        for k, v in indexed_extra_info.items():
            if isinstance(v, dict):
                indexed_extra_info[k] = v[scene_t - init_timestep]

        ego_x = expert_x[ep].reshape((-1, env.state_dim))
        ego_u = expert_u[ep].reshape((-1, env.control_dim))

        gmm_dict: Dict[Node, GMM2D] = indexed_extra_info['predictions']
        features = indexed_extra_info['features']
        scene_offset = features[11:13]

        prediction_gmms = gmm_dict.values()
        pred_mus = torch.stack([gmm.mus.squeeze() + scene_offset for gmm in prediction_gmms])
        pred_probs = torch.stack([gmm.pis_cat_dist.probs.squeeze()[0] for gmm in prediction_gmms])

        prediction_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        pred_mus, pred_probs,
                                                        indexed_extra_info)

        # Getting prediction_horizon
        gt_mus = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t + 1, scene_t + pred_horizon]), pos_dict)) + scene_offset 
                for node in gmm_dict]
        )

        gt_future_sensitivities = env.get_sensitivities(torch.from_numpy(learned_theta), 
                                                        ego_x[scene_t - init_timestep], 
                                                        ego_u[scene_t - init_timestep : scene_t - init_timestep + pred_horizon], 
                                                        gt_mus.unsqueeze(-2), torch.ones((gt_mus.shape[0], 1)),
                                                        indexed_extra_info)[:, 0]

        # Getting the history
        past_pos = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t - 7, scene_t]), pos_dict)) + scene_offset 
                for node in gmm_dict]
        )

        ####################
        # Standard Metrics #
        ####################
        gt_all_finite = ~torch.isnan(gt_mus.sum(-1)).any(-1)
        history_all_finite = ~torch.isnan(past_pos.sum(-1)).any(-1)
        mask = gt_all_finite & history_all_finite
        if not mask.any():
            # No agent satisfies the above conditions.
            continue

        # Doing this here to ensure that we mask these as well (so the agents line up with those that are actually evaluated).
        nodes_list: List[Node] = [node for node in gmm_dict.keys()]
        agents_mask['VEHICLE'] = torch.tensor([True if node.type.name == 'VEHICLE' else False for node in nodes_list])[mask]
        agents_mask['PEDESTRIAN'] = ~agents_mask['VEHICLE']  # Can get away with this because nuScenes effectively only has these two types.

        masked_pred_probs = pred_probs[mask]
        masked_pred_mus = pred_mus[mask]
        masked_gt_mus = gt_mus[mask]

        most_likely_idx = masked_pred_probs.argmax(dim=-1)
        ml_pred_mus = masked_pred_mus[range(masked_pred_mus.shape[0]), :, most_likely_idx, :]
        
        ades = compute_ade_pt(ml_pred_mus, masked_gt_mus)
        fdes = compute_fde_pt(ml_pred_mus.permute(0, 2, 1), masked_gt_mus)
        
        mean_nlls, final_nlls = list(), list()
        for idx, (node, gmm) in enumerate(gmm_dict.items()):
            if not mask[idx]:
                continue

            mean_nll, final_nll = compute_nll_pt(gmm, gt_mus[idx] - scene_offset)
            mean_nlls.append(mean_nll)
            final_nlls.append(final_nll)
        
        mean_nlls = torch.cat(mean_nlls)
        final_nlls = torch.cat(final_nlls)

        mean_lls = torch.logsumexp(-mean_nlls, dim=0) - np.log(mean_nlls.shape[0])
        final_lls = torch.logsumexp(-final_nlls, dim=0) - np.log(mean_nlls.shape[0])

        # print('\tOriginal:', ades.mean().item(), fdes.mean().item(), -mean_lls.item(), -final_lls.item())
        og_metrics['ade'].append(ades.mean().item())
        og_metrics['fde'].append(fdes.mean().item())
        og_metrics['anll'].append(-mean_lls.item())
        og_metrics['fnll'].append(-final_lls.item())

        for agent_type, agent_mask in agents_mask.items():
            raw_metrics_data[agent_type]['ade'].append(ades[agent_mask].numpy())
            raw_metrics_data[agent_type]['fde'].append(fdes[agent_mask].numpy())
            raw_metrics_data[agent_type]['anll'].append(mean_nlls[agent_mask].numpy())
            raw_metrics_data[agent_type]['fnll'].append(final_nlls[agent_mask].numpy())
        
            gt_sens[agent_type].append(gt_future_sensitivities[mask][agent_mask].numpy())
            pred_sens[agent_type].append(prediction_sensitivities[mask][agent_mask].numpy())

            valid_nodes = [node for i, node in enumerate(nodes_list) if mask[i]]
            node_info[agent_type].append([nnode for j, nnode in enumerate(valid_nodes) if agent_mask[j]])

    return gt_sens, pred_sens, raw_metrics_data, node_info


def noisy_dets_sens_and_errors(env: NuScenesEnvironment, 
                               learned_theta: np.ndarray,
                               expert_x: torch.Tensor, 
                               extra_info: Dict[str, Any], 
                               ep: int,
                               noise_mag: float):
    orig_indexed_extra_info = utils.index_dict(extra_info, ep)
    init_timestep = orig_indexed_extra_info['init_timestep']
    
    gt_sens: Dict[str, List[np.ndarray]] = defaultdict(list)
    det_sens: Dict[str, List[np.ndarray]] = defaultdict(list)
    node_info: Dict[str, List[Node]] = defaultdict(list)

    agents_mask: Dict[str, torch.Tensor] = dict()

    time_range = range(init_timestep, init_timestep + orig_indexed_extra_info['ep_lengths'] + 1)
    for scene_t in tqdm(time_range, desc='Timesteps', position=2, disable=True):
        # print(f'Time {scene_t}:')
        
        # Getting prediction sensitivities
        indexed_extra_info = copy.deepcopy(orig_indexed_extra_info)
        indexed_extra_info['features'] = indexed_extra_info['features'][scene_t - init_timestep]
        for k, v in indexed_extra_info.items():
            if isinstance(v, dict):
                indexed_extra_info[k] = v[scene_t - init_timestep]

        ego_x = expert_x[ep].reshape((-1, env.state_dim))

        features = indexed_extra_info['features']
        scene_offset = features[11:13]

        # Getting prediction_horizon
        gt_mus = torch.stack(
            [torch.from_numpy(node.get(np.array([scene_t]), pos_dict)) + scene_offset 
                for node in indexed_extra_info['predictions']]
        )

        gt_future_sensitivities = env.get_det_sensitivities(torch.from_numpy(learned_theta), 
                                                            ego_x[scene_t - init_timestep], 
                                                            gt_mus,
                                                            indexed_extra_info)

        noisy_det_sensitivities = env.get_det_sensitivities(torch.from_numpy(learned_theta), 
                                                            ego_x[scene_t - init_timestep], 
                                                            gt_mus + torch.randn_like(gt_mus) * noise_mag,
                                                            indexed_extra_info)

        # Doing this here to ensure that we mask these as well (so the agents line up with those that are actually evaluated).
        nodes_list: List[Node] = [node for node in indexed_extra_info['predictions'].keys()]
        agents_mask['VEHICLE'] = torch.tensor([True if node.type.name == 'VEHICLE' else False for node in nodes_list])
        agents_mask['PEDESTRIAN'] = ~agents_mask['VEHICLE']  # Can get away with this because nuScenes effectively only has these two types.
        
        for agent_type, agent_mask in agents_mask.items():        
            gt_sens[agent_type].append(gt_future_sensitivities[agent_mask].numpy())
            det_sens[agent_type].append(noisy_det_sensitivities[agent_mask].numpy())

            node_info[agent_type].append([node for j, node in enumerate(nodes_list) if agent_mask[j]])

    return gt_sens, det_sens, node_info
