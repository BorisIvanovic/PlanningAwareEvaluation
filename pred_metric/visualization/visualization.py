import torch
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Optional, Tuple
from matplotlib.figure import Figure

from pred_metric.environment import utils

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap


def interpolate_colors(values: np.ndarray, sens_low_color: np.ndarray, sens_high_color: np.ndarray):
    return sens_low_color + values * (sens_high_color - sens_low_color)


def plot_map(dataroot: str, map_name: str, map_patch: Tuple[float, float, float, float]) -> Tuple[Figure, plt.Axes]:
    nusc_map = NuScenesMap(dataroot, map_name)
    bitmap = None #BitMap(dataroot, map_name, 'basemap')

    return nusc_map.render_map_patch(map_patch, 
                                     ['drivable_area',
                                      'road_segment',
                                      # 'road_block',
                                      'lane',
                                      'ped_crossing',
                                      'walkway',
                                      'stop_line',
                                      'carpark_area',
                                      'road_divider',
                                      'lane_divider'], 
                                     alpha=0.05,
                                     render_egoposes_range=False, 
                                     bitmap=bitmap)


def plot_lane_frenet(lane_states: np.ndarray, 
                     ego_state: Optional[np.ndarray] = None, 
                     projected_ego_pose: Optional[np.ndarray] = None,
                     seg_idx: Optional[int] = None):
    fig, ax = plt.subplots()
    ax.scatter(lane_states[:, 0], lane_states[:, 1], label='Lane')
    ax.scatter(ego_state[0], ego_state[1], label='Ego (Global)')
    ax.scatter(projected_ego_pose[0], projected_ego_pose[1], label='Ego (Proj)')

    ax.arrow(ego_state[0], ego_state[1], 0.1*np.cos(ego_state[2]), 0.1*np.sin(ego_state[2]))
    ax.arrow(projected_ego_pose[0], projected_ego_pose[1], 0.1*np.cos(projected_ego_pose[2]), 0.1*np.sin(projected_ego_pose[2]))

    ax.plot(lane_states[seg_idx:seg_idx+2, 0], lane_states[seg_idx:seg_idx+2, 1], label='Segment', color='blue')
    ax.plot([projected_ego_pose[0], ego_state[0]], [projected_ego_pose[1], ego_state[1]], label='Projection', color='red')

    ax.axis('equal')
    ax.grid(False)
    ax.legend(loc='best')
    plt.show()

def plot_training_loss_curve(training_info: Dict[str, np.ndarray]) -> Figure:
    loss_vals = training_info['loss_vals']

    fig, ax = plt.subplots()
    ax.plot(loss_vals)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'Training Loss (NLL)')
    ax.set_ylim(-50, 50)
    return fig

def plot_training_thetas_2d(training_info: Dict[str, np.ndarray]) -> Figure:
    learned_thetas = training_info['theta_vals']

    fig, ax = plt.subplots()
    ax.scatter(learned_thetas[:, 0], learned_thetas[:, 1], c=range(learned_thetas.shape[0]))
    ax.plot(learned_thetas[:, 0], learned_thetas[:, 1])
    ax.scatter(learned_thetas[[0], 0], learned_thetas[[0], 1], c='b', label='Init')
    ax.scatter(learned_thetas[[-1], 0], learned_thetas[[-1], 1], c='r', label='Final')
    axes = plt.axis()
    ax.plot([0, 100], [0, 100], c='gray', label='Ideal')
    ax.legend(loc='best')
    ax.set_xlabel(r'$\theta_1$')
    ax.set_ylabel(r'$\theta_2$')
    ax.set_xlim(axes[0], axes[1])
    ax.set_ylim(axes[2], axes[3])
    return fig

def plot_training_thetas_nd(training_info: Dict[str, np.ndarray], gt_theta: Optional[np.ndarray] = None) -> Figure:
    learned_thetas = training_info['theta_vals']

    # -1 because we don't want to plot theta_r here (we do it later separately).
    fig, axes = plt.subplots(nrows=(learned_thetas.shape[-1] - 1), figsize=(6, 8))
    for i in range(len(axes)):
        ax = axes[i]
        if gt_theta is not None:
            ax.axhline(y=gt_theta[i], color='darkgray', ls='--')
        ax.plot(learned_thetas[:, i])
        ax.set_ylabel(fr'$\theta_{i+1}$')
        ax.set_xlabel('Iterations')

    fig.tight_layout()
    return fig

def plot_training_slack_vars(training_info: Dict[str, np.ndarray]) -> Figure:
    learned_thetas = training_info['theta_vals']

    fig, ax = plt.subplots()
    ax.semilogy(learned_thetas[:, -1])
    ax.set_xlabel('Iterations')
    ax.set_ylabel(r'$\theta_r$')
    return fig

def plot_predictions_and_sensitivities(env, learned_theta, expert_x, extra_info, ep, t, fname: Optional[str] = None):
    ego_color = 'coral'
    ado_color = 'royalblue'

    reshaped_x = expert_x[ep].reshape((-1, env.state_dim))

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.plot(reshaped_x[:t+1, 0], reshaped_x[:t+1, 1], label='History', c='gray', ls='-', lw=10)
    ax.plot(reshaped_x[:t+1, 0] - reshaped_x[:t+1, 3], reshaped_x[:t+1, 1] - reshaped_x[:t+1, 4], c='gray', ls='-', lw=10)

    ax.add_patch(plt.Circle((reshaped_x[t, 0], reshaped_x[t, 1]),
                             radius=0.1, fc=ego_color, fill=True, zorder=10))
    ax.add_patch(plt.Circle((reshaped_x[t, 0] - reshaped_x[t, 3], reshaped_x[t, 1] - reshaped_x[t, 4]),
                             radius=0.1, fc=ado_color, fill=True, zorder=10))

    ax.scatter(np.nan, np.nan, label='Ego', c=ego_color, s=500)
    ax.scatter(np.nan, np.nan, label='Other', c=ado_color, s=500)

    hzn = 5
    ado_gt_future = torch.stack((reshaped_x[t:t+hzn, 0] - reshaped_x[t:t+hzn, 3], reshaped_x[t:t+hzn, 1] - reshaped_x[t:t+hzn, 4]), dim=-1)
    ax.plot([reshaped_x[t, 0], 0], [reshaped_x[t, 1], 0], label='Ego Plan', c='sandybrown', ls='--', lw=10)
    ax.scatter([0], [0], marker='*', s=1000, c='orangered', zorder=10)
    ax.plot(ado_gt_future[:, 0], ado_gt_future[:, 1], label='GT Future', c='lightblue', ls='-', lw=10)

    predicted_ado_pos = reshaped_x[t:t+hzn, 0:2] - reshaped_x[t:t+hzn, 5:7]
    heading = torch.atan2(predicted_ado_pos[1, 1] - predicted_ado_pos[0, 1],
                          predicted_ado_pos[1, 0] - predicted_ado_pos[0, 0])

    addition = torch.linspace(0, 0.15, 5)
    x_add = addition * torch.cos(heading + np.pi/2)
    y_add = addition * torch.sin(heading + np.pi/2)

    addition_vec = torch.stack((x_add, y_add), dim=-1)
    harmless_predicted = predicted_ado_pos - addition_vec
    dangerous_predicted = predicted_ado_pos + addition_vec

    ax.plot(harmless_predicted[:, 0], harmless_predicted[:, 1], label='Harmless', c='seagreen', ls=':', lw=10)
    ax.plot(dangerous_predicted[:, 0], dangerous_predicted[:, 1], label='Dangerous', c='mediumpurple', ls=':', lw=10)

    harmless_x = torch.clone(reshaped_x)
    harmless_x[t:t+hzn, 5:7] = reshaped_x[t:t+hzn, :2] - harmless_predicted
    harmless_sensitivities = env.get_sensitivities(learned_theta, harmless_x.flatten(), utils.index_dict(extra_info, ep))
    harmless_sens = harmless_sensitivities[t+hzn-1]
    print('Harmless', harmless_sens)

    agent_sensitivities = env.get_sensitivities(learned_theta, expert_x[ep], utils.index_dict(extra_info, ep))
    normal_sens = agent_sensitivities[t+hzn-1]
    print('Normal', normal_sens)

    dangerous_x = torch.clone(reshaped_x)
    dangerous_x[t:t+hzn, 5:7] = reshaped_x[t:t+hzn, :2] - dangerous_predicted
    dangerous_sensitivities = env.get_sensitivities(learned_theta, dangerous_x.flatten(), utils.index_dict(extra_info, ep))
    dangerous_sens = dangerous_sensitivities[t+hzn-1]
    print('Dangerous', dangerous_sens)

    harmless_de = torch.linalg.norm(harmless_predicted - ado_gt_future, dim=-1)
    dangerous_de = torch.linalg.norm(dangerous_predicted - ado_gt_future, dim=-1)

    print('Harmless *DE', harmless_de)
    print('Dangerous *DE', dangerous_de)

    harmless_weight = 1.0 + (harmless_sens - normal_sens).clamp(min=0)
    dangerous_weight = 1.0 + (dangerous_sens - normal_sens).clamp(min=0)

    print('pi Harmless *DE', harmless_de * harmless_weight)
    print('pi Dangerous *DE', dangerous_de * dangerous_weight)

    ax.legend(loc='best', frameon=False)
    ax.axis('equal')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    plt.tight_layout()
    plt.savefig(f'plots/ca_comparison_ep{ep}_t{t}.pdf' if fname is None else f'plots/{fname}', dpi=300)
