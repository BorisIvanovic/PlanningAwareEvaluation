import os
import sys
import torch
import random
import numpy as np
import pandas as pd
import moviepy.editor as mpy

from tqdm import tqdm, trange
from typing import List
from pathlib import Path
from pred_metric.environment import NuScenesEnvironment
from pred_metric.environment.nuScenes_data import analysis
from pred_metric.cioc import CIOC
from pred_metric.visualization import *
from collections import defaultdict

from functools import partial
from pathos.multiprocessing import ProcessPool as Pool

import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data_name', type=str, required=True,
                    help='name of the data file to load (no .pkl at the end)')
parser.add_argument('--num_trajs', type=int, required=True,
                    help='number of sample trajectories to use')
parser.add_argument('--ph', type=int,
                    help='the prediction horizon to use')

parser.add_argument('--full_pred_dist', action='store_true',
                    help='use full prediction distributions in calculating the collision cost term (more computationally intensive)')
parser.add_argument('--ego_pred_type', type=str, default='motion_plan', # Options are ['motion_plan', 'const_vel']
                    help='which open-loop ego motion extrapolation to use')

parser.add_argument('--prediction_module', action='store_true',
                    help="specify that we're evaluating prediction")
parser.add_argument('--detection_module', action='store_true',
                    help="specify that we're evaluating detection")

parser.add_argument('--train', action='store_true',
                    help="learn theta for the specified data")
parser.add_argument('--test', action='store_true',
                    help="run the data processing, but don't save anything")
parser.add_argument('--only_preprocess_data', action='store_true',
                    help='only preprocess data, no IOC training will be performed')
parser.add_argument('--eval_reoptimize', action='store_true',
                    help='only reoptimize with a learned theta, no IOC training will be performed')
parser.add_argument('--sens_analyses', action='store_true',
                    help='perform sensitivity analyses and plot the results')            
parser.add_argument('--nice_pred_plots', action='store_true',
                    help='make nice prediction sensitivity plots')
parser.add_argument('--nice_det_plots', action='store_true',
                    help='make nice detection sensitivity plots')
parser.add_argument('--noise_dets', action='store_true',
                    help='analyze detection sensitivities by noising the GT labels')   
parser.add_argument('--reweight_metric', action='store_true',
                    help='evaluate a prediction model compared to standard metrics')
parser.add_argument('--most_sensitive', action='store_true',
                    help='find the most sensitive agents')
parser.add_argument('--sens_and_errors', action='store_true',
                    help='get the model\'s prediction sensitivities and (raw) errors')

parser.add_argument('--seed', type=int, default=12,
                    help='random seed value')

# We use vectorize=True for autograd.functional.{jacobian,hessian} calls
# and see great performance improvements. Leaving this here in case some
# debugging is needed.
# torch._C._debug_only_display_vmap_fallback_warnings(True)
# torch.autograd.set_detect_anomaly(True)


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = 'nuScenes_' + args.data_name

    cached_file_exists = Path(f'pred_metric/environment/nuScenes_data/cached_data/{args.data_name}_{args.num_trajs}_ph_{args.ph}.dill').is_file()

    if args.prediction_module:
        gt_theta_dim = 6
    elif args.detection_module:
        gt_theta_dim = 5
    else:
        raise ValueError('Please specify which task you are training for with either --prediction_module or --detection_module')

    env = NuScenesEnvironment(env_name=args.data_name,
                              num_timesteps=35, 
                              gt_theta=np.ones((gt_theta_dim, ), dtype=np.float32),
                              data_path=f'./pred_metric/environment/nuScenes_data/cached_data/{data_path}.pkl',
                              load_nusc=(
                                  (not cached_file_exists) 
                                    or args.only_preprocess_data 
                                    or args.eval_reoptimize 
                                    or args.sens_analyses 
                                    or args.nice_pred_plots or args.nice_det_plots
                                    or args.test)
                             )

    expert_x, expert_u, extra_info = env.gen_expert_data(num_samples=args.num_trajs,
                                                         prediction_model_dir='./Trajectron-plus-plus/experiments/nuScenes/models/nuScenes_og_node_freqs-13_Aug_2021_11_52_51',
                                                         prediction_model_ckpt=12,
                                                         pred_horizon=args.ph,
                                                         use_pred_uncertainty=args.full_pred_dist,
                                                         ego_pred_type=args.ego_pred_type,
                                                         test=args.test)
    
    ############
    # Training #
    ############
    if args.train:
        # If none of the evaluation options, learn theta.
        print('Using full predicted distributions in cost function:', args.full_pred_dist)

        init_theta = torch.abs(torch.randn(env.theta_dim, dtype=torch.float32) * 0.01)
        init_theta[0] = 1.0

        extra_info['training'] = True

        ioc_alg = CIOC(env)
        learned_theta, theta_r, training_info = ioc_alg.fit(expert_x, expert_u, extra_info=extra_info, init_theta=init_theta)
        
        print('Learned Theta:', learned_theta)
        print('Slack Variable:', theta_r)

        plot_training_loss_curve(training_info)
        plot_training_slack_vars(training_info)
        # plot_training_thetas_nd(training_info, env.gt_theta)
        plt.show()

    ###############
    # Evaluations #
    ###############
    pred_horizon = extra_info['pred_horizon']
    if gt_theta_dim == 6:
        # Prediction
        if args.ego_pred_type == 'motion_plan' and args.ph == 2:
            learned_theta = np.array([1.719, 0.566, 0., 9.859, 1.353, 1.681]) # With the distance-based prediction term, num_trajs 128, motion_plan ego preds, L-BFGS lr=0.8
        elif args.ego_pred_type == 'motion_plan' and args.ph == 4:
            learned_theta = np.array([1.721, 0.652, 0., 8.755, 1.362, 1.440]) # With the distance-based prediction term, num_trajs 128, motion_plan ego preds, L-BFGS lr=0.95
        elif args.ego_pred_type == 'motion_plan' and args.ph == 6:
            learned_theta = np.array([1.722, 0.562, 0., 11.865, 1.352, 0.241]) # With the distance-based prediction term, num_trajs 128, motion_plan ego preds, L-BFGS lr=0.95

        elif args.ego_pred_type == 'const_vel' and args.ph == 4:
            learned_theta = np.array([1.718, 0.554, 0., 0.849, 1.354, 4.634]) # With the distance-based prediction term, num_trajs 128, const_vel ego preds, L-BFGS lr=0.9

    elif gt_theta_dim == 5:
        if args.ego_pred_type == 'motion_plan':
            learned_theta = np.array([1.13, 0.551, 0., 12.176, 1.359]) # with the new mu_r initialization (old one it was just 1.0), num_trajs 256, L-BFGS lr=0.95

    # Making some of the following analyses able to work with both prediction and detection thetas.
    if gt_theta_dim == 5:
        learned_theta = np.concatenate([learned_theta, np.zeros((1,))], axis=0)

    # Analyses:
    if args.eval_reoptimize:
        analysis.evaluate_learned_theta(env, learned_theta, extra_info, plot=True, verbose=False, prefix=('det' if gt_theta_dim == 5 else 'pred'))

    if args.nice_pred_plots:
        for ep in tqdm([4, 14, 17, 25]): # trange(extra_info['scene_idxs'].shape[0], desc='Scenes', position=0):
            ep_len = extra_info['ep_lengths'][ep] + 1

            figs = list()
            for t in trange(extra_info['init_timestep'], extra_info['init_timestep'] + ep_len, desc='Timesteps', position=1):
                fig = analysis.plot_combined_preds_and_sens(env, learned_theta, expert_x, expert_u, extra_info, 
                                                            ep=ep, scene_t=t, save_individual_figs=True)
                figs.append(fig)

            if len(figs) > 0:
                clip = mpy.ImageSequenceClip(figs, fps=1/0.25)
                clip.write_videofile(f'plots/nuScenes/{env.env_name}_combined/ph_{pred_horizon}_ep_{ep}_movie.mp4')

    if args.sens_analyses:
        Path(f'plots/nuScenes/{env.env_name}_sensitivity_values').mkdir(parents=True, exist_ok=True)
        for ep in trange(extra_info['scene_idxs'].shape[0], desc='Scenes', position=0):
            with open(f"plots/nuScenes/{env.env_name}_sensitivity_values/scene_{ep}.txt", "w") as text_file:
                pass # Just clearing the file.

            pred_figs: List[np.ndarray] = list()
            agent_sens_figs: List[np.ndarray] = list()
            pred_sens_figs: List[np.ndarray] = list()
            plot_pred_fig: bool = True
            ep_len = extra_info['ep_lengths'][ep] + 1

            for t in trange(extra_info['init_timestep'], extra_info['init_timestep'] + ep_len, desc='Timesteps', position=1):
                pred_fig, agent_sens_fig, pred_sens_fig, gt_sens, pred_sens = analysis.predictions_and_sensitivities(env, learned_theta, expert_x, expert_u, extra_info, ep=ep, scene_t=t, ret_pred_fig=plot_pred_fig)
                pred_figs.append(pred_fig)
                agent_sens_figs.append(agent_sens_fig)
                pred_sens_figs.append(pred_sens_fig)

                with open(f"plots/nuScenes/{env.env_name}_sensitivity_values/scene_{ep}.txt", "a") as text_file:
                    text_file.write(f'Time {t}: GT Future Sensitivities\n {np.array2string(gt_sens, threshold=sys.maxsize, max_line_width=100)}\n')
                    text_file.write(f'Time {t}: Prediction Sensitivities\n {np.array2string(pred_sens, threshold=sys.maxsize, max_line_width=100)}\n')

            if len(agent_sens_figs) > 0:
                if plot_pred_fig:
                    clip = mpy.ImageSequenceClip(pred_figs, fps=1/0.25)
                    clip.write_videofile(f'plots/nuScenes/{env.env_name}_preds_and_sens/preds_ep_{ep}_movie.mp4')

                clip = mpy.ImageSequenceClip(agent_sens_figs, fps=1/0.25)
                clip.write_videofile(f'plots/nuScenes/{env.env_name}_preds_and_sens/agent_sens_ep_{ep}_movie.mp4')

                clip = mpy.ImageSequenceClip(pred_sens_figs, fps=1/0.25)
                clip.write_videofile(f'plots/nuScenes/{env.env_name}_preds_and_sens/pred_sens_ep_{ep}_movie.mp4')

    if args.reweight_metric:
        raw_summary: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        # weighted_summary: Dict[str, Dict[str, List[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
        for ep in trange(extra_info['scene_idxs'].shape[0], desc='Scenes', position=0):
            raw_metrics_data = analysis.eval_preds_reweighted(env, learned_theta, expert_x, expert_u, extra_info, ep=ep)

            for agent_type, type_metrics in raw_metrics_data.items():
                for metric, values_list in type_metrics.items():
                    raw_summary[agent_type][metric].append(np.concatenate(values_list))

            # for agent_type, type_metrics in weighted_metrics_data.items():
            #     for metric, values_list in type_metrics.items():
            #         weighted_summary[agent_type][metric].append(np.concatenate(values_list))

        for agent_type, type_metrics in raw_summary.items():
            for metric, values_list in type_metrics.items():
                raw_summary[agent_type][metric] = np.concatenate(values_list)

        # for agent_type, type_metrics in weighted_summary.items():
        #     for metric, values_list in type_metrics.items():
        #         weighted_summary[agent_type][metric] = np.concatenate(values_list)

        for agent_type, type_metrics in raw_summary.items():
            metric_df = pd.DataFrame(type_metrics)
            print(f'Original {agent_type}\n', metric_df.mean(axis=0))
            metric_df.to_csv(f'plots/nuScenes/{env.env_name}_reweighted_metrics/raw_{args.num_trajs}_ph_{pred_horizon}_{agent_type}.csv', index=False)

        # for agent_type, type_metrics in weighted_summary.items():
        #     metric_df = pd.DataFrame(type_metrics)
        #     print(f'Weighted {agent_type}\n', metric_df.mean(axis=0))
        #     metric_df.to_csv(f'plots/nuScenes/{env.env_name}_reweighted_metrics/weighted_{args.num_trajs}_ph_{pred_horizon}_{agent_type}.csv', index=False)

    if args.most_sensitive:
        Path(f'plots/nuScenes/{env.env_name}_most_sensitive').mkdir(parents=True, exist_ok=True)
        with open(f"plots/nuScenes/{env.env_name}_most_sensitive/max_sensitivities.csv", "w") as text_file:
            text_file.write('scene,max_gt_time,max_gt_sens,max_pred_time,max_pred_sens\n')

        for ep in trange(extra_info['scene_idxs'].shape[0], desc='Scenes', position=0):
            (agent_max_vals, agent_max_idxs, 
             pred_max_vals, pred_max_idxs) = analysis.most_sensitive(env, learned_theta, expert_x, expert_u, extra_info, ep=ep)

            with open(f"plots/nuScenes/{env.env_name}_most_sensitive/max_sensitivities.csv", "a") as text_file:
                text_file.write(f'{ep},{agent_max_idxs.item()},{agent_max_vals.item()},{pred_max_idxs.item()},{pred_max_vals.item()}\n')

    if args.sens_and_errors:
        Path(f'plots/nuScenes/{env.env_name}_sens_and_errors').mkdir(parents=True, exist_ok=True)
        with open(f"plots/nuScenes/{env.env_name}_sens_and_errors/sens_and_errors.csv", "w") as text_file:
            text_file.write('scene,time,node_id,gt_sens,og_ade,og_fde,og_anll,og_fnll\n')

        for ep in trange(extra_info['scene_idxs'].shape[0], desc='Scenes', position=0):
            (gt_sens, pred_sens, raw_metrics_data, node_info) = analysis.sens_and_errors(env, learned_theta, expert_x, expert_u, extra_info, ep=ep)

            with open(f"plots/nuScenes/{env.env_name}_sens_and_errors/sens_and_errors.csv", "a") as text_file:
                for node_type in node_info:
                    for scene_t in range(len(gt_sens[node_type])):
                        for i, node in enumerate(node_info[node_type][scene_t]):
                            text_file.write(f'{ep},{scene_t},{node.id},{gt_sens[node_type][scene_t][i]},' + ','.join(f'{og_metrics[scene_t][i]}' for og_metrics in raw_metrics_data[node_type].values()) + '\n')

    if args.noise_dets:
        Path(f'plots/nuScenes/{env.env_name}_noisy_dets').mkdir(parents=True, exist_ok=True)
        noise_mags = np.linspace(0.0, 4.0, num=9).tolist()
        num_workers = 5

        for ep in trange(len(extra_info['predictions']), desc='Removing GMM Objects'):
            # Doing this otherwise the following does not work (GMM objects can't be pickled, I guess).
            # It's ok because we do not need predictions for detection evalutation.
            for time in extra_info['predictions'][ep]:
                for node in extra_info['predictions'][ep][time]:
                    extra_info['predictions'][ep][time][node] = None
        
        with Pool(num_workers) as pool:
            list(
                tqdm(
                    pool.imap(
                        partial(parallel_noisy_dets,
                                env=env,
                                learned_theta=learned_theta,
                                expert_x=expert_x,
                                extra_info=extra_info),
                        list(enumerate(noise_mags))
                    ),
                    desc=f'Processing Noisy Dets ({num_workers} CPUs)',
                    total=len(noise_mags),
                    position=0,
                    disable=True
                )
            )

def parallel_noisy_dets(idx_noise_mag: Tuple[int, float], env, learned_theta, expert_x, extra_info):
    idx, noise_mag = idx_noise_mag
    with open(f"plots/nuScenes/{env.env_name}_noisy_dets/noisy_dets_sens_{noise_mag:.2f}.csv", "w") as text_file:
        text_file.write('scene,time,node_id,det_sens,noise_mag\n')

    for ep in trange(extra_info['scene_idxs'].shape[0], desc=f'{noise_mag}', position=idx):
        (gt_sens, det_sens, node_info) = analysis.noisy_dets_sens_and_errors(env, learned_theta, expert_x, extra_info, ep=ep, noise_mag=noise_mag)

        with open(f"plots/nuScenes/{env.env_name}_noisy_dets/noisy_dets_sens_{noise_mag:.2f}.csv", "a") as text_file:
            for node_type in node_info:
                for scene_t in range(len(det_sens[node_type])):
                    for i, node in enumerate(node_info[node_type][scene_t]):
                        text_file.write(f'{ep},{scene_t},{i},{det_sens[node_type][scene_t][i]},{noise_mag:.2f}\n')


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
