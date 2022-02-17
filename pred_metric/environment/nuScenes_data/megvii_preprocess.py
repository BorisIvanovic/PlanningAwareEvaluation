import sys
import os
import json
import numpy as np
import pandas as pd
import dill
import argparse
from tqdm import tqdm
from pyquaternion import Quaternion

from typing import List

sys.path.append("../../../Trajectron-plus-plus/trajectron")
from nuscenes.nuscenes import NuScenes
from nuscenes.prediction import PredictHelper
from nuscenes.map_expansion.map_api import NuScenesMap
from environment import Environment, Scene, Node

FREQUENCY = 2
dt = 1 / FREQUENCY
data_columns_vehicle = pd.MultiIndex.from_product([['position'], ['x', 'y']])
data_columns_vehicle = data_columns_vehicle.append(pd.MultiIndex.from_tuples([('heading', '°')]))

data_columns_pedestrian = pd.MultiIndex.from_product([['position'], ['x', 'y']])
data_columns_pedestrian = data_columns_pedestrian.append(pd.MultiIndex.from_tuples([('heading', '°')]))

standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0, 'std': 1},
            'y': {'mean': 0, 'std': 1}
        },
        'heading': {
            '°': {'mean': 0, 'std': np.pi}
        }
    },
    'VEHICLE': {
        'position': {
            'x': {'mean': 0, 'std': 80},
            'y': {'mean': 0, 'std': 80}
        },
        'heading': {
            '°': {'mean': 0, 'std': np.pi}
        }
    }
}


def process_data(data_path, version, output_path):
    nusc = NuScenes(version=version, dataroot=data_path, verbose=True)
    helper = PredictHelper(nusc)
    data_class_map = {'train_val': 'val', 
                      'train': 'train'}
    for data_class in ['train_val']: #, 'train']: # ['mini_train', 'mini_val']: 
        with open(f'raw/megvii_detections/megvii_{data_class_map[data_class]}.json', 'r') as f:
            megvii_detections = json.load(f)['results']

        count = 0
        for sample_token, sample_detections in tqdm(megvii_detections.items()):
            try:
                map_name = helper.get_map_name_from_sample_token(sample_token)
                # nusc_map = NuScenesMap(dataroot=data_path, map_name=map_name)
            except KeyError:
                map_name = None

            # env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization, dt=dt)
            # env.robot_type = env.NodeType.VEHICLE

            gt_env = Environment(node_type_list=['VEHICLE', 'PEDESTRIAN'], standardization=standardization, dt=dt)
            gt_env.robot_type = gt_env.NodeType.VEHICLE

            # scene = Scene(timesteps=1, dt=env.dt, name=sample_token,
            #               x_min=0.0, y_min=0.0, map_name=map_name)
            gt_scene = Scene(timesteps=1, dt=gt_env.dt, name=sample_token,
                             x_min=0.0, y_min=0.0, map_name=map_name)

            sample = nusc.get('sample', sample_token)
            sample_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
            ego_data = nusc.get('ego_pose', sample_data['ego_pose_token'])

            ego_pos = ego_data['translation']
            ego_heading = Quaternion(ego_data['rotation']).yaw_pitch_roll[0]

            data_dict = {('position', 'x'): [ego_pos[0]],
                         ('position', 'y'): [ego_pos[1]],
                         ('heading', '°'): [ego_heading]}
            node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            node = Node(node_type=gt_env.NodeType.VEHICLE, 
                        node_id='ego', 
                        length=4,
                        width=1.7,
                        height=1.5,
                        data=node_data)
            node.first_timestep = 0
            node.is_robot = True

            # scene.robot = node
            # scene.nodes.append(node)
            gt_scene.robot = node
            gt_scene.nodes.append(node)

            # GT Detections
            annotation_tokens = sample['anns']
            for idx, annotation_token in enumerate(annotation_tokens):
                annotation = nusc.get('sample_annotation', annotation_token)
                category = annotation['category_name']
                gt_ado_pos = annotation['translation']
                gt_ado_heading = Quaternion(annotation['rotation']).yaw_pitch_roll[0]

                data_dict = {('position', 'x'): [gt_ado_pos[0]],
                             ('position', 'y'): [gt_ado_pos[1]],
                             ('heading', '°'): [gt_ado_heading]}
                node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

                node = Node(node_type=gt_env.NodeType.VEHICLE if 'vehicle' in category else gt_env.NodeType.PEDESTRIAN, 
                            node_id=annotation_token, 
                            length=annotation['size'][0],
                            width=annotation['size'][1],
                            height=annotation['size'][2],
                            data=node_data)
                gt_scene.nodes.append(node)

            # # MEGVII Detections
            # for idx, detection in enumerate(sample_detections):
            #     ado_pos = detection['translation']
            #     ado_heading = Quaternion(detection['rotation']).yaw_pitch_roll[0]

            #     data_dict = {('position', 'x'): [ado_pos[0]],
            #                  ('position', 'y'): [ado_pos[1]],
            #                  ('heading', '°'): [ado_heading]}
            #     node_data = pd.DataFrame(data_dict, columns=data_columns_vehicle)

            #     node = Node(node_type=env.NodeType.VEHICLE if detection['detection_name'] != 'pedestrian' else env.NodeType.PEDESTRIAN, 
            #                 node_id=str(idx), 
            #                 length=detection['size'][0],
            #                 width=detection['size'][1],
            #                 height=detection['size'][2],
            #                 data=node_data)
            #     scene.nodes.append(node)
            
            # env.scenes = [scene]
            gt_env.scenes = [gt_scene]

            # data_dict_path = os.path.join(output_path, f'megvii_dets_{count}.pkl')
            # with open(data_dict_path, 'wb') as f:
            #     dill.dump(env, f, protocol=4) # For Python 3.6 and 3.8 compatability.
            # print('Saved Megvii Environment!')

            data_dict_path = os.path.join(output_path, f'megvii_gt_{count}.pkl')
            with open(data_dict_path, 'wb') as f:
                dill.dump(gt_env, f, protocol=4) # For Python 3.6 and 3.8 compatability.
            print('Saved GT Environment!')

            count += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--version', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    process_data(args.data, args.version, args.output_path)
