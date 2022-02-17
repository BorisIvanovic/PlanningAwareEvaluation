import os
import time
import numpy as np
import pandas as pd

from pathlib import Path


class DataLogger(object):
    def __init__(self, col_names, filepath='./log.csv', data_dir='./', auto_save_every=1):
        # Always have a timestamp
        if 'timestamp' not in col_names:
            col_names = ['timestamp'] + col_names

        self.filepath = filepath
        self.data_dir = data_dir
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)

        self.num_additions = 0
        self.auto_save_every = auto_save_every
        self.iterable_columns = [col_name for col_name in col_names if col_name != 'timestamp']
        self.internal_dict = {col: [] for col in col_names}
        self.indices_dict = dict()


    def add_rows(self, cols_values_dict, update_indices=list()):
        # Add a new row with updated values in the columns
        # cols_values_dict is a dict containing cols with 
        # updated values.
        max_len = max([len(cols_values_dict[col]) for col in cols_values_dict])

        # First extend to size
        for col in cols_values_dict:
            if len(cols_values_dict[col]) < max_len:
                cols_values_dict[col].extend([cols_values_dict[col][-1]] * (max_len - len(cols_values_dict[col])))

        # Then add all
        for col in self.iterable_columns:
            if col in cols_values_dict:
                self.internal_dict[col].extend(cols_values_dict[col])
            
            elif col in update_indices:
                for itr in range(max_len):
                    self.internal_dict[col].append(self.indices_dict[col] + itr)
            
            elif col in self.indices_dict:
                # Just extend it with the current last value.
                self.internal_dict[col].extend([self.indices_dict[col]]*max_len)
            
            else:
                # Just extend it with the current last value.
                self.internal_dict[col].extend([self.internal_dict[col][-1]]*max_len)

        self.internal_dict['timestamp'].extend([time.time()]*max_len)
        
        self.num_additions += 1
        if self.num_additions % self.auto_save_every == 0:
            self.save_to_file()


    def update_indices(self, idx_dict):
        for idx in idx_dict:
            self.indices_dict[idx] = idx_dict[idx]

    def increment(self, idx):
        self.indices_dict[idx] += 1

    def save_to_npy(self, file_prefix, data):
        if len(data) == 0:
            raise ValueError('Received a 0-length list!')

        # Assumes that data is a list of np array
        for index in sorted(self.indices_dict.keys()):
            file_prefix += '_' + index + '-' + str(int(self.indices_dict[index]))

        print('Saving a %d-length list of %s-shape elements to %s.' % (len(data), str(data[0].shape), file_prefix), flush=True)
        stacked_data = np.stack(data)
        np.save(os.path.join(self.data_dir, file_prefix), stacked_data, allow_pickle=False)
        print('Saved the resulting %s-shape array to %s.' % (str(stacked_data.shape), file_prefix), flush=True)


    def save_to_file(self):
        data_df = pd.DataFrame.from_dict(self.internal_dict)
        data_df.to_csv(self.filepath, index=False)
        # print('Saved performance data to %s' % self.filepath, flush=True)


if __name__ == '__main__':
    data_logger = DataLogger(['iter', 'ppo_perf', 'ppo_lens', 'ppo_rews'], 'test_data.csv')
    
    data_logger.update_indices({'iter': 0})
    pct_successful = 0.50
    ep_mean_lens = [11, 12, 13, 14]*2
    ep_mean_rews = [0.2, 0.3, 0.4, 0.5]*2
    data_logger.add_rows({'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews})

    data_logger.update_indices({'iter': 1})
    pct_successful = 0.50
    ep_mean_lens = [11, 12, 13, 14]*2
    ep_mean_rews = [0.2, 0.3, 0.4, 0.5]*2
    data_logger.add_rows({'ppo_perf': [pct_successful], 'ppo_lens': ep_mean_lens, 'ppo_rews': ep_mean_rews})

    data_logger.save_to_file()
