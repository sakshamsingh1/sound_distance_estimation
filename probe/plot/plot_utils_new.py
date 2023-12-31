import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from glob import glob
import concurrent.futures
plt.rcParams["figure.figsize"] = (8,5)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#helpers
def load_output_format_file(_output_format_file):
    """
    Loads DCASE output format csv file and returns it in dictionary format

    :param _output_format_file: DCASE output format CSV
    :return: _output_dict: dictionary
    """
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
        if len(_words) == 2:  # only dist
            _output_dict[_frame_ind].append([float(_words[1])])
        if len(_words) == 3:  # dist + pred_mask
            _output_dict[_frame_ind].append([float(_words[1]), float(_words[2])])
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
    _fid.close()
    return _output_dict

def get_df_from_dict(pred_file, gt_dir):        
    df = pd.DataFrame(columns=['gt_dist', 'pred_dist', 'fold', 'dist_error', 'dist_abs_error'])
    
    gt_file = os.path.basename(pred_file).split('.')[0]+'.npy'
    gt = np.load(os.path.join(gt_dir,gt_file))
    pred = load_output_format_file(pred_file)

    nb_ref_frames = gt.shape[0]

    for frame in range(nb_ref_frames):
        gt_dist = gt[frame][0]
        if gt_dist==0:
            continue
        
        gt_dist = gt[frame][0]
        pred_dist = pred[frame][0][0]

        dist_error = gt_dist - pred_dist
        dist_abs_error = abs(dist_error)
        fold = gt_file.split('_')[0]

        df = df.append({'gt_dist': gt_dist, 'pred_dist':pred_dist, 'dist_error':dist_error, 'fold':fold, 'dist_abs_error':dist_abs_error}, ignore_index=True)
    return df

def plot_scatter(df, title='', col='pred_dist'):
    marker_size = 0.1
    plt.scatter(df['gt_dist'],df[col],s=marker_size)
    x, y = df['gt_dist'], df[col]
    plt.xlabel('gt_dist')
    plt.ylabel(col)
    # m, b = np.polyfit(x, y, 1)
    # plt.plot(x, m*x+b, color='red')
    plt.title(f'Param_no : {title}')
    plt.show()

def box_plot(df, title='', col='pred_dist', bin_size = 0.5):
    
    bins = np.arange(0, df['gt_dist'].max() + bin_size, bin_size)
    groups = df.groupby(pd.cut(df['gt_dist'], bins))
    # import pdb; pdb.set_trace()

    boxplot_data = [group[col] for _, group in groups]
    bins = np.around(bins, decimals=1)
    plt.boxplot(boxplot_data, labels=bins[:-1])
    plt.xlabel('gt_dist')
    plt.ylabel(col)
    # plt.ylim(0, 1.5)
    plt.title(f'Param_no : {title}')
    plt.show()


def create_error_df(pred_dir, gt_dir):
    pred_files = glob(f'{pred_dir}/*.csv')

    df_glob = pd.DataFrame(columns=['gt_dist', 'pred_dist', 'fold','dist_error', 'dist_abs_error'])
    for pred_file in tqdm(pred_files):
        # import pdb; pdb.set_trace()
        df = get_df_from_dict(pred_file, gt_dir)
        df_glob = df_glob.append(df,ignore_index=True)
    return df_glob

def create_error_df_parallel(pred_dir, gt_dir):
    pred_files = glob(f'{pred_dir}/*.csv')
    df_glob = pd.DataFrame(columns=['gt_dist', 'pred_dist', 'fold', 'dist_error', 'dist_abs_error'])
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(get_df_from_dict, pred_file, gt_dir) for pred_file in pred_files]
        
        results = [future.result() for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures))]
        
    for df in results:
        df_glob = df_glob.append(df, ignore_index=True)

    return df_glob


#Run example| copy and paste the code below in a notebook

# import sys
# path = '/experiments/dsynth/scripts/helper/plot/'
# sys.path.append(path)
# from plot_utils import *
# gt_dir = 'mic_dev_label'
# pred_dir = '29_1_dev_split0_accdoa_mic_gcc_20230321085140_test'
# df=create_error_df(pred_dir, gt_dir)
# df_zero = df[df['gt_dist']!=0].copy().reset_index()
# _ = plot_error_dist(df_zero[(df_zero['fold']=='fold5')|(df_zero['fold']=='fold6')], title='locata data | LR=1e-3')
# df_zero.groupby(['fold'])[['dist_abs_error']].mean()

#Run example| SCATTER PLOT

# import matplotlib.pyplot as plt
# plt.rcParams["figure.figsize"] = (4,3)
# %matplotlib inline
# def plot_scatter(df, title=''):
#     marker_size = 0.1
#     plt.scatter(df['gt_dist'],df['pred_dist'],s=marker_size)
#     x, y = df['gt_dist'], df['pred_dist']
#     plt.xlabel('gt_dist')
#     plt.ylabel('pred_dist')
#     m, b = np.polyfit(x, y, 1)
#     plt.plot(x, m*x+b, color='red')
#     plt.title(f'Param_no : {title}')
#     plt.show()

# import sys
# path = 'scripts/helper/plot/'
# sys.path.append(path)
# from plot_utils import *

# gt_dir = 'processed/dist_d_s_l_m_m/mic_dev_label'

# curr_param = '106_1_dev_split0_accdoa_mic_gcc_20230407163748_test'
# pred_dir = f'experiments/dsynth/scripts/seld_run/run/results/{curr_param}'
# df=create_error_df_parallel(pred_dir, gt_dir)

# plot_scatter(df, title=curr_param.split('_')[0])