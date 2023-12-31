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

    for frame in range(gt.shape[0]):
        gt_dist = gt[frame][0]
        if frame in pred:
            pred_dist = pred[frame][0][0]*pred[frame][0][1]
        else:
            pred_dist = 0.0

        dist_error = gt_dist - pred_dist
        dist_abs_error = abs(dist_error)
        fold = gt_file.split('_')[0]

        df = df.append({'gt_dist': gt_dist, 'pred_dist':pred_dist, 'dist_error':dist_error, 'fold':fold, 'dist_abs_error':dist_abs_error}, ignore_index=True)
    return df

def plot_error_dist(df1, err_col = 'dist_abs_error', title = 'CRNN model'):
    gt_dist_bins = [i for i in range(-1, 10)]
    error_bins = [i for i in range(-1, 7)]
    cmap = ListedColormap(plt.cm.get_cmap('RdGy')(np.linspace(0.2, 1, len(error_bins))))
    err_bin_str = ['(1, 2]','(2, 3]','(3, 4]','(0, 1]','(-1, 0]','(4, 5]','(5, 6]']

    cols = ['gt_dist']
    cols.append(err_col)
    df = df1.copy()[cols]
    df['gt_dist_bin'] = pd.cut(df['gt_dist'], gt_dist_bins)
    df['error_bin'] = pd.cut(df[err_col], error_bins)
    df['error_bin'] = df['error_bin'].astype(str)
    df['gt_dist_bin'] = df['gt_dist_bin'].astype(str)
    
    df = df[['error_bin','gt_dist_bin']] 

    for i in err_bin_str:
        new_row = {'error_bin': i, 'gt_dist_bin': '(5, 6]'}
        df = df.append(new_row,ignore_index=True) 

    df_error_dist = df.groupby(['gt_dist_bin', 'error_bin']).size().unstack(fill_value=0)
    
    df_error_dist.plot(kind='bar', stacked=True, colormap=cmap)
    plt.title(f'{title} (err dist)')
    plt.show()
    
    # add total count of bins to gt_dist_bin column to the x axis labels
    df_error_dist['total'] = df_error_dist.sum(axis=1)
    df_error_dist['total'] = df_error_dist['total'].astype(str)
    df_error_dist['gt_dist_bin'] = df_error_dist.index
    df_error_dist['gt_dist_bin'] = df_error_dist['gt_dist_bin'] + ' (N= ' + df_error_dist['total'] + ')'
    df_error_dist = df_error_dist.drop(columns=['total'])
    df_error_dist = df_error_dist.set_index('gt_dist_bin')
    
    df_error_dist_norm = df_error_dist.div(df_error_dist.sum(axis=1), axis=0)

    # plot the stacked bars with the colormap
    df_error_dist_norm.plot(kind='bar', stacked=True, colormap=cmap)
    plt.title(f'{title} (norm err dist)')
    plt.show()

    return df_error_dist

def create_scatter_plot(df):
    marker_size = 0.1
    plt.scatter(df['gt_dist'],df['dist_error'],s=marker_size)
    plt.xlabel('gt_dist')
    plt.ylabel('prediction error(gt-pr)')
    plt.title('DCASE model')
    plt.axhline(y=0, color='red', linewidth=1, linestyle='--')
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
# path = 'helper/plot/'
# sys.path.append(path)
# from plot_utils import *
# gt_dir = 'dist_dcase_locata_starss/mic_dev_label'
# pred_dir = 'results/29_1_dev_split0_accdoa_mic_gcc_20230321085140_test'
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
# path = 'helper/plot/'
# sys.path.append(path)
# from plot_utils import *

# gt_dir = 'data/processed/mic_dev_label'

# curr_param = '106_1_dev_split0_accdoa_mic_gcc_20230407163748_test'
# pred_dir = f'run/results/{curr_param}'
# df=create_error_df_parallel(pred_dir, gt_dir)

# plot_scatter(df, title=curr_param.split('_')[0])