import os
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

def load_output_format_file(_output_format_file):
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
              
        if len(_words) == 2: #polar coordinates 
            _output_dict[_frame_ind].append([float(_words[1])])
        if len(_words) == 3: #polar coordinates 
            _output_dict[_frame_ind].append([float(_words[1]),float(_words[2])])
        if len(_words) == 5: #polar coordinates 
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4])])
        elif len(_words) == 6: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5])])
        elif len(_words) == 7: # cartesian coordinates
            _output_dict[_frame_ind].append([int(_words[1]), int(_words[2]), float(_words[3]), float(_words[4]), float(_words[5]), float(_words[6])])    
    _fid.close()
    return _output_dict

def get_df_from_dict(pred_file, gt_dir):        
    df = pd.DataFrame(columns=['gt_mask', 'pred_mask', 'fold'])
    
    gt_file = os.path.basename(pred_file).split('.')[0]+'.npy'
    gt = np.load(os.path.join(gt_dir,gt_file))
    pred = load_output_format_file(pred_file)

    for frame in range(gt.shape[0]):
        gt_mask = 0
        if gt[frame][0]>0:
            gt_mask = 1
        if frame in pred:
            pred_mask = pred[frame][0][1]
        else:
            pred_mask = 0.0

        fold = gt_file.split('_')[0]

        df = df.append({'gt_mask': gt_mask, 'pred_mask':pred_mask, 'fold':fold}, ignore_index=True)
    return df

def create_error_df(pred_dir, gt_dir):
    pred_files = glob(f'{pred_dir}/*.csv')

    df_glob = pd.DataFrame(columns=['gt_mask', 'pred_mask', 'fold'])
    for pred_file in tqdm(pred_files):
        df = get_df_from_dict(pred_file, gt_dir)
        df_glob = df_glob.append(df,ignore_index=True)
    return df_glob

gt_dir = ''
pred_dir = ''
df = create_error_df(pred_dir, gt_dir)

from sklearn.metrics import recall_score, f1_score, roc_auc_score, precision_score

y_true = np.array(list(df['gt_mask']))
y_scores = np.array(list(df['pred_mask']))
thrs = [0.3,0.6,0.9]

for thr in thrs:
    p = precision_score(y_true, y_scores >= thr)
    r = recall_score(y_true, y_scores >= thr)
    f1 = f1_score(y_true, y_scores >= thr)
    auc = roc_auc_score(y_true, y_scores)
    print(f'{thr}:{round(p,2)} {round(r,2)} {round(f1,2)} {round(auc,2)}')