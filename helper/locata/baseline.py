import os
import numpy as np

locata_data = '/vast/sk8974/experiments/dsynth/data/util_data/base_loc/metadata_dev/locata'
files = os.listdir(locata_data)

def load_file(_output_format_file):
    _output_dict = {}
    _fid = open(_output_format_file, 'r')
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

task_3_train = []
task_3_test = []

for file in files:
    if int(file.split('_')[0][4:]) == 12:
        task_3_test.append(file)
    elif int(file.split('_')[0][4:]) == 13:
        task_3_train.append(file)

def get_mean_median(file_list):
    dists = []
    for file in file_list:
        path = os.path.join(locata_data,file)
        data = load_file(path)
        for _, val in data.items():
            dists.append(val[0][4])
    dist_a = np.array(dists)
    return np.mean(dist_a), np.median(dist_a), dist_a

test_mean, test_median, dists = get_mean_median(task_3_test)

diff = np.mean(np.abs(dists-test_mean))

np.std(np.abs(dists-test_mean))