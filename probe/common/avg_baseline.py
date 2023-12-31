import os
import numpy as np

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

locata_data = '/vast/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco/metadata_dev/aug_marco'
files = os.listdir(locata_data)
train_folds = [9,17]

train_files = []
test_files = []

for file in files:
    if int(file.split('_')[0][4:]) in train_folds:
        train_files.append(file)
    else:
        test_files.append(file)

print(len(train_files), len(test_files), len(files))

def get_dists(file_list):
    dists = []
    for file in file_list:
        path = os.path.join(locata_data,file)
        data = load_file(path)
        for _, val in data.items():
            dists.append(val[0][4])
            
    dist_a = np.array(dists)
    return dist_a

def get_avg_model_stats(train_files, test_files):
    train_mean = np.mean(get_dists(train_files))
    test_dists = get_dists(test_files)
    maes = np.abs(test_dists-train_mean)
    print(np.mean(maes), np.median(maes), np.std(maes))

get_avg_model_stats(train_files, test_files)        