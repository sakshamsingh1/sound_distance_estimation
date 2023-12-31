import os
from glob import glob 
import numpy as np
from tqdm import tqdm

#helper 
def load_file(_output_format_file):

    _output_dict = {}
    _fid = open(_output_format_file, 'r')
    # next(_fid)
    for _line in _fid:
        _words = _line.strip().split(',')
        _frame_ind = int(_words[0])
        if _frame_ind not in _output_dict:
            _output_dict[_frame_ind] = []
    _fid.close()
    return _output_dict

mic_data_dir = "data/input/gen_dcase_stars_loc_metu_marco/metadata_dev"
files = glob(f'{mic_data_dir}/**/*.csv') 

fold_map = {}
for file in files:
    fold = int(os.path.basename(file).split('_')[0][4:])
    if fold not in fold_map :
        fold_map[fold]=1
    else:
        fold_map[fold]+=1

val_splits = [2,4,6,8,10]
train_splits = [1,3,5,7,9]     

train, val = 0, 0
for k, v in fold_map.items():
    if k in train_splits:
        train+=v
    else:
        val+=v

#num of hours
def convert_100ms_hr_min(num_ms):
    milliseconds = num_ms*100

    # Convert milliseconds to seconds
    seconds = milliseconds / 1000

    # Convert seconds to minutes
    minutes = seconds / 60

    # Convert minutes to hours and remaining minutes
    hours = int(minutes / 60)
    remaining_minutes = int(minutes % 60)
    return hours, remaining_minutes        

fold_hours = {}
for file in files:
    fold = int(os.path.basename(file).split('_')[0][4:])
    data = load_file(file)
    k = np.array(list(data.keys()))
    fmax = np.max(k)
    if fold not in fold_hours:
        fold_hours[fold]=0
    fold_hours[fold]+= fmax

for k, v in fold_hours.items():
    hr, m = convert_100ms_hr_min(v)
    print(f'{k} : {hr}hrs {m}min')    

sum_ = 0
for k, v in fold_hours.items():
    sum_ += v
hr, m = convert_100ms_hr_min(sum_)
print(f'{hr}hrs {m}min')      