#imports
import os
import numpy as np
from tqdm import tqdm

read_base_dir = '/vast/sk8974/experiments/dsynth/data/util_data/bal_and_aug_metu/aug_metu'
read_meta_dir = os.path.join(read_base_dir, 'metadata')
read_mic_dir = os.path.join(read_base_dir, 'mic')

write_base_dir = '/vast/sk8974/experiments/dsynth/data/util_data/bal_and_aug_metu/bal_metu'
write_meta_dir = os.path.join(write_base_dir, 'metadata')
write_mic_dir = os.path.join(write_base_dir, 'mic')

if not os.path.exists(write_meta_dir):
    os.makedirs(write_meta_dir)
if not os.path.exists(write_mic_dir):
    os.makedirs(write_mic_dir)

train_fold = [7]


def dist_wrt_mic(coor):
    x,y,z = coor
    x_ = (3-x)*0.5
    y_ = (3-y)*0.5
    z_ = (2-z)*0.3

    r = np.sqrt(x_ ** 2 + y_ ** 2 + z_ ** 2)
    r = round(r,2)
    return r

# get the distance of each recording
file_dist_dict = {}
mic_recs = os.listdir(read_mic_dir)
for rec in tqdm(mic_recs):
    if int(rec.split('_')[0][4:]) not in train_fold:
        continue
    rec_co = rec.split('_')[1]
    x,y,z = int(rec_co[0]), int(rec_co[1]), int(rec_co[2])
    r = dist_wrt_mic((x,y,z))
    file_dist_dict[rec] = r

#bin to files map
bin_to_files = {}
bin_size = 0.2
for file in file_dist_dict:
    dist = file_dist_dict[file]
    dist_bin = int(dist // bin_size) * bin_size

    if dist_bin not in bin_to_files:
        bin_to_files[dist_bin] = [file]
    else:
        bin_to_files[dist_bin].append(file)

# import pdb; pdb.set_trace()
bal_mic_files = []
for rec in tqdm(mic_recs):
    if int(rec.split('_')[0][4:]) not in train_fold:
        bal_mic_files.append(rec)

#from rach bin randomly select 64 files if num > 64 else select all
for bin, files in bin_to_files.items():
    if len(files) > 64:
        files = np.random.choice(files, 64, replace=False)
    bal_mic_files.extend(files)

#copy the files
for file in tqdm(bal_mic_files):
    os.system(f'cp {os.path.join(read_mic_dir, file)} {write_mic_dir}')

    #copy the metadata
    file = file.replace('.wav', '.csv')
    os.system(f'cp {os.path.join(read_meta_dir, file)} {write_meta_dir}')




