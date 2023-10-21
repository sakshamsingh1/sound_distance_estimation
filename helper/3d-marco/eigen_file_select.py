from glob import glob
import os
import shutil

path_dir = '/vast/sk8974/experiments/dsynth/data/util_data/3d_marco'
save_dir = '/vast/sk8974/experiments/dsynth/data/util_data/3d_marco_processed'

ins_dir = os.listdir(path_dir)

ins_dirs = ['Acappella','Organ','Piano solo 1','Piano solo 2','Quartet']
ins_dir_2 = ['Single sources at different positions']

eigen_files = []

for ins_dir in ins_dirs:
    files = glob(f'{path_dir}/{ins_dir}/*wav')

    for file in files:
        if 'Eigenmike' in file:
            eigen_files.append(file)

files = glob(f'{path_dir}/{ins_dir_2[0]}/*/*wav')

for file in files:
        if 'Eigenmike' in file:
            eigen_files.append(file)

for src in eigen_files:
    file = os.path.basename(src)
    dst = os.path.join(save_dir,file)
    shutil.copyfile(src, dst)            