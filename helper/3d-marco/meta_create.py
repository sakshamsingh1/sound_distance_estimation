'''
Marco data : dcase format metadata creation
All files DO NOT have the same length

Meta data is creating by reading resampled mic files
'''
#imports
import numpy as np
import csv
import os
from tqdm import tqdm
import scipy.io.wavfile as wav

#constants
read_dir = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/mic_dev/marco'
save_dir = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/metadata_dev/marco'

file_dist_dict =  {
'-15deg_065_Eigenmike_Raw_32ch.wav': 4.0,
'-30deg_065_Eigenmike_Raw_32ch.wav': 3.0,
'-45deg_065_Eigenmike_Raw_32ch.wav': 4.0,
'-60deg_065_Eigenmike_Raw_32ch.wav': 3.0,
'-75deg_065_Eigenmike_Raw_32ch.wav': 4.0,
'-90deg_065_Eigenmike_Raw_32ch.wav': 3.0,
'0deg_065_Eigenmike_Raw_32ch.wav': 3.0,
'Acappella_Eigenmike_Raw_32ch.wav':2.6,
'Organ_065_Eigenmike_Raw_32ch.wav': 12.0,
'Piano1_065_Eigenmike_Raw_32ch.wav': 3.4,
'Piano2_065_Eigenmike_Raw_32ch.wav': 3.4,
'Quartet_065_Eigenmike_Raw_32ch.wav': 2.6
}

#helper
def make_csv(file):
    save_file = file.split('.')[0]+'.csv'
    save_path = os.path.join(save_dir,save_file)

    rec_file = os.path.join(read_dir,file)
    fs, audio = wav.read(rec_file)
    
    fs *= 0.1
    num_pts = int(audio.shape[0]//fs)
    # import pdb; pdb.set_trace()

    class_no = -1
    source_no = -1
    azimuth = -1
    elevation = -1

    r = file_dist_dict[file]

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for i in range(num_pts):
            writer.writerow([i, class_no, source_no, azimuth, elevation, r])

recs = os.listdir(read_dir)
for rec in tqdm(recs):
    make_csv(rec)