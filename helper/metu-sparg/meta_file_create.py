'''
About Metu data
All wav files are of 2 secs.
Orig sample rate = 48kHz

Event is present only for the first sec
'''

#imports
import numpy as np
import csv
import os
from tqdm import tqdm

#constants
read_dir = '/vast/sk8974/experiments/dsynth/data/util_data/metu-sparg/em32'
save_dir = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/metadata_dev/metu'


#helper
def make_csv(coor, save_file):
    save_file += '.csv'
    save_path = os.path.join(save_dir,save_file)
    x,y,z = coor

    class_no = 0
    source_no = 0
    azimuth = np.arctan2(y, x) * 180 / np.pi
    azimuth = round(azimuth)
    
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2)) * 180 / np.pi
    elevation = round(elevation)

    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    r = round(r,2)

    with open(save_path, 'w') as f:
        writer = csv.writer(f)
        for i in range(10):
            writer.writerow([i, class_no, source_no, azimuth, elevation, r])

def coor_wrt_mic(coor):
    x,y,z = coor
    x_ = (3-x)*0.5
    y_ = (3-y)*0.5
    z_ = (2-z)*0.3
    return (x_, y_, z_)

recs = os.listdir(read_dir)
for rec in tqdm(recs):
    x,y,z = int(rec[0]), int(rec[1]), int(rec[2])
    make_csv(coor_wrt_mic((x,y,z)),rec)
                