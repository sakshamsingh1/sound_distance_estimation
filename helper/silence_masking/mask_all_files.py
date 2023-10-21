import os
from tqdm import tqdm 
from glob import glob
import scipy.io.wavfile as wav
import numpy as np

from utils import *

base_dir = '/vast/sk8974/experiments/dsynth/data/util_data/stars22-23/'
mic_dir = os.path.join(base_dir,'mic_dev')
meta_dir = os.path.join(base_dir,'metadata_dev')

save_dir = '/vast/sk8974/experiments/dsynth/data/util_data/stars22-23_processed/'
save_meta_dir = os.path.join(save_dir,'metadata')
save_mic_dir = os.path.join(save_dir,'mic_data')

for fold in ['train','test']:
    print(f'Processing {fold} ....')

    mic_files = glob(f'{mic_dir}/*{fold}*/*.wav')
    meta_files = glob(f'{meta_dir}/*{fold}*/*.csv')

    for i, meta_file in tqdm(enumerate(meta_files)):
        meta_data = load_file(meta_file)
        mic_file = mic_files[i]

         #sanity check
        assert os.path.basename(meta_file).split('.')[0] == os.path.basename(mic_file).split('.')[0]

        ov_list = get_cons_ov_list(meta_data)
        zero_list = get_cons_zero_list(meta_data)

        fs, mic_data = wav.read(mic_file)
        mic_data = mic_data.T

        con_audio = None
        for interval in zero_list:
            this_audio = get_audio_chunk(interval, mic_data)
            smooth_audio = smooth_audio_edge(this_audio)
            noise_audio = add_normal_noise(smooth_audio)
            if con_audio is None:
                con_audio = noise_audio
            else:
                con_audio = np.concatenate((con_audio, noise_audio),axis=1)

        meta_save_path = os.path.join(save_meta_dir,os.path.basename(meta_file))
        is_non_empty = remove_ov_and_save_meta(meta_file, meta_save_path)

        if is_non_empty:
            mask_mic = mask_ov(mic_data, ov_list, con_audio)        
            mic_save_path = os.path.join(save_mic_dir,os.path.basename(mic_file))
            wav.write(mic_save_path, fs, mask_mic.T.astype(np.int16))

        

