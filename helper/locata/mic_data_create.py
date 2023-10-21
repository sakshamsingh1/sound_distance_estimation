'''
Locata Data

FS: 48kHz -> 24kHz
Channels 32 channels -> [6, 10, 26, 22]
Type: float64 -> int16
'''

# imports
import numpy as np
import os
import scipy.io.wavfile as wav
from scipy import signal
from tqdm import tqdm 

#constants
read_dir = '/vast/sk8974/experiments/dsynth/data/util_data/locata_processed/mic_data'
save_dir = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/mic_dev/locata'
NEW_SR = 24000
tetra_chan_inds = [5, 9, 25, 21]

#helper functions
def resample_and_saveTetra(file):
    rec_path = os.path.join(read_dir,file)
    save_path = os.path.join(save_dir,file)

    fs, audio = wav.read(rec_path)

    audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
    audio_new = audio_new[:,tetra_chan_inds]
    audio_new = (audio_new * np.iinfo(np.int16).max).astype(np.int16)
    wav.write(save_path, NEW_SR, audio_new)

recs = os.listdir(read_dir)
for rec in tqdm(recs):
    _ = resample_and_saveTetra(rec)