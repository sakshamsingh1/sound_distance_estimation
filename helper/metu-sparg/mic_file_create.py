'''
About Metu data
All wav files are of 2 secs.
Sample rate = 48kHz

Ran this script on audioClip
'''

# imports
import numpy as np
import os
import scipy.io.wavfile as wav
from scipy import signal
from tqdm import tqdm 

#constants
read_dir = '/vast/sk8974/experiments/dsynth/data/util_data/metu-sparg/em32'
save_dir = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/mic_dev/metu'
NEW_SR = 24000
FILE_LEN_S = 2
tetra_chan_inds = [5, 9, 25, 21]

#helper functions

def resample_and_combine(dir):
    comb_wav = np.zeros((NEW_SR*FILE_LEN_S,32)) 
    save_file = os.path.basename(dir)+'.wav'
    save_path = os.path.join(save_dir,save_file)

    for i in range(1, 33):
        file = f"IR000{i:02d}.wav"
        file = os.path.join(dir,file)
        fs, audio = wav.read(file)
        audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
        comb_wav[:,i-1] = audio_new
    audio_new = comb_wav[:,tetra_chan_inds]
    wav.write(save_path, NEW_SR, audio_new.astype(np.int16))
    return comb_wav

recs = os.listdir(read_dir)
for rec in tqdm(recs):
    rec_file = os.path.join(read_dir,rec)
    _ = resample_and_combine(rec_file)





