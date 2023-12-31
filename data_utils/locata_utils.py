# Implementation adapted from https://github.com/audiofhrozen/locata_wrapper/blob/master/locata_wrapper/utils/process.py

from argparse import Namespace
import glob
import logging
import pandas as pd
import os
import csv
import numpy as np
import scipy.io.wavfile as wav
from scipy import signal
from tqdm import tqdm

from data_utils.load_metadata import GetTruth
from data_utils.load_metadata import LoadData

from matplotlib import pyplot as plt

fold_vs_files = {}
#fold 12
fold_vs_files['test'] = ['eval_task3_recording1',
                       'eval_task3_recording2',
                       'eval_task3_recording3',
                       'eval_task3_recording4',
                       'eval_task3_recording5']
#fold 13
fold_vs_files['val'] = ['dev_task3_recording1',
                          'dev_task3_recording2',
                          'dev_task3_recording3']

def get_fold_suffix(file):
    # file could be wav or csv
    file = file.split('.')[0]
    suffix = 'fold11_'
    if file in fold_vs_files['test']:
        suffix = 'fold12_'
    elif file in fold_vs_files['val']:
        suffix = 'fold13_'
    return suffix

def ElapsedTime(time_array):
    n_steps = time_array.shape[0]
    elapsed_time = np.zeros([n_steps])
    for i in range(1, n_steps):
        elapsed_time[i] = (time_array[i] - time_array[i - 1]).total_seconds()
    return np.cumsum(elapsed_time)

def ProcessTaskMetadata(this_task, arrays, data_dir, is_dev):
    task_dir = os.path.join(data_dir, 'task{}'.format(this_task))
    # Read all recording IDs available for this task:
    recordings = sorted(glob.glob(os.path.join(task_dir, '*')))
    print("recordings", recordings)
    truth_list = []
    # Parse through all recordings within this task:
    for this_recording in recordings:
        recording_id = int(this_recording.split('recording')[1])
        # Read all recording IDs available for this task:
        array_names = glob.glob(os.path.join(this_recording, '*'))
        for array_dir in array_names:
            this_array = os.path.basename(array_dir)
            if this_array not in arrays:
                continue
            print('Processing task {}, recording {}, array {}.'.format(this_task, recording_id, this_array))
            # Load metadata from csv
            position_array, position_source, required_time, vad_source = LoadData(
                array_dir, None, None, is_dev)
            print('Processing Complete!')
            # Extract ground truth
            # position_array stores all optitrack measurements.
            # Extract valid measurements only (specified by required_time.valid_flag).
            truth = GetTruth(this_array, position_array, position_source, vad_source, required_time, recording_id, is_dev)
            truth_list.append(truth)
    return truth_list

def Locata2DecaseFormat(tasks, data_dir, out_dir, arrays=["eigenmike"], is_dev=True, coord_system="cartesian"):
    FS_POS = 120 # Position labeling done at 120Hz
    FS_AUD = 48000 # Sampling rate of original audio
    split = os.path.basename(data_dir)
    for task_id in tasks:
        truth_list = ProcessTaskMetadata(task_id, arrays, data_dir, is_dev)
        for truth in truth_list:
            curr_file = f'{split}_task{task_id}_recording{truth.recording_id}.csv'
            suffix = get_fold_suffix(curr_file)
            curr_file = suffix + curr_file
            out_filename = f'{out_dir}/{curr_file}'
            with open(out_filename, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                print("Processing {}".format(out_filename))
                for iframe in range(0, truth.frames, FS_POS//10): # sample every 100msec
                    for speaker in truth.source:
                        frame_num = iframe//12
                        total_len = len(list(truth.vad[speaker].vad[0]))
                        upper_bnd = 0
                        if total_len < (frame_num+1)*(FS_AUD//10):
                            upper_bnd = total_len #len(list(truth.vad[speaker].vad[0])[(iframe//12)*(FS_AUD//10):((iframe//12)+1)*(FS_AUD//10)])
                        else:
                            upper_bnd = (frame_num+1)*(FS_AUD//10)
                        #print(speaker)
                        # Voting VAD to identify active speaker in a given frame
                        #print(truth.vad[speaker].vad[0].shape)
                        #print("frame", (iframe//12))
                        #print(len(list(truth.vad[speaker].vad[0])[(iframe//12)*(FS_AUD//10):((iframe//12)+1)*(FS_AUD//10)]))
                        spk_vad = list(truth.vad[speaker].vad[0])[frame_num*(FS_AUD//10):upper_bnd]
                        spk_active = np.mean(spk_vad) > 0.5 # active if more than 50% is 1 within the frame
                        if not spk_active:
                            continue
                        csv_row = [frame_num, 0, 0]
                        if coord_system == "cartesian":
                            csv_row.extend(truth.source[speaker].cart_pos[iframe])
                        elif coord_system == "polar":
                            # csv_row.extend(truth.source[speaker].polar_pos[iframe][:2]) # DCASE doesnt care about distance
                            csv_row.extend(truth.source[speaker].polar_pos[iframe][:3]) # But we do
                        csv_writer.writerow(csv_row)

def get_tetra_mic_data(rec_path, save_path):
    '''
    FS: 48kHz -> 24kHz
    Channels 32 channels -> [6, 10, 26, 22]
    Type: float64 -> int16
    '''
    NEW_SR = 24000
    tetra_chan_inds = [5, 9, 25, 21]
    fs, audio = wav.read(rec_path)

    audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
    audio_new = audio_new[:,tetra_chan_inds]
    audio_new = (audio_new * np.iinfo(np.int16).max).astype(np.int16)
    wav.write(save_path, NEW_SR, audio_new)

def mic_util(read_dir, save_dir):
    splits = ['eval','dev']
    mic = 'eigenmike'
    tasks = ["1","3","5"]

    for split in splits:
        for task in tasks:
            recordings = os.listdir(os.path.join(read_dir,split,f'task{task}'))
            for rec in recordings:
                rec_dir = os.path.join(read_dir,split,f'task{task}',rec,mic)
                if not os.path.exists(rec_dir):
                    continue
                rec_path = os.path.join(rec_dir,'audio_array_eigenmike.wav')
                save_file = f'{split}_task{task}_{rec}.wav'
                suffix = get_fold_suffix(save_file)
                save_file = suffix + save_file
                save_path = os.path.join(save_dir,save_file)
                get_tetra_mic_data(rec_path, save_path)


def aug_data():
    pass