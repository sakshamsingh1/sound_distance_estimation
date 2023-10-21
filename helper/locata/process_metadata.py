# Implementation adapted from https://github.com/audiofhrozen/locata_wrapper/blob/master/locata_wrapper/utils/process.py

from argparse import Namespace
import h5py
import glob
import logging
import numpy as np
import pandas as pd
import os
import timeit
import csv

from load_metadata import GetTruth
from load_metadata import LoadData

from matplotlib import pyplot as plt


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
            out_filename = f'{out_dir}/{split}_task{task_id}_recording{truth.recording_id}.csv'
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

out_dir = '/vast/sk8974/experiments/dsynth/data/util_data/locata_processed/metadata'
base_input_dir = '/vast/sk8974/experiments/dsynth/data/util_data/locata/'
task_list = ["1","3","5"]
splits = ['eval','dev']
for split in splits:
    input_path = base_input_dir + split
    Locata2DecaseFormat(task_list, input_path, out_dir, arrays=["eigenmike"], is_dev=True, coord_system="polar")
