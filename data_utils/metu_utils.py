# metu data balancing
# 1. augment all the data : bith mic and meta

# imports
import os
import scipy.io.wavfile as wav
from glob import glob
from tqdm import tqdm


def metu_aug(read_dir, write_mic_dir, write_meta_dir):
    read_meta_dir = os.path.join(read_dir,'temp_meta_data') 
    read_mic_dir = os.path.join(read_dir,'temp_mic_data')

    aug_list = {}
    aug_list['aug1'] = [2, 4, 1, 3]
    aug_list['aug2'] = [4, 2, 3, 1]
    aug_list['aug3'] = [1, 2, 3, 4]
    aug_list['aug4'] = [2, 1, 4, 3]
    aug_list['aug5'] = [3, 1, 4, 2]
    aug_list['aug6'] = [1, 3, 2, 4]
    aug_list['aug7'] = [4, 3, 2, 1]
    aug_list['aug8'] = [3, 4, 1, 2]

    train_fold = [7]

    for file_path in tqdm(glob(os.path.join(read_mic_dir, '*.wav'))):

        mic_file = os.path.basename(file_path)
        meta_file = mic_file.replace('.wav', '.csv')

        if int(mic_file.split('_')[0][4:]) in train_fold:
            fs, audio = wav.read(file_path)
            for aug in aug_list:
                chan_seq = aug_list[aug]
                chan_seq = [i - 1 for i in chan_seq]
                audio_out = audio[:, chan_seq]
                new_file = os.path.basename(file_path).replace('.wav', '_' + aug + '.wav')
                new_file_path = os.path.join(write_mic_dir, new_file)
                wav.write(new_file_path, fs, audio_out)
        
                # copy the metadata file
                meta_file = os.path.basename(file_path).replace('.wav', '.csv')
                meta_file_path = os.path.join(read_meta_dir, meta_file)
                new_meta_file_path = os.path.join(write_meta_dir, new_file.replace('.wav', '.csv'))
                os.system('cp ' + meta_file_path + ' ' + new_meta_file_path)
        
        else:
            #just copy the file
            os.system('cp ' + file_path + ' ' + write_mic_dir)
            
            meta_file_path = os.path.join(read_meta_dir, meta_file)
            new_meta_file_path = os.path.join(write_meta_dir, meta_file)
            os.system('cp ' + meta_file_path + ' ' + new_meta_file_path)