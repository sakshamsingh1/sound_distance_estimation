import os
import scipy.io.wavfile as wav
from glob import glob
from tqdm import tqdm

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

#test : fold10_
marco_test_files = [ 
    "-15deg_065_Eigenmike_Raw_32ch",
    "-30deg_065_Eigenmike_Raw_32ch",
    "-45deg_065_Eigenmike_Raw_32ch",
    "-60deg_065_Eigenmike_Raw_32ch"
    ]
#val fold17_
marco_val_files = [
    "Acapella_Eigenmike_Raw_32ch_aug3",
    "Organ_065_Eigenmike_Raw_32ch_aug4",
    "Organ_065_Eigenmike_Raw_32ch_aug6",
    "Organ_065_Eigenmike_Raw_32ch_aug8",
    "Piano1_065_Eigenmike_Raw_32ch_aug2",
    "Quartet_065_Eigenmike_Raw_32ch_aug7"
    ]



def marco_aug_data(meta_dir, mic_dir):
    ''' Reference : A Four-Stage Data Augmentation Approach to ResNet-Conformer Based Acoustic Modeling for Sound Event Localization and Detection
        https://arxiv.org/pdf/2101.02919.pdf
        NOTE: This augmentation function is only for the distance experiments. For the angle experiments, we need to update the code.'''

    aug_files = [
    'Acappella_Eigenmike_Raw_32ch.wav',
    'Organ_065_Eigenmike_Raw_32ch.wav',
    'Piano1_065_Eigenmike_Raw_32ch.wav',
    'Piano2_065_Eigenmike_Raw_32ch.wav',
    'Quartet_065_Eigenmike_Raw_32ch.wav'
    ]

    aug_list = {}
    aug_list['aug1'] = [2, 4, 1, 3]
    aug_list['aug2'] = [4, 2, 3, 1]
    aug_list['aug3'] = [1, 2, 3, 4]
    aug_list['aug4'] = [2, 1, 4, 3]
    aug_list['aug5'] = [3, 1, 4, 2]
    aug_list['aug6'] = [1, 3, 2, 4]
    aug_list['aug7'] = [4, 3, 2, 1]
    aug_list['aug8'] = [3, 4, 1, 2]

    for file_path in tqdm(glob(os.path.join(mic_dir, '*.wav'))):

        mic_file = os.path.basename(file_path)
        meta_file = mic_file.replace('.wav', '.csv')

        if mic_file in aug_files:
            fs, audio = wav.read(file_path)
            for aug in aug_list:
                chan_seq = aug_list[aug]
                chan_seq = [i - 1 for i in chan_seq]
                audio_out = audio[:, chan_seq]
                new_file = os.path.basename(file_path).replace('.wav', '_' + aug + '.wav')
                new_file_path = os.path.join(mic_dir, new_file)
                wav.write(new_file_path, fs, audio_out)

                # copy the metadata file
                meta_file = os.path.basename(file_path).replace('.wav', '.csv')
                meta_file_path = os.path.join(meta_dir, meta_file)
                new_meta_file_path = os.path.join(meta_dir, new_file.replace('.wav', '.csv'))
                os.system('cp ' + meta_file_path + ' ' + new_meta_file_path)
            os.remove(os.path.join(meta_dir, meta_file))
            os.remove(os.path.join(mic_dir, mic_file))