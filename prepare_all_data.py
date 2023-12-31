import os
from tqdm import tqdm 
from glob import glob
import scipy.io.wavfile as wav
from scipy import signal
import numpy as np
import shutil
import csv
import pickle

from data_utils.starss_utils import *
from data_utils.locata_utils import Locata2DecaseFormat, mic_util
from data_utils.marco_utils import file_dist_dict, marco_aug_data, marco_test_files, marco_val_files
from data_utils.metu_utils import metu_aug

import warnings
warnings.filterwarnings("ignore")

# TODO: Make the paths relative

DOWNLOAD_PATH = 'data/download'

def get_links(dataset):
    links = {}
    if dataset == 'starss':
        links = {
            'metadata': 'https://zenodo.org/records/7880637/files/metadata_dev.zip',
            'micdata': 'https://zenodo.org/records/7880637/files/mic_dev.zip'
        }
    elif dataset == 'dcase':
        links = {
            'all_data': 'https://huggingface.co/datasets/sakshamsingh1/sound_distance/resolve/main/dcase.zip'
        }
    elif dataset == 'locata':
        links = {
            'dev': 'https://zenodo.org/records/3630471/files/dev.zip',
            'eval': 'https://zenodo.org/records/3630471/files/eval.zip'
        }
    elif dataset == 'marco':
        links = {
            'organ': 'https://zenodo.org/records/3477602/files/06%203D-MARCo%20Samples_Organ.zip',
            'paino_1': 'https://zenodo.org/records/3477602/files/07%203D-MARCo%20Samples_Piano%20solo%201.zip',
            'paino_2': 'https://zenodo.org/records/3477602/files/08%203D-MARCo%20Samples_Piano%20solo%202.zip',
            'acapella': 'https://zenodo.org/records/3477602/files/09%203D-MARCo%20Samples_Acappella.zip',
            'Quartet': 'https://zenodo.org/records/3477602/files/04%203D-MARCo%20Samples_Quartet.zip',
        }
    elif dataset == 'metu':
        links = {
            'all_data': 'https://zenodo.org/records/2635758/files/spargair.zip' 
        }
    return links

def get_cmd(link, save_dir):
    cmd = f'wget {link} -P {save_dir}'
    return cmd

def aug_data(meta_dir, mic_dir, aug_folds):
    ''' Reference : A Four-Stage Data Augmentation Approach to ResNet-Conformer Based Acoustic Modeling for Sound Event Localization and Detection
        https://arxiv.org/pdf/2101.02919.pdf
        NOTE: This augmentation function is only for the distance experiments. For the angle experiments, we need to update the code.'''

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

        if int(mic_file.split('_')[0][4:]) in aug_folds:
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

############################# STARSS #############################
def download_starss(save_path):
    print('Downloading STARSS dataset')
    links = get_links('starss')
    for link_type,link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)
    
def unzip_starss(save_path):
    print('Unzipping STARSS dataset')
    os.system(f'unzip {save_path}/metadata_dev.zip -d {save_path}')
    os.system(f'unzip {save_path}/mic_dev.zip -d {save_path}')
    os.system(f'rm {save_path}/metadata_dev.zip')
    os.system(f'rm {save_path}/mic_dev.zip')

def mask_single_event():
    mic_dir = os.path.join(DOWNLOAD_PATH,'mic_dev')
    meta_dir = os.path.join(DOWNLOAD_PATH,'metadata_dev')

    save_meta_dir = 'data/input/metadata_dev/starss'
    save_mic_dir = 'data/input/mic_dev/starss'

    for fold in ['train','test']:
        print(f'Processing {fold} ....')

        mic_files = glob(f'{mic_dir}/*{fold}*/*.wav')
        mic_files = sorted(mic_files)
        meta_files = glob(f'{meta_dir}/*{fold}*/*.csv')
        meta_files = sorted(meta_files)

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

def rename_starss():
    print('Renaming STARSS dataset')
    save_meta_dir = 'data/input/metadata_dev/starss'
    save_mic_dir = 'data/input/mic_dev/starss'
    
    for file in starss_rename_list:
        file = file.split('.')[0]

        #metadata
        file_old = file + '.csv'
        file_new = 'fold15' + file[5:] + '.csv'
        os.system(f'mv {save_meta_dir}/{file_old} {save_meta_dir}/{file_new}')

        #micdata
        file_old = file + '.wav'
        file_new = 'fold15' + file[5:] + '.wav'
        os.system(f'mv {save_mic_dir}/{file_old} {save_mic_dir}/{file_new}')

def data_prep_starss():
    download_starss(DOWNLOAD_PATH)
    unzip_starss(DOWNLOAD_PATH)
    mask_single_event()
    rename_starss()

############################# DCASE #############################
def download_dcase(save_path):
    print('Downloading DCASE dataset')
    links = get_links('dcase')
    for link_type,link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_dcase(save_path):
    print('Unzipping DCASE dataset')
    os.system(f'unzip {save_path}/dcase.zip -d {save_path}')
    os.system(f'rm {save_path}/dcase.zip')

def move_dcase():
    curr_path = "data/download/zenodo_upload"
    meta_curr_path = os.path.join(curr_path,'meta_data')
    mic_curr_path = os.path.join(curr_path,'mic_data')

    meta_save_path = 'data/input/metadata_dev/dcase'
    mic_save_path = 'data/input/mic_dev/dcase'

    os.system(f'mv {meta_curr_path}/*.csv {meta_save_path}/')
    os.system(f'mv {mic_curr_path}/*.wav {mic_save_path}/')
    shutil.rmtree(curr_path)

def data_prep_dcase():
    download_dcase(DOWNLOAD_PATH)
    unzip_dcase(DOWNLOAD_PATH)
    move_dcase()

############################# LOCATA #############################

def download_locata(save_path):
    print('Downloading LOCATA dataset')
    links = get_links('locata')
    for link_type,link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_locata(save_path):
    print('Unzipping LOCATA dataset')
    os.system(f'unzip {save_path}/dev.zip -d {save_path}')
    os.system(f'unzip {save_path}/eval.zip -d {save_path}')
    os.system(f'rm {save_path}/dev.zip')
    os.system(f'rm {save_path}/eval.zip')

def get_loc_metadata():
    out_dir = 'data/input/metadata_dev/aug_locata'
    base_input_dir = 'data/download/'
    task_list = ["1","3","5"]
    splits = ['eval','dev']
    for split in splits:
        input_path = base_input_dir + split
        Locata2DecaseFormat(task_list, input_path, out_dir, arrays=["eigenmike"], is_dev=True, coord_system="polar")

def get_loc_mic_data():
    read_dir = 'data/download'
    save_dir = 'data/input/mic_dev/aug_locata'
    mic_util(read_dir, save_dir)

def augment_locata():
    print('Augmenting LOCATA dataset')
    mic_dir = 'data/input/mic_dev/aug_locata'
    meta_dir = 'data/input/metadata_dev/aug_locata'
    aug_folds = [11,13]
    aug_data(meta_dir, mic_dir, aug_folds)

def delete_locata():
    print('Deleting LOCATA dataset')
    shutil.rmtree(os.path.join(DOWNLOAD_PATH,'dev'))
    shutil.rmtree(os.path.join(DOWNLOAD_PATH,'eval'))

def data_prep_locata():
    download_locata(DOWNLOAD_PATH)
    unzip_locata(DOWNLOAD_PATH)
    get_loc_metadata()
    get_loc_mic_data()
    augment_locata()
    delete_locata()

##################################### MARCO #####################################
def download_marco(save_path):
    print('Downloading MARCO dataset')
    links = get_links('marco')
    for link_type,link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_marco(save_path):
    print('Unzipping MARCO dataset')
    ## Single source is supposed to be manually downloaded and unzipped in the same folder
    os.system(f'unzip {save_path}/06\ 3D-MARCo\ Samples_Organ.zip -d {save_path}')
    os.system(f'unzip {save_path}/07\ 3D-MARCo\ Samples_Piano\ solo\ 1.zip -d {save_path}')
    os.system(f'unzip {save_path}/08\ 3D-MARCo\ Samples_Piano\ solo\ 2.zip -d {save_path}')
    os.system(f'unzip {save_path}/09\ 3D-MARCo\ Samples_Acappella.zip -d {save_path}')
    os.system(f'unzip {save_path}/04\ 3D-MARCo\ Samples_Quartet.zip -d {save_path}')
    os.system(f'rm {save_path}/06\ 3D-MARCo\ Samples_Organ.zip')
    os.system(f'rm {save_path}/07\ 3D-MARCo\ Samples_Piano\ solo\ 1.zip')
    os.system(f'rm {save_path}/08\ 3D-MARCo\ Samples_Piano\ solo\ 2.zip')
    os.system(f'rm {save_path}/09\ 3D-MARCo\ Samples_Acappella.zip')
    os.system(f'rm {save_path}/04\ 3D-MARCo\ Samples_Quartet.zip')

def get_mic_macro(read_dir):
    save_dir = os.path.join(read_dir,'temp_mic_data') 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    ins_dirs = ['Acappella','Organ','Piano solo 1','Piano solo 2','Quartet']
    ins_dir_2 = ['Single sources at different positions']
    eigen_files = []

    for ins_dir in ins_dirs:
        files = glob(f'{read_dir}/{ins_dir}/*wav')
        for file in files:
            if 'Eigenmike' in file:
                eigen_files.append(file)

    files = glob(f'{read_dir}/{ins_dir_2[0]}/*/*wav')
    for file in files:
            if 'Eigenmike' in file:
                eigen_files.append(file)

    for src in eigen_files:
        file = os.path.basename(src)
        dst = os.path.join(save_dir,file)
        shutil.copyfile(src, dst)       

    for ins_dir in ins_dirs:
        ins_path = os.path.join(read_dir,ins_dir)
        shutil.rmtree(ins_path)
    
    shutil.rmtree(os.path.join(read_dir,ins_dir_2[0]))

def get_tetra_mic_macro(read_dir):
    mic_dir = os.path.join(read_dir,'temp_mic_data') 
    NEW_SR = 24000
    tetra_chan_inds = [5, 9, 25, 21]

    #helper functions
    def resample_and_saveTetra(file):
        rec_path = os.path.join(mic_dir,file)
        save_path = os.path.join(mic_dir,file)

        fs, audio = wav.read(rec_path)

        audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
        audio_new = audio_new[:,tetra_chan_inds]
        audio_new = audio_new.astype('float32')/np.iinfo(np.int32).max
        audio_new = (audio_new * 32767).astype('int16')
        wav.write(save_path, NEW_SR, audio_new)

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs):
        _ = resample_and_saveTetra(rec)

def get_meta_data_marco(read_dir):
    save_dir = os.path.join(read_dir,'temp_meta_data') 
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)

    mic_dir = os.path.join(read_dir,'temp_mic_data') 
    def make_csv(file):
        save_file = file.split('.')[0]+'.csv'
        save_path = os.path.join(save_dir,save_file)

        rec_file = os.path.join(mic_dir,file)
        fs, audio = wav.read(rec_file)
        
        fs *= 0.1
        num_pts = int(audio.shape[0]//fs)

        class_no = -1
        source_no = -1
        azimuth = -1
        elevation = -1

        r = file_dist_dict[file]

        with open(save_path, 'w') as f:
            writer = csv.writer(f)
            for i in range(num_pts):
                writer.writerow([i, class_no, source_no, azimuth, elevation, r])

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs):
        make_csv(rec)
    
def aug_data_marco(read_dir):
    print('Augmenting MARCO dataset')
    meta_dir = os.path.join(read_dir,'temp_meta_data') 
    mic_dir = os.path.join(read_dir,'temp_mic_data')
    marco_aug_data(meta_dir, mic_dir)

def rename_marco(read_dir):
    meta_dir = os.path.join(read_dir,'temp_meta_data') 
    mic_dir = os.path.join(read_dir,'temp_mic_data')

    save_meta_dir = 'data/input/metadata_dev/aug_marco'
    save_mic_dir = 'data/input/mic_dev/aug_marco'

    for file in os.listdir(meta_dir):
        file_pre = file.split('.')[0]
        if file_pre in marco_test_files:
            fold_pre = 'fold10_'
        elif file_pre in marco_val_files:
            fold_pre = 'fold17_'
        else:
            fold_pre = 'fold9_'

        #metadata
        file_old = file_pre + '.csv'
        file_new = fold_pre + file_pre + '.csv'
        os.system(f'mv {meta_dir}/{file_old} {save_meta_dir}/{file_new}')

        #micdata
        file_old = file_pre + '.wav'
        file_new = fold_pre + file_pre + '.wav'
        os.system(f'mv {mic_dir}/{file_old} {save_mic_dir}/{file_new}')

def del_temp_marco(read_dir):
    shutil.rmtree(os.path.join(read_dir,'temp_meta_data'))
    shutil.rmtree(os.path.join(read_dir,'temp_mic_data'))

def data_prep_marco():    
    download_marco(DOWNLOAD_PATH)
    unzip_marco(DOWNLOAD_PATH)
    get_mic_macro(DOWNLOAD_PATH)
    get_tetra_mic_macro(DOWNLOAD_PATH)
    get_meta_data_marco(DOWNLOAD_PATH)
    aug_data_marco(DOWNLOAD_PATH)
    rename_marco(DOWNLOAD_PATH)
    del_temp_marco(DOWNLOAD_PATH)

##################################### METU #####################################
    
def download_metu(save_path):
    print('Downloading METU dataset')
    links = get_links('metu')
    for link_type,link in links.items():
        print(f'Downloading {link_type}..........')
        wget_command = get_cmd(link, save_path)
        os.system(wget_command)

def unzip_metu(save_path):
    print('Unzipping METU dataset')
    os.system(f'unzip {save_path}/spargair.zip -d {save_path}')
    os.system(f'rm {save_path}/spargair.zip')

def get_metu_mic_data(read_dir):
    mic_dir = os.path.join(read_dir,'spargair/em32')
    temp_mic_dir = os.path.join(read_dir,'temp_mic_data')
    if os.path.exists(temp_mic_dir):
        shutil.rmtree(temp_mic_dir)
    os.mkdir(temp_mic_dir)
    
    NEW_SR = 24000
    FILE_LEN_S = 2
    tetra_chan_inds = [5, 9, 25, 21]

    #helper functions

    def resample_and_combine(dir):
        comb_wav = np.zeros((NEW_SR*FILE_LEN_S,32)) 
        save_file = os.path.basename(dir)+'.wav'
        save_path = os.path.join(temp_mic_dir,save_file)

        for i in range(1, 33):
            file = f"IR000{i:02d}.wav"
            file = os.path.join(dir,file)
            fs, audio = wav.read(file)
            audio_new = signal.resample(audio, int(len(audio) * float(NEW_SR) / fs))
            comb_wav[:,i-1] = audio_new
        audio_new = comb_wav[:,tetra_chan_inds]
        wav.write(save_path, NEW_SR, audio_new.astype(np.int16))
        return comb_wav

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs):
        rec_file = os.path.join(mic_dir,rec)
        _ = resample_and_combine(rec_file)

def get_metu_meta_data(read_dir):
    mic_dir = os.path.join(read_dir,'temp_mic_data')
    temp_meta_dir = os.path.join(read_dir,'temp_meta_data')
    if os.path.exists(temp_meta_dir):
        shutil.rmtree(temp_meta_dir)
    os.mkdir(temp_meta_dir)

    def make_csv(coor, save_file):
        save_file = save_file.replace('.wav', '.csv')
        save_path = os.path.join(temp_meta_dir,save_file)
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

    recs = os.listdir(mic_dir)
    for rec in tqdm(recs):
        x,y,z = int(rec[0]), int(rec[1]), int(rec[2])
        make_csv(coor_wrt_mic((x,y,z)),rec)

def metu_rename(read_dir):
    metu_fold = {'test':'fold8_', 'train':'fold7_'}

    metu_meta = os.path.join(read_dir,'temp_meta_data')
    metu_mic = os.path.join(read_dir,'temp_mic_data')

    def is_train(file):
        x,y,z = int(file[0]), int(file[1]), int(file[2])
        if z-2>=0:
            return True

    for dir_type in [metu_meta, metu_mic]:
        for file in os.listdir(dir_type):
            if not is_train(file):
                src = os.path.join(dir_type,file)
                dst = os.path.join(dir_type,metu_fold['test']+file)
                os.rename(src,dst)
            else:
                src = os.path.join(dir_type,file)
                dst = os.path.join(dir_type,metu_fold['train']+file)
                os.rename(src,dst)


def data_aug_metu(read_dir):
    print('Augmenting METU dataset')
    write_meta_dir = 'data/input/metadata_dev/aug_metu'
    write_mic_dir = 'data/input/mic_dev/aug_metu'
    metu_aug(read_dir, write_mic_dir, write_meta_dir)

def del_temp_data_metu(read_dir):
    shutil.rmtree(os.path.join(read_dir,'temp_meta_data'))
    shutil.rmtree(os.path.join(read_dir,'temp_mic_data'))
    shutil.rmtree(os.path.join(read_dir,'spargair'))

def rename_val_metu():
    meta_dir = 'data/input/metadata_dev/aug_metu'
    mic_dir = 'data/input/mic_dev/aug_metu'
    
    fold_files = pickle.load(open('data_utils/metu_val_files.pkl','rb'))
    fold_files = [i.split('.')[0] for i in fold_files]

    for file in os.listdir(meta_dir):
        if file.split('.')[0] in fold_files:
            src = os.path.join(meta_dir,file)
            dst = os.path.join(meta_dir,file.replace('fold7','fold16'))
            os.rename(src,dst)

    for file in os.listdir(mic_dir):
        if file.split('.')[0] in fold_files:
            src = os.path.join(mic_dir,file)
            dst = os.path.join(mic_dir,file.replace('fold7','fold16'))
            os.rename(src,dst)

def data_prep_metu():
    download_metu(DOWNLOAD_PATH)
    unzip_metu(DOWNLOAD_PATH)
    get_metu_mic_data(DOWNLOAD_PATH)
    get_metu_meta_data(DOWNLOAD_PATH)
    metu_rename(DOWNLOAD_PATH)
    data_aug_metu(DOWNLOAD_PATH)
    del_temp_data_metu(DOWNLOAD_PATH)
    rename_val_metu()

if __name__ == '__main__':
    data_prep_starss()
    data_prep_dcase()
    data_prep_locata()
    data_prep_marco()
    data_prep_metu()
