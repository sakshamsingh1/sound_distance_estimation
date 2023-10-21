'''
train vs test
dcase synth : fold 1, 2 
dcase stars : fold 3, 4
locata: fold 5(orig eval), 6(orig dev)
metu: fold 7, 8
marco: fold 9, 10
'''

#imports
import os

#marco
marco_fold = {'test':'fold10_', 'train':'fold9_'}

test_files =  {
'-15deg_065_Eigenmike_Raw_32ch',
'-30deg_065_Eigenmike_Raw_32ch',
'-45deg_065_Eigenmike_Raw_32ch',
'-60deg_065_Eigenmike_Raw_32ch',
'-75deg_065_Eigenmike_Raw_32ch',
'-90deg_065_Eigenmike_Raw_32ch',
'0deg_065_Eigenmike_Raw_32ch'
}

train_files = {
    'Acappella_Eigenmike_Raw_32ch',
    'Organ_065_Eigenmike_Raw_32ch',
    'Piano1_065_Eigenmike_Raw_32ch',
    'Piano2_065_Eigenmike_Raw_32ch',
    'Quartet_065_Eigenmike_Raw_32ch'
}

marco_meta = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/metadata_dev/marco'
marco_mic = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/mic_dev/marco'

for dir_type in [marco_meta, marco_mic]:
    for file in os.listdir(dir_type):
        if file.split('.')[0] in test_files:
            src = os.path.join(dir_type,file)
            dst = os.path.join(dir_type,marco_fold['test']+file)
            os.rename(src,dst)
        elif file.split('.')[0] in train_files:
            src = os.path.join(dir_type,file)
            dst = os.path.join(dir_type,marco_fold['train']+file)
            os.rename(src,dst)





