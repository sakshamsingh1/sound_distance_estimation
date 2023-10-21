'''
train vs test
dcase synth : fold 1, 2 
dcase stars : fold 3, 4
locata: fold 5(orig eval), 6(orig dev)
metu: fold 7, 8 Train vs test [146, 98]
marco: fold 9, 10
'''

#imports
import os

#marco
metu_fold = {'test':'fold8_', 'train':'fold7_'}

metu_meta = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/metadata_dev/metu'
metu_mic = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco/mic_dev/metu'

def is_train(file):
    x,y,z = int(file[0]), int(file[1]), int(file[2])
    if z-2>=0:
        return True

for dir_type in [metu_mic]:
    for file in os.listdir(dir_type):
        if not is_train(file):
            src = os.path.join(dir_type,file)
            dst = os.path.join(dir_type,metu_fold['test']+file)
            os.rename(src,dst)
        else:
            src = os.path.join(dir_type,file)
            dst = os.path.join(dir_type,metu_fold['train']+file)
            os.rename(src,dst)