import cls_data_generator
import parameters
import seldnet_model
import torch
import torch.nn as nn
import torch.optim as optim
from cls_compute_seld_results import reshape_3Dto2D
import os

#helper function
def train_epoch(data_generator, optimizer, model, criterion,  device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    for data, target in data_generator.generate():
        # load one batch of data
        data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()
        optimizer.zero_grad()

        # process the batch of data based on chosen mode
        output = model(data)

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        print(f'batch:loss = {nb_train_batches}:{train_loss}')
        nb_train_batches += 1

    train_loss /= nb_train_batches
    return train_loss

def write_file( _output_format_file, _output_format_dict):
    
    _fid = open(_output_format_file, 'w')
    for _frame_ind in _output_format_dict.keys():
        for _value in _output_format_dict[_frame_ind]:
            
            _fid.write('{},{},{}\n'.format(int(_frame_ind), float(_value[0]), float(_value[1])))
    _fid.close()

def test_epoch_onlyDist(data_generator, model, dcase_output_folder, params, device):
    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            output = model(data)
            output = reshape_3Dto2D(output)
            # import pdb; pdb.set_trace()

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}

            for frame_cnt in range(output.shape[0]):
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                
                output_dict[frame_cnt].append([output[frame_cnt][0],output[frame_cnt][1]])
            write_file(output_file, output_dict)

            nb_test_batches += 1
    return test_loss


params = parameters.get_params('35')
params['batch_size'] = 32
use_cuda = torch.cuda.is_available()
params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'

val_splits = [[2,4,6,8,10]]
train_splits = [[1,3,5,7,9]]

data_gen_train = cls_data_generator.DataGenerator(
    params=params, split=train_splits[0]
)
data_gen_val = cls_data_generator.DataGenerator(
    params=params, split=val_splits[0], shuffle=False, per_file=True
)

device = torch.device("cuda" if use_cuda else "cpu")

data_in, data_out = data_gen_train.get_data_sizes()
model = seldnet_model.CRNN(data_in, data_out, params).to(device)
model.load_state_dict(torch.load(params['pretrained_model_weights'], map_location='cpu'))

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=params['lr'])

dcase_output_folder = '/vast/sk8974/experiments/dsynth/scripts/seld_run/revamp_seld/seld-dcase2022_dist/temp/'
test_epoch_onlyDist(data_gen_val, model, dcase_output_folder, params, device)
    # print(f'output.shape = {output.shape}')


# train_epoch(data_gen_train, optimizer, model, criterion,  device)

