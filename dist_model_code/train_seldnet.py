#
# A wrapper script that trains the SELDnet. The training stops when the early stopping metric - SELD error stops improving.
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plot
import cls_feature_class
import cls_data_generator
import seldnet_model
import parameters
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
plot.switch_backend('agg')
from IPython import embed
from cls_compute_seld_results import ComputeSELDResults, reshape_3Dto2D, compute_dist_metrics, compute_dist_metrics_perm3, compute_dist_metrics_extended_perm3
from SELD_evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model

def get_accdoa_labels(accdoa_in, nb_classes):
    x, y, z = accdoa_in[:, :, :nb_classes], accdoa_in[:, :, nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:]
    sed = np.sqrt(x**2 + y**2 + z**2) > 0.5

    return sed, accdoa_in


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    """
    Args:
        accdoa_in:  [batch_size, frames, num_track*num_axis*num_class=3*3*12]
        nb_classes: scalar
    Return:
        sedX:       [batch_size, frames, num_class=12]
        doaX:       [batch_size, frames, num_axis*num_class=3*12]
    """
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 3*nb_classes:4*nb_classes], accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes]
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 3*nb_classes: 6*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 6*nb_classes:7*nb_classes], accdoa_in[:, :, 7*nb_classes:8*nb_classes], accdoa_in[:, :, 8*nb_classes:]
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 6*nb_classes:]

    return sed0, doa0, sed1, doa1, sed2, doa2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
                                                  doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    # Number of frames for a 60 second audio with 100ms hop length = 600 frames
    # Number of frames in one batch (batch_size* sequence_length) consists of all the 600 frames above with zero padding in the remaining frames
    test_filelist = data_generator.get_filelist()

    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            # load one batch of data
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            # process the batch of data based on chosen mode
            output = model(data)
            loss = criterion(output, target)
            if params['multi_accdoa'] is True:
                sed_pred0, doa_pred0, sed_pred1, doa_pred1, sed_pred2, doa_pred2 = get_multi_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred0 = reshape_3Dto2D(sed_pred0)
                doa_pred0 = reshape_3Dto2D(doa_pred0)
                sed_pred1 = reshape_3Dto2D(sed_pred1)
                doa_pred1 = reshape_3Dto2D(doa_pred1)
                sed_pred2 = reshape_3Dto2D(sed_pred2)
                doa_pred2 = reshape_3Dto2D(doa_pred2)
            else:
                sed_pred, doa_pred = get_accdoa_labels(output.detach().cpu().numpy(), params['unique_classes'])
                sed_pred = reshape_3Dto2D(sed_pred)
                doa_pred = reshape_3Dto2D(doa_pred)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}
            if params['multi_accdoa'] is True:
                for frame_cnt in range(sed_pred0.shape[0]):
                    for class_cnt in range(sed_pred0.shape[1]):
                        # determine whether track0 is similar to track1
                        flag_0sim1 = determine_similar_location(sed_pred0[frame_cnt][class_cnt], sed_pred1[frame_cnt][class_cnt], doa_pred0[frame_cnt], doa_pred1[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_1sim2 = determine_similar_location(sed_pred1[frame_cnt][class_cnt], sed_pred2[frame_cnt][class_cnt], doa_pred1[frame_cnt], doa_pred2[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        flag_2sim0 = determine_similar_location(sed_pred2[frame_cnt][class_cnt], sed_pred0[frame_cnt][class_cnt], doa_pred2[frame_cnt], doa_pred0[frame_cnt], class_cnt, params['thresh_unify'], params['unique_classes'])
                        # unify or not unify according to flag
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            if sed_pred0[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred1[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                            if sed_pred2[frame_cnt][class_cnt]>0.5:
                                if frame_cnt not in output_dict:
                                    output_dict[frame_cnt] = []
                                output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            if flag_0sim1:
                                if sed_pred2[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred2[frame_cnt][class_cnt], doa_pred2[frame_cnt][class_cnt+params['unique_classes']], doa_pred2[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_1sim2:
                                if sed_pred0[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred0[frame_cnt][class_cnt], doa_pred0[frame_cnt][class_cnt+params['unique_classes']], doa_pred0[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                            elif flag_2sim0:
                                if sed_pred1[frame_cnt][class_cnt]>0.5:
                                    output_dict[frame_cnt].append([class_cnt, doa_pred1[frame_cnt][class_cnt], doa_pred1[frame_cnt][class_cnt+params['unique_classes']], doa_pred1[frame_cnt][class_cnt+2*params['unique_classes']]])
                                doa_pred_fc = (doa_pred2[frame_cnt] + doa_pred0[frame_cnt]) / 2
                                output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 >= 2:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            doa_pred_fc = (doa_pred0[frame_cnt] + doa_pred1[frame_cnt] + doa_pred2[frame_cnt]) / 3
                            output_dict[frame_cnt].append([class_cnt, doa_pred_fc[class_cnt], doa_pred_fc[class_cnt+params['unique_classes']], doa_pred_fc[class_cnt+2*params['unique_classes']]])
            else:
                for frame_cnt in range(sed_pred.shape[0]):
                    for class_cnt in range(sed_pred.shape[1]):
                        if sed_pred[frame_cnt][class_cnt]>0.5:
                            if frame_cnt not in output_dict:
                                output_dict[frame_cnt] = []
                            output_dict[frame_cnt].append([class_cnt, doa_pred[frame_cnt][class_cnt], doa_pred[frame_cnt][class_cnt+params['unique_classes']], doa_pred[frame_cnt][class_cnt+2*params['unique_classes']]])
            data_generator.write_output_format_file(output_file, output_dict)

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break


        test_loss /= nb_test_batches

    return test_loss

def test_epoch_onlyDist(data_generator, model, criterion, dcase_output_folder, params, device):

    test_filelist = data_generator.get_filelist()
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0

    perm_2 = params['permutation_2']
    perm_3 = params['permutation_3']

    with torch.no_grad():
        for data, target in data_generator.generate():
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            output = model(data)
            loss = criterion(output, target)
            output = reshape_3Dto2D(output)

            # dump SELD results to the correspondin file
            output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
            file_cnt += 1
            output_dict = {}

            for frame_cnt in range(output.shape[0]):
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                if perm_2 or perm_3:
                    output_dict[frame_cnt].append([output[frame_cnt][0], output[frame_cnt][1]])
                else:
                    output_dict[frame_cnt].append([output[frame_cnt][0]])
            data_generator.write_output_format_file_onlyDist(output_file, output_dict, dist_and_mask=(perm_2 or perm_3))

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss

def test_epoch_onlyDist_noWrite(data_generator, model, criterion, params, device):
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    with torch.no_grad():
        for data, target in data_generator.generate():
            data, target = torch.tensor(data).to(device).float(), torch.tensor(target).to(device).float()

            output = model(data)
            loss = criterion(output, target)
            output = reshape_3Dto2D(output)

            # dump SELD results to the correspondin file
            file_cnt += 1
            output_dict = {}

            for frame_cnt in range(output.shape[0]):
                if frame_cnt not in output_dict:
                    output_dict[frame_cnt] = []
                output_dict[frame_cnt].append([output[frame_cnt][0]])

            test_loss += loss.item()
            nb_test_batches += 1
            if params['quick_test'] and nb_test_batches == 4:
                break

        test_loss /= nb_test_batches

    return test_loss



def train_epoch(data_generator, optimizer, model, criterion, params, device):
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
        nb_train_batches += 1
        if params['quick_test'] and nb_train_batches == 4:
            break

    train_loss /= nb_train_batches

    return train_loss


def main(argv):
    """
    Main wrapper for training sound event localization and detection network.

    :param argv: expects two optional inputs.
        first input: task_id - (optional) To chose the system configuration in parameters.py.
                                (default) 1 - uses default parameters
        second input: job_id - (optional) all the output files will be uniquely represented with this.
                              (default) 1

    """
    print(argv)
    if len(argv) != 3:
        print('\n\n')
        print('-------------------------------------------------------------------------------------------------------')
        print('The code expected two optional inputs')
        print('\t>> python seld.py <task-id> <job-id>')
        print('\t\t<task-id> is used to choose the user-defined parameter set from parameter.py')
        print('Using default inputs for now')
        print('\t\t<job-id> is a unique identifier which is used for output filenames (models, training plots). '
              'You can use any number or string for this.')
        print('-------------------------------------------------------------------------------------------------------')
        print('\n\n')

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)

    # use parameter set defined by user
    task_id = '1' if len(argv) < 2 else argv[1]
    params = parameters.get_params(task_id)

    job_id = 1 if len(argv) < 3 else argv[-1]

    # Training setup
    train_splits, val_splits, test_splits = None, None, None
    if params['mode'] == 'dev':
        '''
        dcase synth : fold 1, 2 
        dcase stars : fold 3, 4
        locata: fold 5(orig eval), 6(orig dev)
        metu: fold 7, 8 Train vs test [146, 98]
        marco: fold 9, 10
        '''
        if params['use_ind_data']:
            test_splits = [params['ind_data_train_test_split'][2]]
            val_splits = [params['ind_data_train_test_split'][1]]
            train_splits = [params['ind_data_train_test_split'][0]]

        # elif params['use_all_data']:
        #     test_splits = [[2,4,6,8,10]]
        #     val_splits = [[2,4,6,8,10]]
        #     train_splits = [[1,3,5,7,9]]

        # elif params['synth_and_real_dcase']:
        #     test_splits = [[2,4]]
        #     val_splits = [[2,4]]
        #     train_splits = [[1,3]]

        # elif params['train_synth_test_synth']:
        #     test_splits = [[2]]
        #     val_splits = [[2]]
        #     train_splits = [[1]]

        # elif '2020' in params['dataset_dir']:
        #     test_splits = [1]
        #     val_splits = [2]
        #     train_splits = [[3, 4, 5, 6]]

        # elif '2021' in params['dataset_dir']:
        #     test_splits = [6]
        #     val_splits = [5]
        #     train_splits = [[1, 2, 3, 4]]

        # elif '2022' in params['dataset_dir']:
        #     test_splits = [[4]]
        #     val_splits = [[4]]
        #     train_splits = [[1, 2, 3]]

        else:
            print('ERROR: Unknown dataset splits')
            exit()
    for split_cnt, split in enumerate(test_splits):
        print('\n\n---------------------------------------------------------------------------------------------------')
        print('------------------------------------      SPLIT {}   -----------------------------------------------'.format(split))
        print('---------------------------------------------------------------------------------------------------')

        # Unique name for the run
        loc_feat = params['dataset']
        if params['dataset'] == 'mic':
            if params['use_salsalite']:
                loc_feat = '{}_salsa'.format(params['dataset'])
            else:
                loc_feat = '{}_gcc'.format(params['dataset'])
        loc_output = 'multiaccdoa' if params['multi_accdoa'] else 'accdoa'

        cls_feature_class.create_folder(params['model_dir'])
        unique_name = '{}_{}_{}_split{}_{}_{}'.format(
            task_id, job_id, params['mode'], split_cnt, loc_output, loc_feat
        )
        model_name = '{}_model.h5'.format(os.path.join(params['model_dir'], unique_name))
        print("unique_name: {}\n".format(unique_name))

        # Load train and validation data
        print('Loading training dataset:')
        data_gen_train = cls_data_generator.DataGenerator(
            params=params, split=train_splits[split_cnt]
        )

        print('Loading validation dataset:')
        data_gen_val = cls_data_generator.DataGenerator(
            params=params, split=val_splits[split_cnt], shuffle=False, per_file=True
        )

        # data_gen_val_dcase, data_gen_val_stars, data_gen_val_loc, data_gen_val_metu, data_gen_val_marco = None, None, None, None, None
        # if params['use_all_data']:
        #     data_gen_val_dcase = cls_data_generator.DataGenerator(
        #         params=params, split=[2], shuffle=False, per_file=True
        #     )

        #     data_gen_val_stars = cls_data_generator.DataGenerator(
        #         params=params, split=[4], shuffle=False, per_file=True
        #     )

        #     data_gen_val_loc = cls_data_generator.DataGenerator(
        #         params=params, split=[6], shuffle=False, per_file=True
        #     )

        #     data_gen_val_metu = cls_data_generator.DataGenerator(
        #         params=params, split=[8], shuffle=False, per_file=True
        #     )

        #     data_gen_val_marco = cls_data_generator.DataGenerator(
        #         params=params, split=[10], shuffle=False, per_file=True
        #     )

        # if params['synth_and_real_dcase']:
        #     data_gen_val_stars = cls_data_generator.DataGenerator(
        #         params=params, split=[4], shuffle=False, per_file=True
        #     )

        #     data_gen_val_dcase = cls_data_generator.DataGenerator(
        #         params=params, split=[2], shuffle=False, per_file=True
        #     )

        # Collect i/o data size and load model configuration
        data_in, data_out = data_gen_train.get_data_sizes()
        model = seldnet_model.CRNN(data_in, data_out, params).to(device)
        if params['finetune_mode']:
            print('Running in finetuning mode. Initializing the model to the weights - {}'.format(params['pretrained_model_weights']))
            model.load_state_dict(torch.load(params['pretrained_model_weights'], map_location='cpu'))

        print('---------------- SELD-net -------------------')
        print('FEATURES:\n\tdata_in: {}\n\tdata_out: {}\n'.format(data_in, data_out))
        print('MODEL:\n\tdropout_rate: {}\n\tCNN: nb_cnn_filt: {}, f_pool_size{}, t_pool_size{}\n\trnn_size: {}, fnn_size: {}\n'.format(
            params['dropout_rate'], params['nb_cnn2d_filt'], params['f_pool_size'], params['t_pool_size'], params['rnn_size'],
            params['fnn_size']))
        print(model)

        # Dump results in DCASE output format for calculating final scores
        dcase_output_val_folder = os.path.join(params['dcase_output_dir'], '{}_{}_val'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_val_folder)
        print('Dumping recording-wise val results in: {}'.format(dcase_output_val_folder))

        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)

        # start training
        best_val_epoch = -1
        best_ER, best_F, best_LE, best_LR, best_seld_scr = 1., 0., 180., 0., 9999
        patience_cnt, epoch_cnt = 0, 0
        best_val_loss, best_train_loss = 9999, 9999
        best_val_mae = 9999
        best_val_gen = 9999
        best_epoch = -1

        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        if params['multi_accdoa'] is True:
            criterion = seldnet_model.MSELoss_ADPIT()
        else:
            # if params['permutation_1']:
            #     criterion = seldnet_model.perm_1()
            #     criterion_val = seldnet_model.perm_1(loss_type='mae')

            # elif params['permutation_2']:
            #     criterion = seldnet_model.perm_2(params['perm_2_loss_type'], only_mask=params['perm_2_onlyMask'], thr=params['perm_2_loss_mpe_type_thr'])
            #     criterion_val = seldnet_model.perm_2(loss_type='mae', only_mask=params['perm_2_onlyMask'])

            # elif params['permutation_3']:
            criterion = seldnet_model.perm_3(params['perm_3_loss_type'], only_mask=params['perm_3_onlyMask'], thr=params['perm_3_loss_mpe_type_thr'])
            criterion_val = seldnet_model.perm_3(loss_type='mae', only_mask=params['perm_3_onlyMask'])

            # else:
            #     criterion = nn.MSELoss()

        while patience_cnt < params['patience']:
            # ---------------------------------------------------------------------
            # TRAINING
            # ---------------------------------------------------------------------
            start_time = time.time()
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device)
            train_time = time.time() - start_time

            # ---------------------------------------------------------------------
            # VALIDATION
            # ---------------------------------------------------------------------
            start_time = time.time()

            if (params['perm_2_onlyMask'] or params['perm_3_onlyMask']):
                val_loss = test_epoch_onlyDist(data_gen_val, model, criterion_val, dcase_output_val_folder, params, device)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(model.state_dict(), model_name)
                    patience_cnt = 0
                    best_epoch = epoch_cnt
                else:
                    patience_cnt += 1

                print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, mask_val_loss : {round(val_loss, 3)}, best_epoch: {best_epoch}')

            elif params['only_dist']:
                val_loss = test_epoch_onlyDist(data_gen_val, model, criterion_val, dcase_output_val_folder, params, device)
                if params['permutation_3']:
                    agg_metrics = compute_dist_metrics_perm3(params, val_splits[split_cnt], ref_files_folder=dcase_output_val_folder)
                    val_gt_mae, val_gt_mpe, val_pred_mae = agg_metrics['global']['gt_mae'], agg_metrics['global']['gt_mpe'], agg_metrics['global']['overall_mae']

                    if params['perm_3_loss_type'] in ['mae','mse']:
                        val_gen = val_gt_mae
                    else:
                        val_gen = val_gt_mpe

                    if val_gen < best_val_gen:
                        best_val_gen = val_gen
                        torch.save(model.state_dict(), model_name)
                        patience_cnt = 0
                        best_epoch = epoch_cnt
                    else:
                        patience_cnt += 1

                    # if params['use_ind_data']:

                    print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, '
                            f'val_loss, val_gt_mae: '
                            f'{round(val_loss, 3)}, {round(val_gt_mae, 3)}')

                    # elif params['use_all_data']:
                    #     val_mae_dcase = agg_metrics['fold']['D']['mae']
                    #     val_mae_stars = agg_metrics['fold']['S']['mae']
                    #     val_mae_loc = agg_metrics['fold']['L']['mae']
                    #     val_mae_metu = agg_metrics['fold']['Me']['mae']
                    #     val_mae_marco = agg_metrics['fold']['Ma']['mae']

                    #     print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, '
                    #           f'val_loss, val_mae, val_mpe | D/S/L/Me/Ma '
                    #           f'{round(val_loss, 3)}, {round(val_mae, 3)}, {round(val_mpe, 3)} | '
                    #           f'{round(val_mae_dcase, 3)}/{round(val_mae_stars, 3)}/{round(val_mae_loc, 3)}/{round(val_mae_metu, 3)}/{round(val_mae_marco, 3)}')

                else:
                    agg_metrics = compute_dist_metrics(params, val_splits[split_cnt], ref_files_folder=dcase_output_val_folder)
                    val_mae, val_mpe = agg_metrics['global']['mae'], agg_metrics['global']['mpe']

                    # check for early stopping on training loss if overfit = True and save model for best val loss
                    if params['overfit']:
                        if train_loss < best_train_loss:
                            best_train_loss = train_loss
                            patience_cnt = 0
                        else:
                            patience_cnt += 1

                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            torch.save(model.state_dict(), model_name)
                    else:

                        if val_mae < best_val_mae:
                            best_val_mae = val_mae
                            torch.save(model.state_dict(), model_name)
                            patience_cnt = 0
                        else:
                            patience_cnt += 1

                    # if params['use_ind_data']:

                    print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, '
                            f'val_loss, val_mae, val_mpe : '
                            f'{round(val_loss, 3)}, {round(val_mae, 3)}, {round(val_mpe, 3)}')

                    # elif params['use_all_data']:
                    #     val_mae_dcase = agg_metrics['fold']['D']['mae']
                    #     val_mae_stars = agg_metrics['fold']['S']['mae']
                    #     val_mae_loc = agg_metrics['fold']['L']['mae']
                    #     val_mae_metu = agg_metrics['fold']['Me']['mae']
                    #     val_mae_marco = agg_metrics['fold']['Ma']['mae']

                    #     print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 3)}, '
                    #           f'val_loss, val_mae, val_mpe | D/S/L/Me/Ma '
                    #           f'{round(val_loss, 3)}, {round(val_mae, 3)}, {round(val_mpe, 3)} | '
                    #           f'{round(val_mae_dcase, 3)}/{round(val_mae_stars, 3)}/{round(val_mae_loc, 3)}/{round(val_mae_metu, 3)}/{round(val_mae_marco, 3)}')

                    # elif params['synth_and_real_dcase']:
                    #     val_loss_real = test_epoch_onlyDist_noWrite(data_gen_val_stars, model, criterion, params, device)
                    #     val_loss_dcase = test_epoch_onlyDist_noWrite(data_gen_val_dcase, model, criterion, params, device)
                    #     print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss, 4)}, val_loss:T/R/D {round(val_loss, 4)}/{round(val_loss_real, 4)}/{round(val_loss_dcase, 4)}')

                    # else:
                    #     print(f'Epoch: {epoch_cnt}, train_loss: {round(train_loss,4)}, val_loss: {round(val_loss,4)}')

            # else:
            #     val_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)

            #     # Calculate the DCASE 2021 metrics - Location-aware detection and Class-aware localization scores
            #     val_ER, val_F, val_LE, val_LR, val_seld_scr, classwise_val_scr = score_obj.get_SELD_Results(dcase_output_val_folder)

            #     val_time = time.time() - start_time

            #     # Save model if loss is good
            #     if val_seld_scr <= best_seld_scr:
            #         best_val_epoch, best_ER, best_F, best_LE, best_LR, best_seld_scr = epoch_cnt, val_ER, val_F, val_LE, val_LR, val_seld_scr
            #         torch.save(model.state_dict(), model_name)

            #     # Print stats
            #     print(
            #         'epoch: {}, time: {:0.2f}/{:0.2f}, '
            #         # 'train_loss: {:0.2f}, val_loss: {:0.2f}, '
            #         'train_loss: {:0.4f}, val_loss: {:0.4f}, '
            #         'ER/F/LE/LR/SELD: {}, '
            #         'best_val_epoch: {} {}'.format(
            #             epoch_cnt, train_time, val_time,
            #             train_loss, val_loss,
            #             '{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}'.format(val_ER, val_F, val_LE, val_LR, val_seld_scr),
            #             best_val_epoch, '({:0.2f}/{:0.2f}/{:0.2f}/{:0.2f}/{:0.2f})'.format(best_ER, best_F, best_LE, best_LR, best_seld_scr))
            #     )

            #     patience_cnt += 1
            #     if patience_cnt > params['patience']:
            #         break
            epoch_cnt += 1

        # ---------------------------------------------------------------------
        # Evaluate on unseen test data
        # ---------------------------------------------------------------------
        print('Load best model weights')
        model.load_state_dict(torch.load(model_name, map_location='cpu'))

        print('Loading unseen test dataset:')
        data_gen_test = cls_data_generator.DataGenerator(
            params=params, split=test_splits[split_cnt], shuffle=False, per_file=True
        )

        # Dump results in DCASE output format for calculating final scores
        dcase_output_test_folder = os.path.join(params['dcase_output_dir'],
                                                '{}_{}_test'.format(unique_name, strftime("%Y%m%d%H%M%S", gmtime())))
        cls_feature_class.delete_and_create_folder(dcase_output_test_folder)
        print('Dumping recording-wise test results in: {}'.format(dcase_output_test_folder))

        # if params['only_dist']:
        test_loss = test_epoch_onlyDist(data_gen_test, model, criterion_val, dcase_output_test_folder, params, device)

        agg_metrics = compute_dist_metrics_extended_perm3(params, test_splits[split_cnt], ref_files_folder=dcase_output_test_folder)
        val_gt_mae, val_gt_median, val_gt_std = agg_metrics['global']['gt_mae'], agg_metrics['global']['gt_median'], agg_metrics['global']['gt_std']
        print(f'test_loss, test_mae, test_median, test_std : {round(test_loss, 3)}, {round(val_gt_mae, 3)}, {round(val_gt_median, 3)}, {round(val_gt_std, 3)}')

        # else:
        #     test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device)

        #     use_jackknife=True
        #     test_ER, test_F, test_LE, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(dcase_output_test_folder, is_jackknife=use_jackknife )
        #     print('\nTest Loss')
        #     print('SELD score (early stopping metric): {:0.2f} {}'.format(test_seld_scr[0] if use_jackknife else test_seld_scr, '[{:0.2f}, {:0.2f}]'.format(test_seld_scr[1][0], test_seld_scr[1][1]) if use_jackknife else ''))
        #     print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(test_ER[0]  if use_jackknife else test_ER, '[{:0.2f}, {:0.2f}]'.format(test_ER[1][0], test_ER[1][1]) if use_jackknife else '', 100* test_F[0]  if use_jackknife else 100* test_F, '[{:0.2f}, {:0.2f}]'.format(100* test_F[1][0], 100* test_F[1][1]) if use_jackknife else ''))
        #     print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(test_LE[0] if use_jackknife else test_LE, '[{:0.2f} , {:0.2f}]'.format(test_LE[1][0], test_LE[1][1]) if use_jackknife else '', 100*test_LR[0]  if use_jackknife else 100*test_LR,'[{:0.2f}, {:0.2f}]'.format(100*test_LR[1][0], 100*test_LR[1][1]) if use_jackknife else ''))
        #     if params['average']=='macro':
        #         print('Classwise results on unseen test data')
        #         print('Class\tER\tF\tLE\tLR\tSELD_score')
        #         for cls_cnt in range(params['unique_classes']):
        #             print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
        #                  cls_cnt,
        #                  classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0], classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
        #                  classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0], classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
        #                  classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0], classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
        #                  classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0], classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
        #                  classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt], '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0], classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))



if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)

