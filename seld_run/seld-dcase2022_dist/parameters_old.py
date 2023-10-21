# Parameters used in the feature extraction, neural network model, and training the SELDnet can be changed here.
#
# Ideally, do not change the values of the default parameters. Create separate cases with unique <task-id> as seen in
# the code below (if-else loop) and use them. This way you can easily reproduce a configuration on a later time.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters ##############
    params = dict(
        quick_test=True,  # To do quick test. Trains/test on small subset of dataset, and # of epochs

        finetune_mode=False,
        # Finetune on existing model, requires the pretrained model path set - pretrained_model_weights
        pretrained_model_weights='models/1_1_foa_dev_split6_model.h5',

        # INPUT PATH
        # dataset_dir='DCASE2020_SELD_dataset/',  # Base folder containing the foa/mic and metadata folders
        dataset_dir='/scratch/asignal/partha/DCASE2022_SELD_dataset',

        # OUTPUT PATHS
        # feat_label_dir='DCASE2020_SELD_dataset/feat_label_hnet/',  # Directory to dump extracted features and labels
        feat_label_dir='/scratch/asignal/partha/DCASE2022_SELD_dataset/seld_feat_label',

        model_dir='models/',  # Dumps the trained models and training curves in this folder
        dcase_output_dir='results/',  # recording-wise results are dumped in this path.

        # DATASET LOADING PARAMETERS
        mode='dev',  # 'dev' - development or 'eval' - evaluation dataset
        dataset='foa',  # 'foa' - ambisonic or 'mic' - microphone signals

        # FEATURE PARAMS
        fs=24000,
        hop_len_s=0.02,
        label_hop_len_s=0.1,
        max_audio_len_s=60,
        nb_mel_bins=64,

        use_salsalite=False,  # Used for MIC dataset only. If true use salsalite features, else use GCC features
        fmin_doa_salsalite=50,
        fmax_doa_salsalite=2000,
        fmax_spectra_salsalite=9000,

        # MODEL TYPE
        multi_accdoa=False,  # False - Single-ACCDOA or True - Multi-ACCDOA
        thresh_unify=15,  # Required for Multi-ACCDOA only. Threshold of unification for inference in degrees.

        # DNN MODEL PARAMETERS
        label_sequence_length=50,  # Feature sequence length
        batch_size=128,  # Batch size
        dropout_rate=0.05,  # Dropout rate, constant for all layers
        nb_cnn2d_filt=64,  # Number of CNN nodes, constant for each layer
        f_pool_size=[4, 4, 2],
        # CNN frequency pooling, length of list = number of CNN layers, list value = pooling per layer

        nb_rnn_layers=2,
        rnn_size=128,  # RNN contents, length of list = number of layers, list value = number of nodes

        self_attn=False,
        nb_heads=4,

        nb_fnn_layers=1,
        fnn_size=128,  # FNN contents, length of list = number of layers, list value = number of nodes

        nb_epochs=100,  # Train for maximum epochs
        lr=1e-3,

        # METRIC
        average='macro',  # Supports 'micro': sample-wise average and 'macro': class-wise average
        lad_doa_thresh=20,

        # distance
        train_synth_test_synth=False,
        only_dist=False,
        patience=20,
        overfit=False,
        # permutation 1: using original event mask | Won't propogate error for missing events
        permutation_1=False,
        # permutation 2: MSE(y,y_hat) + CE(M,M_hat)

        permutation_2=False, # Not completely implemented yet
        perm_2_loss_type = 'mse', # possible values: mse, mae, mape, mspe, thr_mape
        perm_2_loss_mpe_type_thr = 0.1,
        perm_2_onlyMask = False,

        synth_and_real_dcase=False,# use synth and modified real dcase dataset
        chan_swap_aug=False, # use channel swap augmentation
        chan_aug_folds=[1,3],
        use_all_data=False, # use dcase, starss, locata, metu, marco
        use_ind_data=False, # Is use one of the above datasets
        ind_data_train_test_split = [[1],[2]] # train and test split [Default is Dcase]
    )

    # ########### User defined parameters ##############

    if argv == '1':
        print("USING DEFAULT PARAMETERS\n")

    elif argv == '2':
        print("FOA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = False

    elif argv == '3':
        print("FOA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'foa'
        params['multi_accdoa'] = True

    elif argv == '4':
        print("MIC + GCC + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False

    elif argv == '5':
        print("MIC + SALSA + ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = False

    elif argv == '6':
        print("MIC + GCC + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = True

    elif argv == '7':
        print("MIC + SALSA + multi ACCDOA\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = True
        params['multi_accdoa'] = True

    elif argv == '8':
        print("MIC + GCC + dist\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True

    elif argv == '9':
        print("MIC + GCC + dist + dummy\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['only_dist'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/dummy_gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dummy_dist_feat_label'

    elif argv == '10':
        print("MIC + GCC + dist + train overfit\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['overfit'] = True

    elif argv == '11':
        print("MIC + GCC + dist + train overfit + patience 50\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['overfit'] = True
        params['patience'] = 50

    elif argv == '12':
        print("MIC + GCC + dist + event detection + patience 50\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_event_feat_label'
        params['only_dist'] = True

    elif argv == '13':
        print("MIC + GCC + dist + event detection + patience 50\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40

    elif argv == '14':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-1\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-1

    elif argv == '15':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-2

    elif argv == '16':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-4

    elif argv == '17':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-5

    elif argv == '18':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-6\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-6

    elif argv == '19':
        print("MIC + GCC + dist + event detection + patience 40 + lr : 1e-7\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-7

    elif argv == '20':
        print("MIC + GCC + dist + event detection + 2400 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv_2400'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label_2400'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40

    elif argv == '21':
        print("MIC + GCC + dist + event detection + 3600\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv_3600'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label_3600'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40

    elif argv == '22':
        print("MIC + GCC + dist + event detection + 2400 + lr=1e-6\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv_2400'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label_2400'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-6

    elif argv == '23':
        print("MIC + GCC + dist  + 3600 + lr=1e-6\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_onlySynth_dist_noOv_3600'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_feat_label_3600'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['lr'] = 1e-6

    elif argv == '24':
        print("MIC + GCC + dist + locata\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_and_locata'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_and_locata'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['use_locata'] = True
        params['lr'] = 1e-6

    elif argv == '25':
        print("MIC + GCC + dist + synthDcase & starss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['synth_and_real_dcase'] = True
        params['lr'] = 1e-6

    elif argv == '26':
        print("MIC + GCC + dist + synthDcase & starss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['synth_and_real_dcase'] = True

    elif argv == '27':
        print("MIC + GCC + dist + synthDcase & starss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['synth_and_real_dcase'] = True
        params['lr'] = 1e-6
        params['patience'] = 40

    elif argv == '28':
        print("MIC + GCC + dist + synthDcase & starss\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['synth_and_real_dcase'] = True
        params['patience'] = 40

    elif argv == '29':
        print("MIC + GCC + dist + synthDcase & starss + locata\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['use_locata'] = True
        params['patience'] = 40

    elif argv == '30':
        print("MIC + GCC + dist + synthDcase & starss + locata + chan swap\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss_aug'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['use_locata'] = True
        params['chan_swap_aug'] = True

    elif argv == '31':
        print("MIC + GCC + dist + synthDcase & starss + locata + chan swap\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_synDcase_locata_starss'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_dcase_locata_starss_aug'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['use_locata'] = True
        params['chan_swap_aug'] = True
        params['lr'] = 1e-5

    elif argv == '32':
        print("MIC + GCC + dist + All data + lr:1e-3\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['use_all_data'] = True

    elif argv == '33':
        print("MIC + GCC + dist + All data + lr:1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['lr'] = 1e-5

    elif argv == '34':
        print("MIC + GCC + dist + All data + lr:1e-6\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['train_synth_test_synth'] = True
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_1'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['lr'] = 1e-6

    elif argv == '35':
        print("MIC + GCC + dist + All data + testing mask training\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True

    elif argv == '36':
        print("MIC + GCC + dist + All data + post pretrained mask training + mse\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '37':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '38':
        print("MIC + GCC + dist + All data + post pretrained mask training + mse\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-6

    elif argv == '39':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-6

    elif argv == '40':
        print("MIC + GCC + dist + All data + post pretrained mask training + mse\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '41':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '42':
        print("MIC + GCC + dist + All data + post pretrained mask training + mse\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-4

    elif argv == '43':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-4

    elif argv == '44':
        print("MIC + GCC + dist + All data + no pretrained model\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['perm_2_loss_type'] = 'mse'

    elif argv == '45':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mpe'

    elif argv == '46':
        print("MIC + GCC + dist + All data + post pretrained mask training + mae\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mpe'

    ############################
    elif argv == '47':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '48':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '49':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '50':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '51':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '52':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '53':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '54':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '55':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '56':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '57':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '58':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    elif argv == '59':
        print("MIC + GCC + dist + All data + no pretrained model\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_all_data'] = True
        params['perm_2_loss_type'] = 'mse'

    #### INDIVIDUAL EXPERIMENTS ####
    elif argv == '60':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '61':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '62':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '63':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '64':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '65':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '66':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '67':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '68':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '69':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '70':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '71':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    ### STARS DATA ###
    elif argv == '72':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '73':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '74':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '75':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '76':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '77':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '78':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '79':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '80':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '81':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '82':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '83':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    ### LOCATA ###
    elif argv == '84':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '85':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '86':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '87':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '88':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '89':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '90':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '91':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '92':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '93':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '94':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '95':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ### MARCO ###
    elif argv == '96':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '97':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '98':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '99':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '100':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '101':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '102':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '103':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '104':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '105':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '106':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '107':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

### METU ###
    elif argv == '108':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '109':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '110':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '111':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '112':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '113':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '114':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '115':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '116':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '117':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '118':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] =  [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '119':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

### DCASE + STARS ###
    elif argv == '120':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '121':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '122':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '123':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '124':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '125':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '126':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '127':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '128':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '129':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '130':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '131':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

### METU Balanced ###
    elif argv == '132':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '133':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '134':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '135':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '136':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '137':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '138':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '139':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '140':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '141':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '142':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] =  [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '143':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/bal_metu'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_bal_metu'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    ### Leave one out experiments ###

    ## mask pretraining ##

    ## leave: DCASE ##
    elif argv == '144':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]

    ## leave: STARS ##
    elif argv == '145':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]

    ## leave: LOC ##
    elif argv == '146':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]

    ## leave: METU ##
    elif argv == '147':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]

    ## leave: MARCO ##
    elif argv == '148':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7], [2,4,6,8,10]]

    ### leave one out full training ###
    ## leave: DCASE ##
    elif argv == '149':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '150':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '151':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '152':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '153':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '154':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '155':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '156':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '157':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '158':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '159':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '160':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/144_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    ## leave: STARSS ##
    elif argv == '161':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '162':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '163':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '164':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '165':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '166':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '167':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '168':
        print("MSPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '169':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '170':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '171':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '172':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/145_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    ## leave: LOCATA ##
    elif argv == '173':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'

    elif argv == '174':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '175':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'

    elif argv == '176':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '177':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'

    elif argv == '178':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '179':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'

    elif argv == '180':
        print("MSPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '181':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1

    elif argv == '182':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01

    elif argv == '183':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2

    elif argv == '184':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/146_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4

    ## locata baseline ##
    ## mask pretraining ##
    elif argv == '185':
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['perm_2_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['batch_size'] = 32

    elif argv == '186':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/185_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['batch_size'] = 32
        

    elif argv == '1890':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '187':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '188':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '88':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '89':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '90':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '91':
        print("MSPE + lr=1e-5 + Re-starting due to timeout\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/54_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '92':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '93':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '94':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '95':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_loc_metu_marco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_l_m_m'
        params['only_dist'] = True
        params['permutation_2'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/35_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_2_loss_type'] = 'thr_mape'
        params['perm_2_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    elif argv == '999':
        print("QUICK TEST MODE\n")
        params['quick_test'] = True

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()
    feature_label_resolution = int(params['label_hop_len_s'] // params['hop_len_s'])
    params['feature_sequence_length'] = params['label_sequence_length'] * feature_label_resolution
    params['t_pool_size'] = [feature_label_resolution, 1, 1]  # CNN time pooling
    # Stop training if patience is reached

    params['unique_classes'] = 13

    if '2020' in params['dataset_dir']:
        params['unique_classes'] = 14
    elif '2021' in params['dataset_dir']:
        params['unique_classes'] = 12
    elif '2022' in params['dataset_dir']:
        params['unique_classes'] = 13

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params
