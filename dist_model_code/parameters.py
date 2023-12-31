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

        permutation_3=False,  # Not completely implemented yet
        perm_3_loss_type='mse',  # possible values: mse, mae, mape, mspe, thr_mape
        perm_3_loss_mpe_type_thr=0.1,
        perm_3_onlyMask=False,

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
        print("Pretraining : only training the event detector\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['perm_3_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
    
    elif argv == '3':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['perm_3_loss_type'] = 'mse'

    elif argv == '4':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['finetune_mode'] = True
        params['permutation_3'] = True
        params['pretrained_model_weights'] = 'models/pretrained_dcase_event_detector.h5'
        params['perm_3_loss_type'] = 'mse'
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        # params['perm_3_loss_mpe_type_thr'] = 0.4

    elif argv == '5':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = 'models/pretrained_dcase_event_detector.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
    
    elif argv == '6':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['finetune_mode'] = True
        params['permutation_3'] = True
        params['pretrained_model_weights'] = 'models/pretrained_dcase_event_detector.h5'
        params['perm_3_loss_type'] = 'mape'
        params['ind_data_train_test_split'] = [[1,3], [13, 14], [4]]
        params['perm_3_loss_mpe_type_thr'] = 0.4

    elif argv == '7':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = 'data/input/'
        params['feat_label_dir'] = 'data/processed/'
        params['only_dist'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['finetune_mode'] = True
        params['permutation_3'] = True
        params['pretrained_model_weights'] = 'models/pretrained_dcase_event_detector.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['ind_data_train_test_split'] = [[11,3], [13, 15], [4]]
        params['perm_3_loss_mpe_type_thr'] = 0.01

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