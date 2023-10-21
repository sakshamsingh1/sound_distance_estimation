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

        #### base (non aug) loc ####
    elif argv == '504':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '505':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '506':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '507':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '508':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '509':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '510':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '511':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        #### base (aug) loc with DCASE (mask pretrain)  ####
    elif argv == '512':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '513':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [2,12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '514':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '515':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '516':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '517':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '518':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '519':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [2,12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        #### base (non aug) loc with DCASE (mask pretrain)  ####
    elif argv == '520':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '521':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [2,12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '522':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '523':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '524':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '525':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '526':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '527':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [2,12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        #### base (aug) loc (DCASE pretrain)  ####
    elif argv == '528':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '529':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '530':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '531':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '532':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 11], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '533':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '534':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '535':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        #### base (non aug) loc (DCASE pretrain)  ####
    elif argv == '536':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '537':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '538':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '539':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '540':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '541':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '542':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '543':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ### train all (Aug loc) ###
    elif argv == '544':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '545':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '546':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '547':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '548':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9, 11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '549':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '550':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '551':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### train all (Non Aug loc) ###
    elif argv == '552':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '553':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '554':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '555':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '556':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9, 11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '557':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '558':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '559':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9,11], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### Leave locata out ###
    elif argv == '560':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '561':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '562':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '563':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '564':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '565':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '566':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '567':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2, 4, 8, 10, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

#### STARSS only: to be used for finetuning on Loc later ####
    elif argv == '568':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '569':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '570':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '571':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '572':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '573':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '574':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '575':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

        #### STARSS + LOC ####
    elif argv == '576':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '577':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '578':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '579':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '580':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '581':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '582':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '583':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### STARSS + LOC(Non Aug) #####
    elif argv == '584':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '585':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '586':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '587':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '588':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '589':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '590':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '591':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

##### AUG LOC Finetuning with STARSS ####
    elif argv == '592':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '593':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '594':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '595':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '596':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '597':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '598':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '599':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        ##### (Non AUG) LOC Finetuning with STARSS ####
    elif argv == '600':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '601':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '602':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '603':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '604':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '605':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '606':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '607':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_baseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

        ##### DCASE (STARSS Finetuning) ####
    elif argv == '608':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '609':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '610':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '611':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '612':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '613':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '614':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '615':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

        ##### METU (STARSS Finetuning) ####
    elif argv == '616':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10


    elif argv == '617':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '618':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '619':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '620':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '621':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '622':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '623':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

        ##### MARCo (STARSS Finetuning) ####
    elif argv == '624':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10


    elif argv == '625':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '626':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '627':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '628':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '629':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '630':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '631':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Abaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_ABasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/224_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10















################ fixed locata ###################
    elif argv == '632':
        ## Feature generate
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc_fixed_aug'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc_fixed_aug'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '633':
        ## Feature generate
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    ### Fixed Aug Loc  ###
    elif argv == '634':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '635':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '636':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '637':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '638':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '639':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '640':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '641':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

##### Locata + DCASE (optimise L) #####
    elif argv == '642':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '643':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '644':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '645':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '646':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '647':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '648':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '649':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,11], [13], [2, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4


        ##### All data (optimise L) #####
    elif argv == '650':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '651':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '652':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '653':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '654':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '655':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '656':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '657':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 11, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

        ##### All data without loc (optimise L) #####
    elif argv == '658':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '659':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '660':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '661':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '662':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '663':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '664':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '665':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 7, 9], [13], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### Locata + STARSS (optimise L) #####
    elif argv == '666':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '667':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '668':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '669':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '670':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '671':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '672':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '673':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3, 11], [13], [4, 12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### only DCASE #####
    elif argv == '674':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '675':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '676':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '677':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '678':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '679':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '680':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '681':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### only STARSS #####
    elif argv == '682':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '683':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '684':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '685':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '686':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '687':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '688':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '689':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [15], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### only MeTU #####
    elif argv == '690':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '691':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '692':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '693':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '694':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '695':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '696':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '697':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    ##### only Marco #####
    elif argv == '698':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '699':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '700':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '701':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '702':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '703':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '704':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '705':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    ### LOC (Starss pretrained)  ###
    elif argv == '706':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '707':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '708':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '709':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '710':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '711':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '712':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '713':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32


    ### LOC (DCASE pretrained)  ###
    elif argv == '714':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '715':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '716':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '717':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '718':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '719':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '720':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '721':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [13], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/674_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ##### only DCASE (STARSS Finetune) #####
    elif argv == '722':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '723':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '724':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '725':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '726':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '727':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '728':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '729':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4


    ##### only MeTU (STarss Finetune) #####
    elif argv == '730':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '731':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '732':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '733':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '734':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '735':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '736':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '737':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [16], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

        ##### only Marco (Starss Finetune) #####
    elif argv == '738':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '739':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '740':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '741':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '742':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '743':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '744':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 20

    elif argv == '745':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [17], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/irr2020/dist_est_saksham/models/689_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 20


###### Train dcase, test all other datasets ######
    elif argv == '746':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '747':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '748':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '749':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '750':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '751':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '752':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '753':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

        ###### Train Starss, test all other datasets ######
    elif argv == '754':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '755':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '756':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '757':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '758':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params['perm_3_loss_type'] = 'thr_mape'
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '759':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '760':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '761':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params[
            'dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [14, 15, 13, 16, 17], [2, 4, 12, 8, 10]]
        params['finetune_mode'] = True
        params[
            'pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

##### DCASE TRAINED FROM SCRATCH #####
    elif argv == '762':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_newSplit_dcase_stars_AFixbaseloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_newSplit_d_s_AFixbasel_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [14], [2]]
        params['perm_3_loss_type'] = 'mse'















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