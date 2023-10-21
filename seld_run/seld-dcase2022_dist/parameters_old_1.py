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

    #pretraining on DCASE
    elif argv == '200':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['perm_3_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]

    elif argv == '201':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['perm_3_onlyMask'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['lr'] = 1e-5

    ### Individual models ###
    ### DCASE ###
    elif argv == '202':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '203':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '204':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '205':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '206':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '207':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '208':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '209':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '210':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '211':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '212':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '213':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1], [2]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    #### STARSS ####
    elif argv == '214':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '215':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '216':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '217':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '218':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '219':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '220':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '221':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '222':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '223':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '224':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '225':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### LOCATA ###
    elif argv == '226':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '227':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '228':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '229':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '230':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '231':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '232':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '233':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '234':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '235':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '236':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '237':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ### METU ###
    elif argv == '238':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '239':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '240':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '241':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '242':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '243':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '244':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '245':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '246':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '247':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '248':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '249':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    ##### Marco#####
    elif argv == '250':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '251':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '252':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '253':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '254':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '255':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '256':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '257':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '258':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '259':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '260':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '261':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ### SINGLE DATA + DCASE (from scratch) ###

    ### STARSS (+ DCASE) ###
    elif argv == '262':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '263':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '264':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '265':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '266':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '267':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '268':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '269':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '270':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '271':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '272':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '273':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3], [2,4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### LOCATA (+DCASE) ###
    elif argv == '274':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '275':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '276':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '277':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '278':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '279':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '280':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '281':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '282':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '283':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '284':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '285':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5], [2,6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    #### METU (+DCASE) ####
    elif argv == '286':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '287':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '288':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '289':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '290':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '291':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '292':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '293':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '294':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '295':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '296':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '297':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### Marco (+ DCASE) ####
    elif argv == '298':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '299':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '300':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '301':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '302':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '303':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '304':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '305':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '306':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '307':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '308':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '309':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### Single dataset (+DCASE) + DCASE best start #####
        ### STARSS (+ DCASE) ###
    elif argv == '310':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '311':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '312':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '313':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '314':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '315':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '316':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '317':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '318':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '319':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '320':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '321':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3], [2, 4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### LOCATA (+DCASE) ###
    elif argv == '322':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '323':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '324':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '325':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '326':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '327':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '328':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '329':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '330':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '331':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '332':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '333':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 5], [2, 6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    #### METU (+DCASE) ####
    elif argv == '334':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '335':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '336':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '337':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '338':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '339':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '340':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '341':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '342':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '343':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '344':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '345':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 7], [2, 8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### Marco (+ DCASE) ####
    elif argv == '346':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '347':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '348':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '349':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '350':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '351':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '352':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '353':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '354':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '355':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '356':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '357':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 9], [2, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### Train all #####
    elif argv == '358':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '359':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '360':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '361':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '362':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '363':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '364':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '365':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '366':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '367':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '368':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '369':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ###### Leave one out experiments ######
    ###### leave DCASE out ########
    elif argv == '370':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '371':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '372':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '373':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '374':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '375':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '376':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '377':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '378':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '379':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '380':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '381':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ###### leave STARSS out ########
    elif argv == '382':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '383':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '384':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '385':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '386':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '387':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '388':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '389':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '390':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '391':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '392':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '393':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,5,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ###### leave LOCATA out ########
    elif argv == '394':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '395':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '396':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '397':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '398':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '399':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '400':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '401':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '402':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '403':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '404':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '405':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,7,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ###### leave METU out ########
    elif argv == '406':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '407':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '408':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '409':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '410':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '411':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '412':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '413':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '414':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '415':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '416':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '417':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1,3,5,9], [2,4,6,8,10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

        ###### leave MARCO out ########
    elif argv == '418':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '419':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '420':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '421':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '422':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '423':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '424':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '425':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '426':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '427':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '428':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '429':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[1, 3, 5, 7], [2, 4, 6, 8, 10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4


    ##### [FIXED]Single dataset (+DCASE) + DCASE best start #####
        ### STARSS (+ DCASE) ###
    elif argv == '430':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '431':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '432':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '433':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '434':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '435':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '436':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '437':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '438':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '439':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '440':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '441':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[3], [4]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ### LOCATA (+DCASE) ###
    elif argv == '442':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '443':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '444':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '445':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '446':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '447':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '448':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '449':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '450':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '451':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '452':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '453':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[5], [6]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    #### METU (+DCASE) ####
    elif argv == '454':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '455':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '456':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '457':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '458':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '459':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '460':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '461':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '462':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '463':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '464':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    elif argv == '465':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[7], [8]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32
        params['label_sequence_length'] = 10

    ##### Marco (+ DCASE) ####
    elif argv == '466':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'

    elif argv == '467':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5

    elif argv == '468':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'

    elif argv == '469':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5

    elif argv == '470':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'

    elif argv == '471':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5

    elif argv == '472':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'

    elif argv == '473':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5

    elif argv == '474':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1

    elif argv == '475':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01

    elif argv == '476':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2

    elif argv == '477':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/gen_dcase_stars_Aloc_Ametu_Amarco'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_d_s_Al_Am_Am'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[9], [10]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    ##### BASELINE LOC #####
    elif argv == '478':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc_aug'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc_aug'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/202_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    elif argv == '479':
        ##### BASELINE #####
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4

    #### BASELINE LOC ####
    elif argv == '480':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '481':
        print("MAE + lr=1e-5 \n")
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '482':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '483':
        print("MSE + lr=1e-5 \n")
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '484':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '485':
        print("MAPE + lr=1e-5 \n")
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '486':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '487':
        print("MSPE + lr=1e-5\n")
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '488':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.1
        params['batch_size'] = 32

    elif argv == '489':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.01
        params['batch_size'] = 32

    elif argv == '490':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.2
        params['batch_size'] = 32

    elif argv == '491':
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
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'thr_mape'
        params['perm_3_loss_mpe_type_thr'] = 0.4
        params['batch_size'] = 32

    ### base loc no aug ###
    elif argv == '492':
        print("MAE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['batch_size'] = 32

    elif argv == '493':
        print("MAE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mae'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '494':
        print("MSE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['batch_size'] = 32

    elif argv == '495':
        print("MSE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mse'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '496':
        print("MAPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['batch_size'] = 32

    elif argv == '497':
        print("MAPE + lr=1e-5 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mape'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '498':
        print("MSPE + lr=1e-3 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['batch_size'] = 32

    elif argv == '499':
        print("MSPE + lr=1e-5\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
        params['only_dist'] = True
        params['permutation_3'] = True
        params['patience'] = 40
        params['use_ind_data'] = True  # Is use one of the above datasets
        params['ind_data_train_test_split'] = [[11], [12]]
        params['finetune_mode'] = True
        params['pretrained_model_weights'] = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/models/200_1_dev_split0_accdoa_mic_gcc_model.h5'
        params['perm_3_loss_type'] = 'mspe'
        params['lr'] = 1e-5
        params['batch_size'] = 32

    elif argv == '500':
        print("THR_MAPE + lr=1e-3 + thr = 0.1 \n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
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

    elif argv == '501':
        print("MSPE + lr=1e-3 + thr=0.01\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
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

    elif argv == '502':
        print("MSPE + lr=1e-3 + thr=0.2\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
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

    elif argv == '503':
        print("MSPE + lr=1e-3 + thr=0.4\n")
        params['quick_test'] = False
        params['dataset'] = 'mic'
        params['use_salsalite'] = False
        params['multi_accdoa'] = False
        params['dataset_dir'] = '/vast/sk8974/experiments/dsynth/data/input/base_loc'
        params['feat_label_dir'] = '/vast/sk8974/experiments/dsynth/data/processed/dist_base_loc'
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
