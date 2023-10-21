# Parameters used in the data generation process.


def get_params(argv='1'):
    print("SET: {}".format(argv))
    # ########### default parameters (NIGENS data) ##############
    params = dict(
        db_name = 'nigens',  # name of the audio dataset used for data generation
        rirpath = '/scratch/asignal/krauseda/DCASE_data_generator/RIR_DB',   # path containing Room Impulse Responses (RIRs)
        mixturepath = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Dataset-NIGENS',  # output path for the generated dataset
        noisepath = '/scratch/asignal/krauseda/DCASE_data_generator/Noise_DB',  # path containing background noise recordings
        nb_folds = 2,  # number of folds (default 2 - training and testing)
        rooms2fold = [[10, 6, 1, 4, 3, 8], # FOLD 1, rooms assigned to each fold (0's are ignored)
                      [9, 5, 2, 0, 0, 0]], # FOLD 2
        db_path = 'E:/DCASE2022/TAU_Spatial_RIR_Database_2021/Code/NIGENS',  # path containing audio events to be utilized during data generation
        max_polyphony = 3,  # maximum number of overlapping sound events
        active_classes = [0, 1, 2, 3, 5, 6, 8, 9, 10, 11, 12, 13],  # list of sound classes to be used for data generation
        nb_mixtures_per_fold = [900, 300], # if scalar, same number of mixtures for each fold
        mixture_duration = 60., #in seconds
        event_time_per_layer = 40., #in seconds (should be less than mixture_duration)
        audio_format = 'both', # 'foa' (First Order Ambisonics) or 'mic' (four microphones) or 'both'
        obj_path = 'db_config_fsd.obj',
        old_meta_synth = False,
            )
        

    # ########### User defined parameters ##############
    if argv == '1':
        print("USING DEFAULT PARAMETERS FOR NIGENS DATA\n")

    elif argv == '2': ###### FSD50k DATA
        params['db_name'] = 'fsd50k'
        params['db_path']= '/scratch/asignal/krauseda/DCASE_data_generator/Code/FSD50k'
        params['mixturepath'] = '/scratch/asignal/krauseda/Data-FSD'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2

    elif argv == '3': ###### NIGENS interference data
        params['active_classes'] = [4, 7, 14] 
        params['max_polyphony'] = 1

    elif argv == '4': ###### FSD50k DATA
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_synth_dist'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'

    elif argv == '5': ###### FSD50k DATA
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_synth_maxPol_1'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj.pkl'
        params['old_meta_synth'] = True

    elif argv == '6':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'

    elif argv == '7':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_oldSynth_itr1'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj.pkl'
        params['old_meta_synth'] = True
        params['audio_format'] = 'mic'

    elif argv == '8':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_oldSynth_itr2'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj.pkl'
        params['old_meta_synth'] = True
        params['audio_format'] = 'mic'

    elif argv == '9':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_oldSynth_itr3'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj.pkl'
        params['old_meta_synth'] = True
        params['audio_format'] = 'mic'

    elif argv == '10':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_newFold'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 2
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['audio_format'] = 'mic'
        params['rooms2fold'] = [[10, 5, 1, 4, 3, 8],[9, 6, 2, 0, 0, 0]]
        params['nb_mixtures_per_fold'] = [1800, 300]

    elif argv == '11':
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_noOverlap'
        params['noisepath'] = '/scratch/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/scratch/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'

    elif argv == '12':
        print('revamp dummy run test')
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/dummy_gen_synth_distFloat'
        params['noisepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/vast/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['nb_mixtures_per_fold'] = [10, 5]

    elif argv == '13':
        print('distance data generator for 10k mixtures')
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_noOverlap_10k'
        params['noisepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/vast/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['nb_mixtures_per_fold'] = [7500, 2500]

    elif argv == '14':
        print('distance data generator for 2400 mixtures')
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_noOverlap_2400'
        params['noisepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/vast/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['nb_mixtures_per_fold'] = [1801, 601]
        
    elif argv == '15':
        print('distance data generator for 3600 mixtures')
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_noOverlap_3600'
        params['noisepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/vast/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['nb_mixtures_per_fold'] = [2700, 900]

    elif argv == '16':
        print('distance data generator for 2400 mixtures')
        params['db_name'] = 'fsd50k'
        params['rirpath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SRIR_DB'
        params['mixturepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/gen_synth_distFloat_noOverlap_2400'
        params['noisepath'] = '/vast/sk8974/experiments/dsynth/data/util_data/TAU-SNoise_DB'
        params['db_path'] = '/vast/sk8974/experiments/dsynth/data/util_data/fsdk_upd'
        params['active_classes'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        params['max_polyphony'] = 1
        params['obj_path'] = 'dcase22_updated_obj_dist.pkl'
        params['nb_mixtures_per_fold'] = [1800, 600]
        params['audio_format'] = 'mic'

    else:
        print('ERROR: unknown argument {}'.format(argv))
        exit()

    for key, value in params.items():
        print("\t{}: {}".format(key, value))
    return params