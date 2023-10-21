import os
import SELD_evaluation_metrics
import cls_feature_class
import parameters_old_1
import numpy as np
from scipy import stats
from IPython import embed


def jackknife_estimation(global_value, partial_estimates, significance_level=0.05):
    """
    Compute jackknife statistics from a global value and partial estimates.
    Original function by Nicolas Turpault

    :param global_value: Value calculated using all (N) examples
    :param partial_estimates: Partial estimates using N-1 examples at a time
    :param significance_level: Significance value used for t-test

    :return:
    estimate: estimated value using partial estimates
    bias: Bias computed between global value and the partial estimates
    std_err: Standard deviation of partial estimates
    conf_interval: Confidence interval obtained after t-test
    """

    mean_jack_stat = np.mean(partial_estimates)
    n = len(partial_estimates)
    bias = (n - 1) * (mean_jack_stat - global_value)

    std_err = np.sqrt(
        (n - 1) * np.mean((partial_estimates - mean_jack_stat) * (partial_estimates - mean_jack_stat), axis=0)
    )

    # bias-corrected "jackknifed estimate"
    estimate = global_value - bias

    # jackknife confidence interval
    if not (0 < significance_level < 1):
        raise ValueError("confidence level must be in (0, 1).")

    t_value = stats.t.ppf(1 - significance_level / 2, n - 1)

    # t-test
    conf_interval = estimate + t_value * np.array((-std_err, std_err))

    return estimate, bias, std_err, conf_interval

'''
How to use the function below
'''


'''
import os
import sys
import numpy as np

path = '/vast/sk8974/experiments/dsynth/scripts/seld_run/revamp_seld/seld-dcase2022_dist'
sys.path.append(path)
import cls_feature_class
import parameters
from tqdm import tqdm
from cls_compute_seld_results import compute_dist_metrics_extended, format_dict_to_str

params = parameters.get_params('60')
dcase_output_val_folder = '/vast/sk8974/experiments/dsynth/scripts/seld_run/run/results/60_1_dev_split0_accdoa_mic_gcc_20230406204025_test'

data_wise_metrics = compute_dist_metrics_extended(params, val_fold_split=[2,4,6,8,10], ref_files_folder=dcase_output_val_folder)
format_dict_to_str(data_wise_metrics, ind=True)
'''

def compute_dist_metrics_extended(params, val_fold_split, ref_files_folder=None):
    '''
    ONLY FOR PERMUTATION 2
    Compute distance MAE & M%E for combined and individual datasets
    Returns: MAE, M%E
    '''

    def get_all_metrics(true_dists, out_dists, gt_masks, pred_masks, thr=0.5):
        '''
        Compute MAE, MPE, gt_MAE, gt_MPE, pred_MAE, pred_MPE, Precision, Recall, F1
        '''
        deno_eps = 0.1

        #MAE calculation
        true_dists, out_dists, pred_masks, gt_masks = np.array(true_dists), np.array(out_dists), np.array(pred_masks), np.array(gt_masks)
        pred_dists = out_dists * pred_masks
        maes = np.abs(true_dists - pred_dists)
        mae = np.mean(maes)

        # MPE calculation
        targ_deno = true_dists.copy()
        targ_deno[true_dists == 0] += deno_eps
        mpes = maes / targ_deno
        mpe = np.mean(mpes)

        # PRF calculation
        true_pos = np.sum((gt_masks == 1) & (pred_masks >= thr))
        false_pos = np.sum((gt_masks == 0) & (pred_masks >= thr))
        false_neg = np.sum((gt_masks == 1) & (pred_masks < thr))

        prec = true_pos / (true_pos + false_pos)
        rec = true_pos / (true_pos + false_neg)
        f1 = 2 * prec * rec / (prec + rec)

        # gt_MAE and gt_MPE calculation
        #calculate MAE and MPE for gt_mask = 1
        gt_mae = np.mean(maes[gt_masks == 1])
        gt_mpe = np.mean(mpes[gt_masks == 1])

        # pred_MAE and pred_MPE calculation
        # calculate MAE and MPE for pred_mask >= thr
        pred_mae = np.mean(maes[pred_masks >= thr])
        pred_mpe = np.mean(mpes[pred_masks >= thr])

        return mae, mpe, prec, rec, f1, gt_mae, gt_mpe, pred_mae, pred_mpe

    def agg_metrics(this_metrics):
        '''

        Aggregate metrics across datasets and foldwise

        dcase synth : fold 1, 2
        dcase stars : fold 3, 4
        locata: fold 5(orig eval), 6(orig dev)
        metu: fold 7, 8 Train vs test [146, 98]
        marco: fold 9, 10
        '''

        FOLD_MAP = {}
        FOLD_MAP[2] = 'D'
        FOLD_MAP[4] = 'S'
        FOLD_MAP[12] = 'L'
        FOLD_MAP[8] = 'Me'
        FOLD_MAP[10] = 'Ma'

        agg_dict = {'global': {}, 'fold': {}}
        glob_gt_dist, glob_out_dist, glob_gt_mask, glob_pred_mask = [], [], [], []

        #this_metrics = data_wise_metrics[this_fold]['gt_dist']
        for fold in this_metrics:
            fold_name = FOLD_MAP[fold]

            mae, mpe, prec, rec, f1, gt_mae, gt_mpe, pred_mae, pred_mpe = \
                get_all_metrics(this_metrics[fold]['gt_dist'], this_metrics[fold]['out_dist'],
                            this_metrics[fold]['gt_mask'], this_metrics[fold]['pred_mask'])

            agg_dict['fold'][fold_name] = {'mae': mae, 'mpe': mpe, 'prec': prec, 'rec': rec, 'f1': f1,
                                           'gt_mae': gt_mae, 'gt_mpe': gt_mpe, 'pred_mae': pred_mae, 'pred_mpe': pred_mpe}

            glob_gt_dist.extend(this_metrics[fold]['gt_dist'])
            glob_out_dist.extend(this_metrics[fold]['out_dist'])
            glob_gt_mask.extend(this_metrics[fold]['gt_mask'])
            glob_pred_mask.extend(this_metrics[fold]['pred_mask'])

        mae, mpe, prec, rec, f1, gt_mae, gt_mpe, pred_mae, pred_mpe = \
            get_all_metrics(glob_gt_dist, glob_out_dist, glob_gt_mask, glob_pred_mask)

        agg_dict['global'] = {'mae': mae, 'mpe': mpe, 'prec': prec, 'rec': rec, 'f1': f1,
                              'gt_mae': gt_mae, 'gt_mpe': gt_mpe, 'pred_mae': pred_mae, 'pred_mpe': pred_mpe}
        return agg_dict

    perm_2 = params['permutation_2']
    gt_sup_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
    feat_cls = cls_feature_class.FeatureClass(params)

    data_wise_metrics = {}

    for data_set in os.listdir(gt_sup_dir):

        for gt_file in os.listdir(os.path.join(gt_sup_dir, data_set)):
            this_fold = int(gt_file.split('_')[0][4:])

            if this_fold in val_fold_split:
                if this_fold not in data_wise_metrics:
                    data_wise_metrics[this_fold] = {'gt_dist': [], 'out_dist': [],
                                                    'gt_mask': [], 'pred_mask': []}

                output_file = os.path.join(ref_files_folder, gt_file)

                gt_dict = feat_cls.load_output_format_file(os.path.join(gt_sup_dir, data_set, gt_file))
                try:
                    out_dict = feat_cls.load_output_format_file(output_file)
                except:
                    continue

                nb_ref_frames = max(list(gt_dict.keys()))

                for frame in range(nb_ref_frames + 1):
                    gt_dist = 0
                    gt_mask = 0
                    pred_mask = None
                    if frame in gt_dict:
                        gt_dist = gt_dict[frame][0][4]
                        gt_mask = 1
                    if perm_2:
                        out_dist = out_dict[frame][0][0]
                        pred_mask = out_dict[frame][0][1]
                    else:
                        out_dist = out_dict[frame][0][0]

                    data_wise_metrics[this_fold]['gt_dist'].append(gt_dist)
                    data_wise_metrics[this_fold]['out_dist'].append(out_dist)
                    data_wise_metrics[this_fold]['gt_mask'].append(gt_mask)
                    data_wise_metrics[this_fold]['pred_mask'].append(pred_mask)

    results = agg_metrics(data_wise_metrics)
    return results


def format_dict_to_str(input_dict, ind=False):
    global_dict = input_dict.get('global', {})
    fold_dict = input_dict.get('fold', {})

    fold_names = ['D', 'S', 'L', 'Me', 'Ma']

    # Extract the global values
    global_vals = [
        str(global_dict.get('mae', '')),
        str(global_dict.get('mpe', '')),
        str(global_dict.get('gt_mae', '')),
        str(global_dict.get('gt_mpe', '')),
        str(global_dict.get('pred_mae', '')),
        str(global_dict.get('pred_mpe', '')),
        str(global_dict.get('prec', '')),
        str(global_dict.get('rec', '')),
        str(global_dict.get('f1', ''))
    ]

    fold_vals = []
    if not ind:
    # Extract the fold values
        fold_metrics = ['mae', 'mpe', 'gt_mae', 'gt_mpe', 'pred_mae', 'pred_mpe', 'prec', 'rec', 'f1']
        for met in fold_metrics:
            for fold_name in fold_names:
                if fold_name not in fold_dict:
                    continue
                fold_data = fold_dict[fold_name]
                fold_vals.extend([str(fold_data.get(met, ''))])

    # Extract the fold metrics
    output = global_vals + fold_vals
    output = [round(float(x), 3) for x in output]
    output_str = ','.join(map(str, output))
    return output_str

############## Permutation 3 ####################
def compute_dist_metrics_extended_perm3(params, val_fold_split, ref_files_folder=None):
    '''
    ONLY FOR PERMUTATION 3
    Compute distance GT_MAE, GT_M%E & Out_MAE for combined and individual datasets
    Returns: MAE, M%E
    '''

    def get_all_metrics(true_dists, out_dists, gt_masks, pred_masks, thr=0.5):
        '''
        Compute GT_MAE, GT_MPE, overall_mae, pred_MAE, Precision, Recall, F1
        '''

        #MAE calculation
        true_dists, out_dists, pred_masks, gt_masks = np.array(true_dists), np.array(out_dists), np.array(pred_masks), np.array(gt_masks)

        # GT_MAE, GT_MPE calculation
        true_dist_pos = true_dists[gt_masks == 1]
        out_dist_pos = out_dists[gt_masks == 1]
        maes = np.abs(true_dist_pos - out_dist_pos)
        import pdb; pdb.set_trace()
        gt_mae = np.mean(maes)
        gt_mpe = np.mean(maes / true_dist_pos)

        # overall_mae calculation
        overall_mae = np.mean(np.abs(true_dists - pred_masks*out_dists))
        # import pdb; pdb.set_trace()

        # pred_MAE and pred_MPE calculation
        # calculate MAE and MPE for pred_mask >= thr
        true_dist_pos = true_dists[pred_masks >= thr]
        out_dist_pos = out_dists[pred_masks >= thr]
        pred_mae = np.mean(np.abs(true_dist_pos - out_dist_pos))

        # PRF calculation
        true_pos = np.sum((gt_masks == 1) & (pred_masks >= thr))
        false_pos = np.sum((gt_masks == 0) & (pred_masks >= thr))
        false_neg = np.sum((gt_masks == 1) & (pred_masks < thr))

        prec = true_pos / (true_pos + false_pos)
        rec = true_pos / (true_pos + false_neg)
        f1 = 2 * prec * rec / (prec + rec)

        return gt_mae, gt_mpe, overall_mae, pred_mae, prec, rec, f1

    def agg_metrics(this_metrics):
        '''

        Aggregate metrics across datasets and foldwise

        dcase synth : fold 1, 2
        dcase stars : fold 3, 4
        locata: fold 5(orig eval), 6(orig dev)
        metu: fold 7, 8 Train vs test [146, 98]
        marco: fold 9, 10
        '''

        FOLD_MAP = {}
        FOLD_MAP[2] = 'D'
        FOLD_MAP[4] = 'S'
        FOLD_MAP[12] = 'L'
        FOLD_MAP[8] = 'Me'
        FOLD_MAP[10] = 'Ma'

        agg_dict = {'global': {}, 'fold': {}}
        glob_gt_dist, glob_out_dist, glob_gt_mask, glob_pred_mask = [], [], [], []

        #this_metrics = data_wise_metrics[this_fold]['gt_dist']
        for fold in this_metrics:
            fold_name = FOLD_MAP[fold]

            gt_mae, gt_mpe, overall_mae, pred_mae, prec, rec, f1 = \
                get_all_metrics(this_metrics[fold]['gt_dist'], this_metrics[fold]['out_dist'],
                            this_metrics[fold]['gt_mask'], this_metrics[fold]['pred_mask'])

            agg_dict['fold'][fold_name] = {'gt_mae': gt_mae, 'gt_mpe': gt_mpe, 'overall_mae': overall_mae,
                                           'pred_mae': pred_mae, 'prec': prec, 'rec': rec, 'f1': f1}

            glob_gt_dist.extend(this_metrics[fold]['gt_dist'])
            glob_out_dist.extend(this_metrics[fold]['out_dist'])
            glob_gt_mask.extend(this_metrics[fold]['gt_mask'])
            glob_pred_mask.extend(this_metrics[fold]['pred_mask'])


        # mae, mpe, prec, rec, f1, gt_mae, gt_mpe, pred_mae, pred_mpe =
        gt_mae, gt_mpe, overall_mae, pred_mae, prec, rec, f1 = \
            get_all_metrics(glob_gt_dist, glob_out_dist, glob_gt_mask, glob_pred_mask)

        agg_dict['global'] = {'gt_mae': gt_mae, 'gt_mpe': gt_mpe, 'overall_mae': overall_mae,
                              'pred_mae': pred_mae, 'prec': prec, 'rec': rec, 'f1': f1}
        return agg_dict

    gt_sup_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
    feat_cls = cls_feature_class.FeatureClass(params)

    data_wise_metrics = {}

    for data_set in os.listdir(gt_sup_dir):

        for gt_file in os.listdir(os.path.join(gt_sup_dir, data_set)):
            this_fold = int(gt_file.split('_')[0][4:])

            if this_fold in val_fold_split:
                if this_fold not in data_wise_metrics:
                    data_wise_metrics[this_fold] = {'gt_dist': [], 'out_dist': [],
                                                    'gt_mask': [], 'pred_mask': []}

                output_file = os.path.join(ref_files_folder, gt_file)

                gt_dict = feat_cls.load_output_format_file(os.path.join(gt_sup_dir, data_set, gt_file))
                try:
                    out_dict = feat_cls.load_output_format_file(output_file)
                except:
                    continue

                nb_ref_frames = max(list(gt_dict.keys()))

                for frame in range(nb_ref_frames + 1):
                    gt_dist, gt_mask = 0, 0
                    if frame in gt_dict:
                        gt_dist = gt_dict[frame][0][4]
                        gt_mask = 1

                    out_dist = out_dict[frame][0][0]
                    pred_mask = out_dict[frame][0][1]

                    data_wise_metrics[this_fold]['gt_dist'].append(gt_dist)
                    data_wise_metrics[this_fold]['out_dist'].append(out_dist)
                    data_wise_metrics[this_fold]['gt_mask'].append(gt_mask)
                    data_wise_metrics[this_fold]['pred_mask'].append(pred_mask)

    results = agg_metrics(data_wise_metrics)
    return results

def format_dict_to_str_perm3(input_dict, ind=False):
    global_dict = input_dict.get('global', {})
    fold_dict = input_dict.get('fold', {})

    fold_names = ['D', 'S', 'L', 'Me', 'Ma']

    # Extract the global values
    global_vals = [
        str(global_dict.get('gt_mae', '')),
        str(global_dict.get('gt_mpe', '')),
        str(global_dict.get('overall_mae', '')),
        str(global_dict.get('pred_mae', '')),
        str(global_dict.get('prec', '')),
        str(global_dict.get('rec', '')),
        str(global_dict.get('f1', ''))
    ]

    fold_vals = []
    if not ind:
    # Extract the fold values
        fold_metrics = ['gt_mae', 'gt_mpe', 'overall_mae', 'overall_mae', 'pred_mae', 'prec', 'rec', 'f1']
        for met in fold_metrics:
            for fold_name in fold_names:
                if fold_name not in fold_dict:
                    continue
                fold_data = fold_dict[fold_name]
                fold_vals.extend([str(fold_data.get(met, ''))])

    # Extract the fold metrics
    output = global_vals + fold_vals
    output = [round(float(x), 3) for x in output]
    output_str = ','.join(map(str, output))
    return output_str

def compute_dist_metrics_perm3(params, val_fold_split, ref_files_folder=None):
    '''
    Compute true distance MAE & M%E combined and individual datasets
    Returns: MAE, M%E
    '''

    if not params['permutation_3']:
        raise Exception('Wrong permutation')

    def get_mae_mpe(true_dist, pred_dist_gt_mask, pred_dist):
        '''

        Compute MAE and MPE.
        For gt=0, deno = 0.1
        '''
        gt_mae = abs(true_dist - pred_dist_gt_mask)

        deno = true_dist
        if true_dist == 0:
            deno = true_dist + 0.1
        gt_mpe = gt_mae / deno

        overall_mae = abs(true_dist - pred_dist)

        return gt_mae, gt_mpe, overall_mae


    def agg_metrics(this_metrics):
        '''
        Aggregate metrics across datasets and foldwise

        dcase synth : fold 1, 2
        dcase stars : fold 3, 4
        locata: fold 5(orig eval), 6(orig dev)
        metu: fold 7, 8 Train vs test [146, 98]
        marco: fold 9, 10
        '''

        FOLD_MAP = {}
        FOLD_MAP[14] = 'D'
        FOLD_MAP[15] = 'S'
        FOLD_MAP[13] = 'L'
        FOLD_MAP[16] = 'Me'
        FOLD_MAP[17] = 'Ma'

        agg_dict = {'global': {}, 'fold': {}}
        glob_gt_mae = []
        glob_gt_mpe = []
        glob_overall_mpe = []

        for fold in this_metrics:
            fold_name = FOLD_MAP[fold]
            agg_dict['fold'][fold_name] = {}
            agg_dict['fold'][fold_name]['gt_mae'] = np.mean(this_metrics[fold]['gt_mae'])
            agg_dict['fold'][fold_name]['gt_mpe'] = np.mean(this_metrics[fold]['gt_mpe'])
            agg_dict['fold'][fold_name]['overall_mae'] = np.mean(this_metrics[fold]['overall_mae'])

            glob_gt_mae.extend(data_wise_metrics[fold]['gt_mae'])
            glob_gt_mpe.extend(data_wise_metrics[fold]['gt_mpe'])
            glob_overall_mpe.extend(data_wise_metrics[fold]['overall_mae'])

        agg_dict['global']['gt_mae'] = np.mean(glob_gt_mae)
        agg_dict['global']['gt_mpe'] = np.mean(glob_gt_mpe)
        agg_dict['global']['overall_mae'] = np.mean(glob_overall_mpe)
        return agg_dict

    gt_sup_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
    feat_cls = cls_feature_class.FeatureClass(params)

    data_wise_metrics = {}

    for data_set in os.listdir(gt_sup_dir):

        for gt_file in os.listdir(os.path.join(gt_sup_dir, data_set)):
            this_fold = int(gt_file.split('_')[0][4:])

            if this_fold in val_fold_split:
                if this_fold not in data_wise_metrics:
                    data_wise_metrics[this_fold] = {}
                    data_wise_metrics[this_fold]['gt_mae'] = []
                    data_wise_metrics[this_fold]['gt_mpe'] = []
                    data_wise_metrics[this_fold]['overall_mae'] = []

                output_file = os.path.join(ref_files_folder, gt_file)

                gt_dict = feat_cls.load_output_format_file(os.path.join(gt_sup_dir, data_set, gt_file))
                try:
                    out_dict = feat_cls.load_output_format_file(output_file)
                except:
                    continue

                nb_ref_frames = max(list(gt_dict.keys()))

                for frame in range(nb_ref_frames + 1):
                    gt_dist, gt_mask = 0, 0
                    if frame in gt_dict:
                        gt_dist = gt_dict[frame][0][4]
                        gt_mask = 1

                    out_dist = out_dict[frame][0][0]
                    out_mask = out_dict[frame][0][1]
                    pred_dist = out_dist * out_mask

                    pred_dist_gt_mask = out_dist * gt_mask

                    this_gt_mae, this_gt_mpe, this_over_mae = get_mae_mpe(gt_dist, pred_dist_gt_mask, pred_dist)
                    if gt_mask == 1:
                        data_wise_metrics[this_fold]['gt_mae'].append(this_gt_mae)
                        data_wise_metrics[this_fold]['gt_mpe'].append(this_gt_mpe)
                    data_wise_metrics[this_fold]['overall_mae'].append(this_over_mae)

    results = agg_metrics(data_wise_metrics)
    return results

def compute_dist_metrics(params, val_fold_split, ref_files_folder=None):
    '''
    Compute true distance MAE & M%E combined and individual datasets
    Returns: MAE, M%E
    '''

    # import pdb; pdb.set_trace()

    def get_mae_mpe(true_dist, pred_dist):
        '''
        Compute MAE and MPE.
        For gt=0, deno = 0.1
        '''
        mae = abs(true_dist - pred_dist)

        deno = true_dist
        if true_dist == 0:
            deno = true_dist + 0.1
        mpe = mae / deno
        return mae, mpe

    def agg_metrics(this_metrics):
        '''
        Aggregate metrics across datasets and foldwise

        dcase synth : fold 1, 2
        dcase stars : fold 3, 4
        locata: fold 5(orig eval), 6(orig dev)
        metu: fold 7, 8 Train vs test [146, 98]
        marco: fold 9, 10
        '''

        FOLD_MAP = {}
        FOLD_MAP[2] = 'D'
        FOLD_MAP[4] = 'S'
        FOLD_MAP[12] = 'L'
        FOLD_MAP[8] = 'Me'
        FOLD_MAP[10] = 'Ma'

        agg_dict = {'global': {}, 'fold': {}}
        glob_mae = []
        glob_mpe = []
        for fold in this_metrics:
            fold_name = FOLD_MAP[fold]
            agg_dict['fold'][fold_name] = {}
            agg_dict['fold'][fold_name]['mae'] = np.mean(this_metrics[fold]['mae'])
            agg_dict['fold'][fold_name]['mpe'] = np.mean(this_metrics[fold]['mpe'])

            glob_mae.extend(data_wise_metrics[fold]['mae'])
            glob_mpe.extend(data_wise_metrics[fold]['mpe'])

        agg_dict['global']['mae'] = np.mean(glob_mae)
        agg_dict['global']['mpe'] = np.mean(glob_mpe)

        return agg_dict

    perm_2 = params['permutation_2']

    gt_sup_dir = os.path.join(params['dataset_dir'], 'metadata_dev')
    feat_cls = cls_feature_class.FeatureClass(params)

    data_wise_metrics = {}

    for data_set in os.listdir(gt_sup_dir):

        for gt_file in os.listdir(os.path.join(gt_sup_dir, data_set)):
            this_fold = int(gt_file.split('_')[0][4:])

            if this_fold in val_fold_split:
                if this_fold not in data_wise_metrics:
                    data_wise_metrics[this_fold] = {}
                    data_wise_metrics[this_fold]['mae'] = []
                    data_wise_metrics[this_fold]['mpe'] = []

                output_file = os.path.join(ref_files_folder, gt_file)

                gt_dict = feat_cls.load_output_format_file(os.path.join(gt_sup_dir, data_set, gt_file))
                try:
                    out_dict = feat_cls.load_output_format_file(output_file)
                except:
                    continue

                nb_ref_frames = max(list(gt_dict.keys()))

                for frame in range(nb_ref_frames + 1):
                    gt_dist = 0
                    if frame in gt_dict:
                        gt_dist = gt_dict[frame][0][4]
                    if perm_2:
                        out_dist = out_dict[frame][0][0]
                        out_mask = out_dict[frame][0][1]
                        pred_dist = out_dist * out_mask
                    else:
                        pred_dist = out_dict[frame][0][0]

                    this_mae, this_mpe = get_mae_mpe(gt_dist, pred_dist)

                    data_wise_metrics[this_fold]['mae'].append(this_mae)
                    data_wise_metrics[this_fold]['mpe'].append(this_mpe)
    results = agg_metrics(data_wise_metrics)
    return results


class ComputeSELDResults(object):
    def __init__(
            self, params, ref_files_folder=None, use_polar_format=True
    ):
        self._use_polar_format = use_polar_format
        self._desc_dir = ref_files_folder if ref_files_folder is not None else os.path.join(params['dataset_dir'],
                                                                                            'metadata_dev')
        self._doa_thresh = params['lad_doa_thresh']

        # Load feature class
        self._feat_cls = cls_feature_class.FeatureClass(params)

        # collect reference files
        self._ref_labels = {}
        for split in os.listdir(self._desc_dir):
            for ref_file in os.listdir(os.path.join(self._desc_dir, split)):
                # Load reference description file
                gt_dict = self._feat_cls.load_output_format_file(os.path.join(self._desc_dir, split, ref_file))
                if not self._use_polar_format:
                    gt_dict = self._feat_cls.convert_output_format_polar_to_cartesian(gt_dict)
                nb_ref_frames = max(list(gt_dict.keys()))
                self._ref_labels[ref_file] = [self._feat_cls.segment_labels(gt_dict, nb_ref_frames), nb_ref_frames]

        self._nb_ref_files = len(self._ref_labels)
        self._average = params['average']

    @staticmethod
    def get_nb_files(file_list, tag='all'):
        '''
        Given the file_list, this function returns a subset of files corresponding to the tag.

        Tags supported
        'all' -
        'ir'

        :param file_list: complete list of predicted files
        :param tag: Supports two tags 'all', 'ir'
        :return: Subset of files according to chosen tag
        '''
        _group_ind = {'room': 10}
        _cnt_dict = {}
        for _filename in file_list:

            if tag == 'all':
                _ind = 0
            else:
                _ind = int(_filename[_group_ind[tag]])

            if _ind not in _cnt_dict:
                _cnt_dict[_ind] = []
            _cnt_dict[_ind].append(_filename)

        return _cnt_dict

    def get_SELD_Results(self, pred_files_path, is_jackknife=False):
        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        pred_labels_dict = {}
        eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                                   doa_threshold=self._doa_thresh, average=self._average)
        for pred_cnt, pred_file in enumerate(pred_files):
            # Load predicted output format file
            pred_dict = self._feat_cls.load_output_format_file(os.path.join(pred_files_path, pred_file))
            if self._use_polar_format:
                pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
            pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])
            # Calculated scores
            eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])
            if is_jackknife:
                pred_labels_dict[pred_file] = pred_labels
        # Overall SED and DOA scores
        ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

        if is_jackknife:
            global_values = [ER, F, LE, LR, seld_scr]
            if len(classwise_results):
                global_values.extend(classwise_results.reshape(-1).tolist())
            partial_estimates = []
            # Calculate partial estimates by leave-one-out method
            for leave_file in pred_files:
                leave_one_out_list = pred_files[:]
                leave_one_out_list.remove(leave_file)
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                                           doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(leave_one_out_list):
                    # Calculated scores
                    eval.update_seld_scores(pred_labels_dict[pred_file], self._ref_labels[pred_file][0])
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()
                leave_one_out_est = [ER, F, LE, LR, seld_scr]
                if len(classwise_results):
                    leave_one_out_est.extend(classwise_results.reshape(-1).tolist())

                # Overall SED and DOA scores
                partial_estimates.append(leave_one_out_est)
            partial_estimates = np.array(partial_estimates)

            estimate, bias, std_err, conf_interval = [-1] * len(global_values), [-1] * len(global_values), [-1] * len(
                global_values), [-1] * len(global_values)
            for i in range(len(global_values)):
                estimate[i], bias[i], std_err[i], conf_interval[i] = jackknife_estimation(
                    global_value=global_values[i],
                    partial_estimates=partial_estimates[:, i],
                    significance_level=0.05
                )
            return [ER, conf_interval[0]], [F, conf_interval[1]], [LE, conf_interval[2]], [LR, conf_interval[3]], [
                seld_scr, conf_interval[4]], [classwise_results, np.array(conf_interval)[5:].reshape(5, 13, 2) if len(
                classwise_results) else []]

        else:
            return ER, F, LE, LR, seld_scr, classwise_results

    def get_consolidated_SELD_results(self, pred_files_path, score_type_list=['all', 'room']):
        '''
            Get all categories of results.
            ;score_type_list: Supported
                'all' - all the predicted files
                'room' - for individual rooms

        '''

        # collect predicted files info
        pred_files = os.listdir(pred_files_path)
        nb_pred_files = len(pred_files)

        # Calculate scores for different splits, overlapping sound events, and impulse responses (reverberant scenes)

        print('Number of predicted files: {}\nNumber of reference files: {}'.format(nb_pred_files, self._nb_ref_files))
        print('\nCalculating {} scores for {}'.format(score_type_list, os.path.basename(pred_output_format_files)))

        for score_type in score_type_list:
            print(
                '\n\n---------------------------------------------------------------------------------------------------')
            print('------------------------------------  {}   ---------------------------------------------'.format(
                'Total score' if score_type == 'all' else 'score per {}'.format(score_type)))
            print('---------------------------------------------------------------------------------------------------')

            split_cnt_dict = self.get_nb_files(pred_files, tag=score_type)  # collect files corresponding to score_type
            # Calculate scores across files for a given score_type
            for split_key in np.sort(list(split_cnt_dict)):
                # Load evaluation metric class
                eval = SELD_evaluation_metrics.SELDMetrics(nb_classes=self._feat_cls.get_nb_classes(),
                                                           doa_threshold=self._doa_thresh, average=self._average)
                for pred_cnt, pred_file in enumerate(split_cnt_dict[split_key]):
                    # Load predicted output format file
                    pred_dict = self._feat_cls.load_output_format_file(
                        os.path.join(pred_output_format_files, pred_file))
                    if self._use_polar_format:
                        pred_dict = self._feat_cls.convert_output_format_cartesian_to_polar(pred_dict)
                    pred_labels = self._feat_cls.segment_labels(pred_dict, self._ref_labels[pred_file][1])

                    # Calculated scores
                    eval.update_seld_scores(pred_labels, self._ref_labels[pred_file][0])

                # Overall SED and DOA scores
                ER, F, LE, LR, seld_scr, classwise_results = eval.compute_seld_scores()

                print('\nAverage score for {} {} data using {} coordinates'.format(score_type,
                                                                                   'fold' if score_type == 'all' else split_key,
                                                                                   'Polar' if self._use_polar_format else 'Cartesian'))
                print('SELD score (early stopping metric): {:0.2f}'.format(seld_scr))
                print('SED metrics: Error rate: {:0.2f}, F-score:{:0.1f}'.format(ER, 100 * F))
                print('DOA metrics: Localization error: {:0.1f}, Localization Recall: {:0.1f}'.format(LE, 100 * LR))


def reshape_3Dto2D(A):
    return A.reshape(A.shape[0] * A.shape[1], A.shape[2])


if __name__ == "__main__":
    pred_output_format_files = 'results/3_11553814_dev_split0_multiaccdoa_foa_20220429142557_test'  # Path of the DCASEoutput format files
    params = parameters.get_params()
    # Compute just the DCASE final results
    score_obj = ComputeSELDResults(params)
    use_jackknife = False
    ER, F, LE, LR, seld_scr, classwise_test_scr = score_obj.get_SELD_Results(pred_output_format_files,
                                                                             is_jackknife=use_jackknife)

    print('SELD score (early stopping metric): {:0.2f} {}'.format(seld_scr[0] if use_jackknife else seld_scr,
                                                                  '[{:0.2f}, {:0.2f}]'.format(seld_scr[1][0],
                                                                                              seld_scr[1][
                                                                                                  1]) if use_jackknife else ''))
    print('SED metrics: Error rate: {:0.2f} {}, F-score: {:0.1f} {}'.format(ER[0] if use_jackknife else ER,
                                                                            '[{:0.2f},  {:0.2f}]'.format(ER[1][0],
                                                                                                         ER[1][
                                                                                                             1]) if use_jackknife else '',
                                                                            100 * F[0] if use_jackknife else 100 * F,
                                                                            '[{:0.2f}, {:0.2f}]'.format(100 * F[1][0],
                                                                                                        100 * F[1][
                                                                                                            1]) if use_jackknife else ''))
    print('DOA metrics: Localization error: {:0.1f} {}, Localization Recall: {:0.1f} {}'.format(
        LE[0] if use_jackknife else LE, '[{:0.2f}, {:0.2f}]'.format(LE[1][0], LE[1][1]) if use_jackknife else '',
        100 * LR[0] if use_jackknife else 100 * LR,
        '[{:0.2f}, {:0.2f}]'.format(100 * LR[1][0], 100 * LR[1][1]) if use_jackknife else ''))
    if params['average'] == 'macro':
        print('Classwise results on unseen test data')
        print('Class\tER\tF\tLE\tLR\tSELD_score')
        for cls_cnt in range(params['unique_classes']):
            print('{}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}\t{:0.2f} {}'.format(
                cls_cnt,
                classwise_test_scr[0][0][cls_cnt] if use_jackknife else classwise_test_scr[0][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][0][cls_cnt][0],
                                            classwise_test_scr[1][0][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][1][cls_cnt] if use_jackknife else classwise_test_scr[1][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][1][cls_cnt][0],
                                            classwise_test_scr[1][1][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][2][cls_cnt] if use_jackknife else classwise_test_scr[2][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][2][cls_cnt][0],
                                            classwise_test_scr[1][2][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][3][cls_cnt] if use_jackknife else classwise_test_scr[3][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][3][cls_cnt][0],
                                            classwise_test_scr[1][3][cls_cnt][1]) if use_jackknife else '',
                classwise_test_scr[0][4][cls_cnt] if use_jackknife else classwise_test_scr[4][cls_cnt],
                '[{:0.2f}, {:0.2f}]'.format(classwise_test_scr[1][4][cls_cnt][0],
                                            classwise_test_scr[1][4][cls_cnt][1]) if use_jackknife else ''))

    # UNCOMMENT to Compute DCASE results along with room-wise performance
    # score_obj.get_consolidated_SELD_results(pred_output_format_files)

