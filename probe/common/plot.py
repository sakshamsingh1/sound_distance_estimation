import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (6,4)
%matplotlib inline
import seaborn as sns

import sys
path = '/vast/experiments/dsynth/scripts/helper/plot/'
sys.path.append(path)
from plot_utils_new import *
import matplotlib.lines as mlines

# gt_dir = '/vast/experiments/dsynth/data/processed/dist_base_loc/mic_dev_label'
gt_dir = '/vast/experiments/dsynth/data/processed/dist_d_s_basel_Am_Am/mic_dev_label'

curr_param = '713_1_dev_split0_accdoa_mic_gcc_20230428025006_test'
pred_dir = f'/dist_est_saksham/results/{curr_param}'
# pred_dir = f'/vast/sk8974/experiments/dsynth/scripts/seld_run/run/results/{curr_param}'
# df=create_error_df(pred_dir, gt_dir)
df=create_error_df(pred_dir, gt_dir)
# plot_scatter(df, title=curr_param.split('_')[0])


bin_size = 0.2
df['gt_dist_bins'] = pd.cut(df['gt_dist'], bins=np.arange(0, 4.2, bin_size))
summary_df = df.groupby('gt_dist_bins').agg({'dist_abs_error': ['mean', 'std', 'count']})
summary_df['dist_abs_error', 'sem'] = summary_df['dist_abs_error', 'std'] / np.sqrt(summary_df['dist_abs_error', 'count'])
summary_df['dist_abs_error', 'ci'] = 1.96 * summary_df['dist_abs_error', 'sem']
summary_df = summary_df.reset_index()


df_mae = summary_df.copy()
df_mape = summary_df.copy()
df_mape_thr1 = summary_df.copy()
df_mape_thr40 = summary_df.copy()

plt.rc('axes', titlesize=15)
plt.rc('axes', labelsize=15)
plt.rc('xtick',labelsize=15)
plt.rc('ytick',labelsize=15)
def plot95(dfs, title='', col='dist_abs_error', bin_size = 0.2):

    sns.set_style('whitegrid')
    plt.figure(figsize=(10, 6))
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'] # list of colors to use
    labels = ['MAE', 'MAPE', 'MAPE (1% Thr)','MAPE (40% Thr)'] # list of labels for the legend

    for i, summary_df in enumerate(dfs):
        plt.plot(summary_df['gt_dist_bins'].apply(lambda x: x.mid), summary_df['dist_abs_error', 'mean'] + summary_df['dist_abs_error', 'ci'], color=colors[i], alpha=0.2, label=labels[i])
        plt.plot(summary_df['gt_dist_bins'].apply(lambda x: x.mid), summary_df['dist_abs_error', 'mean'] - summary_df['dist_abs_error', 'ci'], color=colors[i], alpha=0.2)
        plt.fill_between(summary_df['gt_dist_bins'].apply(lambda x: x.mid), summary_df['dist_abs_error', 'mean'] + summary_df['dist_abs_error', 'ci'], summary_df['dist_abs_error', 'mean'] - summary_df['dist_abs_error', 'ci'], color=colors[i], alpha=0.2)
#         plt.plot(summary_df['gt_dist_bins'].apply(lambda x: x.mid), summary_df['dist_abs_error', 'mean'], marker='o', color=colors[i])
        plt.plot(summary_df['gt_dist_bins'].apply(lambda x: x.mid), summary_df['dist_abs_error', 'mean'], color=colors[i])

    plt.xlabel('Ground truth distance (meters)')
    plt.ylabel('Absolute prediction error (meters)')
#     plt.title('Loss comparison for Locata')

    blue_line = mlines.Line2D([], [], color='blue', label='AE')
    orange_line = mlines.Line2D([], [], color='orange',  label='APE')
    green_line = mlines.Line2D([], [], color='green', label='TAPE ($\delta=0.01$)')
    red_line = mlines.Line2D([], [], color='red', label='TAPE ($\delta=0.40$)')
#     ax.legend(handles=[blue_line])
    plt.legend(prop={'size':15},handles=[blue_line,orange_line,green_line,red_line])
#     plt.legend(prop={'size':20},handles=[blue_line,orange_line])
    plt.savefig('loss.png', dpi=300, bbox_inches='tight')
    plt.show()

dfs = [df_mae, df_mape, df_mape_thr1, df_mape_thr40]
# dfs = [df_mse, df_mspe]
plot95(dfs,title=curr_param.split('_')[0])