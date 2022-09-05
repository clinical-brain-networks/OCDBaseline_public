# %%
# # Calculates basic inter-rater reliability (irr) metrics for
# the fix classifications

import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
import sys
sys.path.insert(0, '..')
from functions.data_helpers import get_computer

# paths
_, proj_dir = get_computer()
fix_dir = proj_dir+'data/derivatives/fix/'
results_dir = '../../results/'

# tasks
task_list = ['fearRev', 'rest']

# subjects used to train classifier trainers
train_list = ['sub-control01',
              'sub-control08',
              'sub-control11',
              'sub-patient01',
              'sub-patient04',
              'sub-patient11']

# calcuate IRRs
irr_df = pd.DataFrame(columns=['subj', 'task', 'perc', 'cohen'])
for subj in train_list:
    for task in task_list:
        file_loc = fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/hand_classification_'

        rater_data = {}

        for rater in ['lh', 'cr']:
            # open the text file and only save the last line
            # count the number of components using c
            c = 0
            with open(file_loc+rater+'.txt') as f:
                for line in f:
                    c += 1
                noise_data = line
            num_comps = c-2

            # convert strings to useable array
            noise_list = str.split(noise_data[1:-2], sep=', ')
            # -1 to account for pythonic indexing
            res = [int(i)-1 for i in noise_list]
            data = np.zeros((num_comps, 1))
            data[res] = 1

            # save in dict
            rater_data[rater] = []
            rater_data[rater].append(data)

        # IRR metrics
        # percentage overlap
        res = [rater_data['lh'][0][i] == rater_data['cr'][0][i]
               for i in range(num_comps)]
        perc_overlap = (sum(res) / num_comps)

        # cohen's kappa
        # see https://en.wikipedia.org/wiki/Cohen%27s_kappa
        cohen_kappa = cohen_kappa_score(
            rater_data['lh'][0], rater_data['cr'][0])

        data = {'subj': subj, 'task': task,
                'perc': perc_overlap[0], 'cohen': cohen_kappa}
        irr_df = irr_df.append(data, ignore_index=True)

# print results to screen
print(irr_df)
print('AVERAGES:')
print(irr_df.mean())

# save
irr_df.to_csv(results_dir+'fix_stats.txt', index=False)
irr_df.mean().to_csv(results_dir+'fix_stats_avg.txt')

# %%

# Create fix result plots
# paths
# for now, assume running on Lucky2
prefix = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/data/'
fix_path = prefix+'derivatives/fix/'
df = pd.DataFrame()
model_list = ['rest', 'fearRev', 'rest_and_fearRev',
              'rest_By_fearRev', 'fearRev_By_rest']
for model in model_list:

    if os.path.isfile(fix_path+'training_files/'+model+'_results'):
        results_file = fix_path+'training_files/'+model+'_results'
    else:
        results_file = fix_path+'training_files/'+model+'_LOO_results'

    # read in the text file
    c = 0
    with open(results_file) as f:
        c = 0
        for line in f:
            if line == '\n':
                stop_row = c
                break
            c = c + 1

    # read in just the mean values
    with open(results_file) as f:
        text = [next(f) for x in range(stop_row)]

    # reformat
    thresholds = [1, 2, 5, 10, 20, 30, 40, 50]

    for i, line in enumerate(text):
        # remove /n
        line = line[0:-1]

        # convert to numbers
        values = [float(x) for x in line.split(' ')]

        # loop through thresholds
        # for plotting purposes we only care about 5 - 20
        # as recommend by FIX
        for t in range(0, 10, 2):
            tpr = values[t]
            tnr = values[t+1]
            avg = (tpr+tnr)/2
            w_avg = (3*tpr+tnr)/4

            # save to the df
            _df = pd.DataFrame()
            _df['Model'] = [model]
            _df['subj'] = i
            _df['Threshold'] = thresholds[int(t/2)]
            _df['True positive rate (TPR)'] = tpr
            _df['True negative rate (TNR)'] = tnr
            _df['(TPR+TNR)/2'] = avg
            _df['(3*TPR+TNR)/4'] = w_avg

            df = pd.concat([df, _df])

# %%

fig_dir = '../../figures/'


# PLOTS
fig, axs = plt.subplot_mosaic("""
                              ABC
                              """, figsize=(7, 2.5),
                              constrained_layout=True)
plt.rc('axes', labelsize=10)

sns.lineplot(data=df, x='Threshold', y='True positive rate (TPR)',
             hue='Model', ax=axs['A'])
plt.ylim([70, 100])

g = sns.lineplot(data=df, x='Threshold',
                 y='True negative rate (TNR)', hue='Model', ax=axs['B'])
g.get_legend().remove()
plt.ylim([70, 100])

g = sns.lineplot(data=df, x='Threshold', y='(3*TPR+TNR)/4',
                 hue='Model', ax=axs['C'])
g.get_legend().remove()
plt.ylim([70, 100])

sns.despine()
#plt.savefig(fig_dir+'FIX_results.jpeg')
plt.show()

#display(df[df.Threshold == 10].groupby('Model').mean())
#df[df.Threshold == 10].groupby('Model').mean().to_csv(results_dir+'fix_pred_stats_thr10.txt')

# %%
# # set global plot properties
# fig, axs = plt.subplot_mosaic("""
#                               ABB
#                               """, figsize=(3.5, 2.0), constrained_layout=True)
# plt.rcParams['svg.fonttype'] = 'none'

# # plot the time remaining
# g = sns.stripplot(data=motion_df, y='Time (min)', x='group',
#                   hue='Excluded', dodge=False, size=4, alpha=0.5,
#                   palette=palette2, ax=axs['A'])

# # change font sizes and labels
# g.set_ylabel('Data remaining (min)', size=9)
# g.set_xlabel('')

# g.set_yticks(range(2, 14, 2))
# g.set_yticklabels(range(2, 14, 2), fontsize=9)
# g.set_xticks(range(0, 2))
# g.set_xticklabels(['HC', 'OCD'], fontsize=9)
# sns.despine()
# g.get_legend().remove()

# # create new df with exclusions as a seperate group
# new_df = motion_df.copy()
# idx = new_df.Excluded == True
# new_df.loc[idx, 'group'] = 'Excluded'
# # plot the correlation in FD

# g = sns.scatterplot(x='FD_avg_rest', y='FD_avg_task', data=new_df, hue='group',
#                     palette=palette, ax=axs['B'], alpha=0.5)

# # change font sizes and labels
# g.set_ylabel('Task FD average (mm)', size=9)
# g.set_xlabel('Rest FD average (mm)', size=9)

# ticks = np.round(np.arange(0, 0.6, 0.1), 1)
# g.set_yticks(ticks)
# g.set_yticklabels(ticks, fontsize=9)
# g.set_xticks(ticks)
# g.set_xticklabels(ticks, fontsize=9)
# sns.despine()
# g.get_legend().remove()

# plt.savefig(fig_dir+'head_motion.jpeg')
# plt.show()