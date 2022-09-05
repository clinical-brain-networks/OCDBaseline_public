# %%
'''
Uses FD to exclude participants.
'''
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import seaborn as sns
from functions.data_helpers import get_phenotype, get_computer

palette = ['gray', 'cornflowerblue', 'orange']
palette2 = ['gray', 'orange']
fig_dir = '../figures/'

# We've used a threshold of FD 0.5
# This threshold is arbitrary and was chosen based upon a mixture of good
# practice, e.g., Power et al., but keeping in mind we should keep as much
# data as possible.

# paths
_, proj_dir = get_computer()
postprep_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
prep_dir = proj_dir+'data/derivatives/fmriprep/'
denoise_strat = 'detrend_filtered_scrub_smooth-6mm'
subj_list = list(np.loadtxt('subject_list.txt', dtype='str'))

# cutoff of excluding participants, in minutes
# again, arbitrary
cutoff = 8

motion_df = pd.DataFrame()
for subj in subj_list:
    
    # get scrubbing results from denoised data
    json_file = postprep_dir+subj+'/func/fmripop_'+denoise_strat+'_parameters.json'

    # Opening JSON file
    f = open(json_file)

    # returns JSON object as a dict
    data = json.load(f)

    # save to df
    subj_df = pd.DataFrame()
    subj_df['subj'] = [subj]
    subj_df['group'] = get_phenotype([subj])['group'].values[0]
    subj_df['Time (min)'] = data['scrubbed_length_min']
    subj_df['Scrub percentage (%)'] = data['scrubbed_percentage']
    subj_df['Excluded'] = data['scrubbed_length_min'] < cutoff

    # Close file
    f.close()

    # get average FD for correlations
    # rest
    conf_loc = prep_dir+subj+'/func/'+subj + \
        '_task-rest_desc-confounds_timeseries.tsv'
    conf_df = pd.read_csv(conf_loc, delimiter='\t')
    fd = conf_df['framewise_displacement'].values
    avg_fd_rest = np.nanmean(fd)  # get the avg FD
    subj_df['FD_avg_rest'] = avg_fd_rest

    # rest
    conf_loc = prep_dir+subj+'/func/'+subj + \
        '_task-fearRev_desc-confounds_timeseries.tsv'
    conf_df = pd.read_csv(conf_loc, delimiter='\t')
    fd = conf_df['framewise_displacement'].values
    avg_fd_task = np.nanmean(fd)  # get the avg FD
    subj_df['FD_avg_task'] = avg_fd_task

    # put in df
    motion_df = pd.concat([motion_df, subj_df])

display(motion_df.head())
print('Total suggested exlcusions:', sum(motion_df.Excluded))

# save the subject list to text
out_df = motion_df[motion_df.Excluded == False]
subj_list_excl = out_df.subj.unique()
np.savetxt('subject_list_exclusions.txt', subj_list_excl, fmt="%s")
# %%

# set global plot properties
fig, axs = plt.subplot_mosaic("""
                              ABB
                              """, figsize=(3.5, 2.0), constrained_layout=True)
plt.rcParams['svg.fonttype'] = 'none'

# plot the time remaining
g = sns.stripplot(data=motion_df, y='Time (min)', x='group',
                  hue='Excluded', dodge=False, size=4, alpha=0.5,
                  palette=palette2, ax=axs['A'])

# change font sizes and labels
g.set_ylabel('Data remaining (min)', size=9)
g.set_xlabel('')

g.set_yticks(range(2, 14, 2))
g.set_yticklabels(range(2, 14, 2), fontsize=9)
g.set_xticks(range(0, 2))
g.set_xticklabels(['HC', 'OCD'], fontsize=9)
sns.despine()
g.get_legend().remove()

# create new df with exclusions as a seperate group
new_df = motion_df.copy()
idx = new_df.Excluded == True
new_df.loc[idx, 'group'] = 'Excluded'
# plot the correlation in FD

g = sns.scatterplot(x='FD_avg_rest', y='FD_avg_task', data=new_df, hue='group',
                    palette=palette, ax=axs['B'], alpha=0.5)

# change font sizes and labels
g.set_ylabel('Task FD average (mm)', size=9)
g.set_xlabel('Rest FD average (mm)', size=9)

ticks = np.round(np.arange(0, 0.6, 0.1), 1)
g.set_yticks(ticks)
g.set_yticklabels(ticks, fontsize=9)
g.set_xticks(ticks)
g.set_xticklabels(ticks, fontsize=9)
sns.despine()
g.get_legend().remove()

plt.savefig(fig_dir+'head_motion.jpeg')
plt.show()
# %%
