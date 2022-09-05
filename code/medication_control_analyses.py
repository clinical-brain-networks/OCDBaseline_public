# %%
'''
Splits patients into medicated and non-medicated
Reruns the behavioural and glm analyses


'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import pingouin as pg
from functions.data_helpers import get_phenotype, get_task_beh_data
from functions.data_helpers import get_computer

subj_list = list(np.loadtxt('subject_list_exclusions.txt', dtype='str'))
df = get_phenotype(subj_list)
df['medgroup'] = df['group'] + '-' + df['Medicated']

# % Behavioural analysis
task_df = get_task_beh_data(subj_list, ratings_only=True)
task_df['medgroup'] = np.nan

# populate the task data df with the new groups
for subj in task_df.participant_id.unique():
    new_group_value = df.loc[df.participant_id == subj, 'medgroup'].values[0]
    task_df.loc[task_df.participant_id == subj, 'medgroup'] = new_group_value
    
# quick and dirty plots
tmp = task_df[task_df.rating_measure == 'valence']
sns.catplot(data=tmp, kind='bar', x='trial_type', y='rating', hue='medgroup',
            col='phase')
plt.show()

tmp = task_df[task_df.rating_measure == 'arousal']
sns.catplot(data=tmp, kind='bar', x='trial_type', y='rating', hue='medgroup',
            col='phase')
plt.show()

# statistical analysis
# comparing medicated and unmedicated patients
group_idx = df.medgroup
res_df = pd.DataFrame()
for label in ['arousal', 'valence']:
    xdf = task_df[task_df.rating_measure == label]

    # safety rev
    srev = ((xdf.loc[(xdf.phase == 'reversal')
                    & (xdf.trial_type == 'CS-')].rating.values)
            - (xdf.loc[(xdf.phase == 'conditioning')
                      & (xdf.trial_type == 'CS+')].rating.values))
    # threat rev
    trev = ((xdf.loc[(xdf.phase == 'reversal')
                    & (xdf.trial_type == 'CS+')].rating.values)
            - (xdf.loc[(xdf.phase == 'conditioning')
                      & (xdf.trial_type == 'CS-')].rating.values))

    # do a between t-test srev
    res = pg.ttest(srev[group_idx == 'patient-Medicated'],
                   srev[group_idx == 'patient-Unmedicated'])
    res['measure'] = label
    res['contrast'] = 'Safety reversal'
    res['test'] = 'Two sample'
    res_df = pd.concat([res_df, res])

    # do a between t-test ttrev
    res = pg.ttest(trev[group_idx == 'patient-Medicated'],
                   trev[group_idx == 'patient-Unmedicated'])
    res['measure'] = label
    res['contrast'] = 'Threat reversal'
    res['test'] = 'Two sample'
    res_df = pd.concat([res_df, res])
print(res_df.head())

# % activation analysis...

# paths
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
roi_dir = proj_dir+'data/derivatives/masks/'
fig_dir = '../../figures/'
bg_img = (proj_dir + 'data/derivatives/masks/'
          + 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz')

# beta extraction method
method = 'region'
eroded = False

# sphere radius size
radius = 4

# glm analysis stream to use
glm_label = 'smooth-6mm_despike'

# roi information for plotting and statistics
roi_dict = {'Insula (L)': {'contrast': 'Threat',
                              'label': ['leftInsula'],
                              'files': [roi_dir+'Savage_threat_leftInsula.nii.gz']
                              },
            'Insula (R)': {'contrast': 'Threat',
                               'label': ['rightInsula'],
                               'files': [roi_dir+'Savage_threat_rightInsula.nii.gz']
                               },
            'dACC': {'contrast': 'Threat',
                     'label': ['acc'],
                     'files': [roi_dir+'Savage_threat_acc.nii.gz']
                     },
            'vmPFC': {'contrast': 'Safety',
                      'label': ['vmpfc'],
                      'files': [roi_dir+'Savage_safety_vmpfc.nii.gz']
                      },
            'PCC': {'contrast': 'Safety',
                    'label': ['pcc'],
                    'files': [roi_dir+'Savage_safety_pcc.nii.gz']
                    },
            'Putamen (L)': {'contrast': 'Threat',
                        'label': ['leftPUT'],
                        'files': [roi_dir+'tian_threat_leftPUT.nii.gz']
                        },
            'Putamen (R)': {'contrast': 'Threat',
                        'label': ['rightPUT'],
                        'files': [roi_dir+'tian_threat_rightPUT.nii.gz']
                        },
            'Caudate (L)': {'contrast': 'Threat',
                        'label': ['leftCAU'],
                        'files': [roi_dir+'tian_threat_leftCAU.nii.gz']
                        },
            'Caudate (R)': {'contrast': 'Threat',
                        'label': ['rightCAU'],
                        'files': [roi_dir+'tian_threat_rightCAU.nii.gz']
                                },
            'GP (L)': {'contrast': 'Threat',
                                       'label': ['leftGP'],
                                       'files': [roi_dir+'tian_threat_leftGP.nii.gz']
                                       },
            'GP (R)': {'contrast': 'Threat',
                                        'label': ['rightGP'],
                                        'files': [roi_dir+'tian_threat_rightGP.nii.gz']
                                        }
            }

results_df = pd.DataFrame()
for subj in subj_list:
    if eroded:
        xdf = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
            + '/extracted_betas/'+subj+'_Savage_'+str(radius)+'mm_eroded.csv')
    else:
        xdf = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                    + '/extracted_betas/'+subj+'_Savage_'+str(radius)+'mm.csv')
    results_df = pd.concat([results_df, xdf])

    if eroded:
        xdf = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
            + '/extracted_betas/'+subj+'_tian_'+str(radius)+'mm_eroded.csv')
    else:
        xdf = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                    + '/extracted_betas/'+subj+'_tian_'+str(radius)+'mm.csv')
    results_df = pd.concat([results_df, xdf])

# Do statistics
stat_df = pd.DataFrame()

# regions to include in statistical analysis
roi_list = ['Insula (L)',
            'Insula (R)',
            'dACC',
            'vmPFC',
            'PCC',
            'Putamen (L)',
            'Putamen (R)',
            'Caudate (L)',
            'Caudate (R)',
            'GP (L)',
            'GP (R)'
            ]

# add the new groups to the activation dataframe
for subj in results_df.subj.unique():
    new_group_value = df.loc[df.participant_id == subj, 'medgroup'].values[0]
    results_df.loc[results_df.subj == subj, 'medgroup'] = new_group_value

for roi_label in roi_list:

    # get info from dict
    roi = roi_dict[roi_label]['label'][0]

    # get contrast data
    if roi_dict[roi_label]['contrast'] == 'Threat':
        contrast = 'Threat reversal'

    elif roi_dict[roi_label]['contrast'] == 'Safety':
        contrast = 'Safety reversal'

    # two sample t-test
    a = results_df.loc[(results_df.method == method)
                       & (results_df.roi == roi)
                       & (results_df.contrast == contrast)
                       & (results_df.medgroup == 'patient-Medicated')].value.values

    b = results_df.loc[(results_df.method == method)
                       & (results_df.roi == roi)
                       & (results_df.contrast == contrast)
                       & (results_df.medgroup == 'patient-Unmedicated')].value.values

    res = pg.ttest(a, b)
    res['roi'] = roi_label
    res['contrast'] = contrast
    res['test'] = '2samp'
    stat_df = pd.concat([stat_df, res])

# perform multiple comparison correction
stat_df['p-val-corrected'] = 1
for test in ['1samp', '2samp']:
    reject, pvals_corr = pg.multicomp(
        stat_df.loc[stat_df.test == test]['p-val'].values, method='fdr_by')
    stat_df.loc[stat_df.test == test, 'p-val-corrected'] = pvals_corr

# print results
for i, row in stat_df.iterrows():
    print(row.roi, ':', row.contrast, ':', row.test,
    ': t=', np.round(row['T'], 2),
    ': p=', np.round(row['p-val-corrected'], 4))

# stat_df.to_csv('../../results/ROI_mean_stats.csv')

# Clean stat df
clean_df = pd.concat([stat_df[stat_df.test == '1samp'],
              stat_df[stat_df.test == '2samp']])
clean_df = clean_df[['test', 'roi', 'contrast', 'T', 'dof', 'cohen-d', 'p-val-corrected', 'BF10']]
clean_df['T'] = clean_df['T'].round(2)
clean_df.dof = clean_df.dof.round(2)
clean_df['cohen-d'] = clean_df['cohen-d'].round(2)
clean_df['p-val-corrected'] = clean_df['p-val-corrected'].round(4)
clean_df['BF10'] = clean_df['BF10'].astype("float")
clean_df['BF10'] = clean_df['BF10'].round(2)
# clean_df.to_csv('../../results/ROI_mean_stats_clean.csv')

# %%
clean_df.head(20)
# %%
