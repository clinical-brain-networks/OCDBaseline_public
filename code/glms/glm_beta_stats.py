# %%
'''
Runs one-sample and between-group t-test on the activation betas
calculated in SPM.
'''

import sys
import numpy as np
import pandas as pd
import pingouin as pg
sys.path.insert(0, '..')
from functions.data_helpers import get_computer

# global variables
# paths and default settings
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'

# beta extraction method
method = 'region'
eroded = False

# sphere radius size
radius = 4

# glm analysis stream to use
glm_label = 'smooth-6mm_despike'


def load_betas(subj_list, deriv_dir=deriv_dir, glm_label='smooth-6mm_despike', radius=4, eroded=False):
    # load in the beta activations (depending on the inputs)

    results_df = pd.DataFrame()
    for subj in subj_list:
        if eroded:
            df = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                             + '/extracted_betas/'+subj+'_Savage_'+str(radius)+'mm_eroded.csv')
        else:
            df = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                             + '/extracted_betas/'+subj+'_Savage_'+str(radius)+'mm.csv')
        results_df = pd.concat([results_df, df])

        if eroded:
            df = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                             + '/extracted_betas/'+subj+'_tian_'+str(radius)+'mm_eroded.csv')
        else:
            df = pd.read_csv(deriv_dir+'spm_group/glm_'+glm_label
                             + '/extracted_betas/'+subj+'_tian_'+str(radius)+'mm.csv')
        results_df = pd.concat([results_df, df])
    return results_df


def run_roi_stats(results_df, roi_df):

    # loop through ROI and perform statistics
    stat_df = pd.DataFrame()
    for index, row in roi_df.iterrows():

        # one sample t-tests
        a = results_df.loc[(results_df.method == method)
                           & (results_df.roi == row.label)
                           & (results_df.contrast == row.contrast)].value.values

        res = pg.ttest(a, 0)
        res['roi'] = row.ROI  # use the neater ROI label
        res['contrast'] = row.contrast
        res['test'] = '1samp'
        stat_df = pd.concat([stat_df, res])

        # two sample t-test
        a = results_df.loc[(results_df.method == method)
                           & (results_df.roi == row.label)
                           & (results_df.contrast == row.contrast)
                           & (results_df.group == 'control')].value.values

        b = results_df.loc[(results_df.method == method)
                           & (results_df.roi == row.label)
                           & (results_df.contrast == row.contrast)
                           & (results_df.group == 'patient')].value.values

        res = pg.ttest(a, b)
        res['roi'] = row.ROI
        res['contrast'] = row.contrast
        res['test'] = '2samp'
        stat_df = pd.concat([stat_df, res])

    # perform multiple comparison correction across
    # ROI for the two types of test, and the two
    # types of contrast
    stat_df['p-val-corrected'] = 1
    for contrast in ['Threat reversal', 'Safety reversal']:
        for test in ['1samp', '2samp']:
            pvals = stat_df.loc[(stat_df.test == test)
                                & (stat_df.contrast == contrast)]['p-val'].values
            reject, pvals_corr = pg.multicomp(pvals, method='fdr_by')
            stat_df.loc[(stat_df.test == test)
                        & (stat_df.contrast == contrast),
                        'p-val-corrected'] = pvals_corr

    # Make the stats dataframe more presentable
    clean_df = pd.concat([stat_df[stat_df.test == '1samp'],
                          stat_df[stat_df.test == '2samp']])
    clean_df = clean_df[['test', 'roi', 'contrast', 'T',
                         'dof', 'cohen-d', 'p-val-corrected', 'BF10']]
    clean_df['T'] = clean_df['T'].round(2)
    clean_df.dof = clean_df.dof.round(2)
    clean_df['cohen-d'] = clean_df['cohen-d'].round(2)
    clean_df['p-val-corrected'] = clean_df['p-val-corrected'].round(4)
    clean_df['BF10'] = clean_df['BF10'].astype("float")
    clean_df['BF10'] = clean_df['BF10'].round(2)
    # clean_df.to_csv('../../results/ROI_mean_stats_clean.csv')
    #clean_df.head(5)
    return clean_df

if __name__ == "__main__":

    # subject list
    subj_list = list(np.loadtxt('../subject_list_exclusions.txt', dtype='str'))

    # roi information for plotting and statistics
    roi_df = pd.read_csv('../roi_details.csv')

    results_df = load_betas(subj_list, deriv_dir,
                            glm_label=glm_label, radius=radius, eroded=eroded)
    stats_df = run_roi_stats(results_df, roi_df)
    stats_df.head(10)
