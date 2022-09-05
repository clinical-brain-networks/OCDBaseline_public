# %%
'''
Perform statistics on connectivity results when considering
the contigency awareness

'''

from nilearn import plotting
from numpy.ma import mask_rowcols
import pingouin as pg
import seaborn as sns
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bct.nbs import nbs_bct
import sys
sys.path.insert(0, '..')
from functions.data_helpers import get_phenotype
from functions.data_helpers import get_computer
from functions.data_helpers import get_awareness_labels

def sym_by_avg(data):
    data_all = np.dstack((data, data.T))
    data_avg = np.mean(data_all, axis=2)

    mask = np.eye(data.shape[0], dtype=bool)
    data_avg[mask] = 0
    return data_avg


def save_nbs(pval, adj, thresh, perms, contrast, test, ts_method):

    df = pd.DataFrame()
    for comp in range(len(pval)):
        # get adj related to comp
        adj_comp = adj == comp+1

        # append basic information to data frame
        row = {'thresh': thresh,
               'perms': perms,
                'contrast': contrast,
                'test': test,
                'comp_n': comp+1,
                'comp_size': int(np.sum(adj_comp) / 2),
                'pval': pval[comp]
               }
        df = df.append(row, ignore_index=True)

        # save component masks out
        out = ('nbs_mats/nbs_awareness_thresh-'+str(thresh)
               + '_perms-'+str(perms)
                + '_contrast-'+contrast
                + '_test-'+test
                + '_comp_n-'+str(comp)
                + '_method-'+ts_method
                + '.csv'
               )

        np.savetxt(result_dir+out, adj_comp, delimiter=',', fmt='%i')
    return df


# global variables
# paths
_, proj_dir = get_computer()
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
roi_dir = proj_dir+'data/derivatives/masks/'
result_dir = '../../results/'
fig_dir = '../../figures/'

# subject list
subj_list = list(np.loadtxt('../subject_list_exclusions.txt', dtype='str'))

# analysis variables
ts_method = 'region'
up_rate = 3
rest_fc = 'reg'
eroded = False

# nbs settings
run_nbs = True
nbs_thresh = [1.0, 2.5, 3.0, 3.5]
perms = 10000
seed = 1990  # for replicability

# see extract_timeseries.py
roi_labels = ['Insula (L)',
              'Insula (R)',
              'dACC',
              'PCC',
              'vmPFC',
              'Putamen (L)',
              'Putamen (R)',
              'Caudate (L)',
              'Caudate (R)',
              'GP (L)',
              'GP (R)']

# preallocate ppi data
subj = subj_list[0]
if eroded:
    in_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
               + ts_method+'_ppi_up-'+str(up_rate)+'_eroded.h5')
else:
    in_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
               + ts_method+'_ppi_up-'+str(up_rate)+'.h5')
hf = h5py.File(in_file, 'r')
data = hf['ppi_mat'][:]
hf.close()

# get roi and edge numbers
n_roi = data.shape[0]
n_cond = data.shape[2]
n_edge = int((n_roi*(n_roi-1)) / 2)
con_mats = np.zeros((n_roi, n_roi, n_cond+1, len(subj_list)))

# get fc data
for s, subj in enumerate(subj_list):
    # get ppi data
    if eroded:
        in_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
                   + ts_method+'_ppi_up-'+str(up_rate)+'_eroded.h5')
    else:
        in_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
                   + ts_method+'_ppi_up-'+str(up_rate)+'.h5')

    hf = h5py.File(in_file, 'r')
    ppi_data = hf['ppi_mat'][:]
    hf.close()
    con_mats[:, :, 0:n_cond, s] = ppi_data.copy()

    # get rest data
    if eroded:
        in_file = (deriv_dir+subj+'/fc/'+subj+'_task-rest_method-'
                   + ts_method+'_'+rest_fc+'.h5')
    else:
        in_file = (deriv_dir+subj+'/fc/'+subj+'_task-rest_method-'
                   + ts_method+'_'+rest_fc+'.h5')
    hf = h5py.File(in_file, 'r')
    rest_data = hf['fc'][:]
    hf.close()
    con_mats[:, :, -1, s] = rest_data.copy()

# preprocess matrices
'''
Given regression approach, the matrix is not
symmetric, so we take the average across
upper and lower triangles

We take 5 contrasts that include resting state
and the four conditions that are used to calculate
safety and threat reversal (after accounting for
habituation).

Recall condition information:
'conditioning_CS+':0
'conditioning_CS-':2
'habituation_CS+':3
'habituation_CS-':4
'reversal_CS+':5
'reversal_CS-':7
'''

con_vec = np.zeros((len(subj_list), n_edge, 5))
con_mats_sym = np.zeros((n_roi, n_roi, 5, len(subj_list)))

for s, subj in enumerate(subj_list):

    # Threat contrast
    #  CS+REV - CS-HAB
    idx = 0
    data = (con_mats[:, :, 5, s] - con_mats[:, :, 4, s])
    con_mats_sym[:, :, idx, s] = sym_by_avg(data)

    #  CS-CON - CS-HAB
    idx = 1
    data = (con_mats[:, :, 2, s] - con_mats[:, :, 4, s])
    con_mats_sym[:, :, idx, s] = sym_by_avg(data)

    # Safety contrast
    # CS-REV - CS+HAB
    idx = 2
    data = (con_mats[:, :, 7, s] - con_mats[:, :, 2, s])
    con_mats_sym[:, :, idx, s] = sym_by_avg(data)

    # CS+CON - CS+HAB
    idx = 3
    data = (con_mats[:, :, 0, s] - con_mats[:, :, 2, s])
    con_mats_sym[:, :, idx, s] = sym_by_avg(data)

    # Resting state
    idx = 4
    data = con_mats[:, :, -1, s]
    con_mats_sym[:, :, idx, s] = sym_by_avg(data)

# # save the connectivity matrices to a text file
# if eroded:
#     np.save(result_dir+'con_mats_sym_eroded.npy', con_mats_sym)
# else:
#     np.save(result_dir+'con_mats_sym.npy', con_mats_sym)

# perform network based statistic
print('Running NBS')

nbs_df = pd.DataFrame(
    columns=['thresh', 'perms', 'contrast', 'test',
                'comp_n', 'comp_size', 'pval'])

#  Awareness based tests
group_idx = get_awareness_labels(subj_list)
for thresh in nbs_thresh:
        try:
            # Threat
            con_data = con_mats_sym[:, :, 0, :] - con_mats_sym[:, :, 1, :]
            x = con_data[:, :, group_idx == 'control-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'threat', '2samp-Control-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass

        # Safety
        try:
            con_data = con_mats_sym[:, :, 2, :] - con_mats_sym[:, :, 3, :]
            x = con_data[:, :, group_idx == 'control-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'safety', '2samp-Control-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass

        # Rest
        try:
            con_data = con_mats_sym[:, :, 4, :]
            x = con_data[:, :, group_idx == 'control-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'rest', '2samp-Control-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass

        try:
            # Threat
            con_data = con_mats_sym[:, :, 0, :] - con_mats_sym[:, :, 1, :]
            x = con_data[:, :, group_idx == 'patient-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'threat', '2samp-Patient-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass

        # Safety
        try:
            con_data = con_mats_sym[:, :, 2, :] - con_mats_sym[:, :, 3, :]
            x = con_data[:, :, group_idx == 'patient-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'safety', '2samp-Patient-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass

        # Rest
        try:
            con_data = con_mats_sym[:, :, 4, :]
            x = con_data[:, :, group_idx == 'patient-aware']
            y = con_data[:, :, group_idx == 'patient-unaware']

            pval, adj, null = nbs_bct(x, y,
                                        thresh, k=perms,
                                        tail='both', paired=False,
                                        seed=seed)
            df = save_nbs(pval, adj, thresh, perms,
                            'rest', '2samp-Patient-Unaware', ts_method)
            nbs_df = pd.concat([nbs_df, df])
        except:
            print("Oops!", sys.exc_info(), "occurred.")
            print('Carrying on...')
            pass
# save out
if eroded:
    nbs_df.to_csv(result_dir+'nbs_results_awareness_perms-'+str(perms)+ \
                    '_method-'+ts_method+'_eroded.csv', index=False)
else:
    nbs_df.to_csv(result_dir+'nbs_results_awareness_perms-'+str(perms)+ \
                    '_method-'+ts_method+'.csv', index=False)
# %%
