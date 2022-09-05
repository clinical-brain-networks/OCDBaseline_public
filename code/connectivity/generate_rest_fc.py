# %%
'''
Generates fc matrices from rest. Can be run on Lucky as it is fast.

'''
import h5py
import sys
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from time import time
from nilearn.signal import clean
sys.path.insert(0, '..')
from functions.data_helpers import get_computer

# global variables
# paths
_, proj_dir = get_computer()
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'

# subject list
subj_list = list(np.loadtxt('../subject_list.txt', dtype='str'))

# analysis variables
ts_method = 'region'
denoise_method = 'detrend_filtered_scrub'

# roi labels: these must match the extracted signal labels
# in the h5 file
# see extract_timeseries.py
roi_labels = ['leftInsula',
              'rightInsula',
              'acc',
              'pcc',
              'vmpfc',
              'leftPUT',
              'rightPUT',
              'leftCAU',
              'rightCAU',
              'leftGP',
              'rightGP']

for subj in subj_list:
    print(subj)
    
    # prepare output
    try:
        os.makedirs(deriv_dir+subj+'/fc/')
        print("Directory ", deriv_dir+subj+'/fc/', " created ")
    except FileExistsError:
        print("Directory ", deriv_dir+subj+'/fc/', " already exists")
        pass

    start = time()

    # get roi signals
    in_file = (deriv_dir+subj+'/timeseries/'+subj+'_task-rest_method-'
                + ts_method+'_desc-'+denoise_method+'.h5')
    hf = h5py.File(in_file, 'r')

    # get n_scans
    n_scans = hf[roi_labels[0]].shape[0]

    # get timeseries (stored as TR x region)
    ts = np.zeros((1, n_scans))
    for i, roi in enumerate(roi_labels):
        data = hf[roi][:]
        ts = np.vstack((ts, data.T))
    ts = np.delete(ts, 0, axis=0)  # remove the 0 row
    ts = ts.T
    hf.close()
    
    # detrend the timeseries (mirror ppi)
    ts = clean(ts, standardize=False, detrend=True)
    ts = ts.T

    # perform correlation based fc
    fc_corr = np.corrcoef(ts)

    # save output
    out_file = (deriv_dir+subj+'/fc/'+subj+'_task-rest_method-'
                + ts_method+'_corr.h5')
    hf = h5py.File(out_file, 'w')
    hf.create_dataset(name='fc', data=fc_corr)
    hf.close()

    # perform regression based fc, as per actflow toolbox
    nnodes = ts.shape[0]
    connectivity_mat = np.zeros((nnodes, nnodes))
    for targetnode in range(nnodes):
        othernodes = list(range(nnodes))
        othernodes.remove(targetnode)  # Remove target node from 'other nodes'
        X = ts[othernodes, :].T
        y = ts[targetnode, :]

        # Note: LinearRegression fits intercept by default (intercept beta not included in coef_ output)
        reg = LinearRegression().fit(X, y)
        connectivity_mat[targetnode, othernodes] = reg.coef_
    fc_reg = connectivity_mat.copy()

    # save output
    out_file = (deriv_dir+subj+'/fc/'+subj+'_task-rest_method-'
                + ts_method+'_reg.h5')
    hf = h5py.File(out_file, 'w')
    hf.create_dataset(name='fc', data=fc_reg)
    hf.close()

    finish = time()
    print('\tTime elapsed (s):', round(finish-start))
# %%
