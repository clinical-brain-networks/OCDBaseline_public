# %%
'''
Runs PPI within GLM regions of interest
Note that this only runs on the cluster because it needs AFNI

'''
import platform
import h5py
import sys
import os
import numpy as np
import pandas as pd
from time import time
from nilearn.signal import clean
from ppi_afni import perform_ppi
sys.path.insert(0, '..')
from glms.glm_helpers import get_firstlevel_design

def get_computer():
    '''
    Detects which computer the code is being run on using platform.
    Useful for changing paths on the hpc vs. lucky2 automatically
    Returns
    -------
    str
        'hpc' or 'lucky' identifier
    str
        project directory, e.g., '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'
    '''
    if platform.release() == '3.10.0-693.11.6.el7.x86_64':
        computer = 'hpc'
        project_directory = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/'

    elif platform.release() == '4.15.0-43-generic':
        computer = 'lucky2'
        project_directory = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'

    elif platform.release() == '5.8.0-61-generic':
        computer = 'lucky3'
        project_directory = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'

    else:
        print('Unknown computer! Assuming on the cluster...')
        print(platform.release())
        computer = 'Unknown'
        project_directory = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/'
    return computer, project_directory


# global variables
# paths
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
prep_dir = proj_dir+'data/derivatives/fmriprep/'
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
work_dir = proj_dir+'data/scratch/'

# analysis variables
firstlevel_design = 'Savage'
up_rate = 3
tr = 0.81
task = 'fearRev'
img_space = 'MNI152NLin2009cAsym'
gsr_reg = False  # global signal regression is performed within the design matrix

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


def run_ppi(subj, ts_method='region', eroded=False):
    start = time()

    # get roi signals
    if eroded:
        in_file = (deriv_dir+subj+'/timeseries/'+subj+'_task-fearRev_method-'
                   + ts_method+'_desc-detrend_eroded.h5')
    else:
        in_file = (deriv_dir+subj+'/timeseries/'+subj+'_task-fearRev_method-'
                   + ts_method+'_desc-detrend.h5')
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

    # detrend the timeseries
    ts = clean(ts, standardize=False, detrend=True)

    # get any confounds
    # get motion regressors for spike regression
    conf_loc = (prep_dir+subj+'/func/'+subj+'_task-'+task
                + '_desc-confounds_timeseries.tsv')
    fd = pd.read_csv(conf_loc, delimiter='\t').framewise_displacement.values
    spike_reg = (fd > 0.5) * 1

    # get task events / design matrix / contrasts
    event_loc = bids_dir+subj+'/func/'+subj+'_task-'+task+'_events.tsv'
    event_df = get_firstlevel_design(event_loc, firstlevel_design)

    # create ppi out dir
    afni_dir = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/scratch/afni_batch/'+subj+'/'
    try:
        os.makedirs(afni_dir)
        print("Directory ", afni_dir, " created ")
    except FileExistsError:
        print("Directory ", afni_dir, " already exists")
        pass

    # run the ppi
    ppi_mat = perform_ppi(ts, event_df, tr,
                          confounds=pd.DataFrame(
                              data=spike_reg, columns=['spike_reg']),
                          method='seed-to-seed',
                          up_rate=up_rate,
                          afni_dir=afni_dir)

    # save output
    try:
        os.makedirs(deriv_dir+subj+'/fc/')
        print("Directory ", deriv_dir+subj+'/fc/', " created ")
    except FileExistsError:
        print("Directory ", deriv_dir+subj+'/fc/', " already exists")
        pass

    if eroded:
        out_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
                    + ts_method+'_ppi_up-'+str(up_rate)+'_eroded.h5')
    else:
      out_file = (deriv_dir+subj+'/fc/'+subj+'_task-fearRev_method-'
                  + ts_method+'_ppi_up-'+str(up_rate)+'.h5')
    hf = h5py.File(out_file, 'w')
    hf.create_dataset(name='ppi_mat', data=ppi_mat)
    hf.close()
    finish = time()
    print('\tTime elapsed (s):', round(finish-start))


if __name__ == '__main__':
    # define subj
    subj = sys.argv[1]
    print(subj)

    # run all three ppi types
    print('Running PPIs...')
    run_ppi(subj, ts_method='region', eroded=False)
    run_ppi(subj, ts_method='region', eroded=True)
    print('Finished ppis')
