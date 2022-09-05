# %%
'''
Recoded to run on lucky2. Copies (not SYMLINK) data from BIDS and fix cleaned data

'''

import subprocess
import numpy as np
import os
from time import time
from shutil import copyfile
import sys
sys.path.append("..")
from functions.data_helpers import get_computer

# tasks
task_list = ['fearRev', 'rest']

# paths
_, proj_dir = get_computer()
fix_path = proj_dir+'data/derivatives/fix/'


def copy_fix_to_bids(subj):
    '''
    [summary]

    Parameters
    ----------
    subj : [type]
        [description]
    '''

    print(subj.upper())
    # check cleaned data exist
    for task in task_list:
        source = fix_path+task+'/'+subj+'/feat_'+subj+'.feat/filtered_func_data_clean.nii.gz'
        if os.path.isfile(source):
            print('\tClean file exists')
        else:
            print('\tClean file not found... quitting')
            sys.exit()

    # COPY original bids directory
    print('Copying original BIDS dir...')
    start = time()
    source = proj_dir+'data/bids/'+subj+'/'
    dest = proj_dir+'data/derivatives/bids-fix/'
    cmd = 'cp -r '+source+' '+dest
    #print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    print(out, err)
    print('\tTime elapsed:', round(time() - start))

    # replace old data with new clean data
    print('Copying clean data...')
    start = time()
    for task in task_list:
        source = fix_path+task+'/'+subj+'/feat_'+subj+'.feat/filtered_func_data_clean.nii.gz'
        dest = proj_dir+'data/derivatives/bids-fix/'+subj+'/func/'+subj+'_task-'+task+'_bold.nii.gz'
        # remove the original bids data in the new bids fix dir
        os.remove(dest)
        # copy the data over
        copyfile(source, dest)
    print('\tTime elapsed:', round(time() - start))

    # COPY existing anatomical preprocessing
    print('Copying freesurfer data...')
    start = time()
    source = proj_dir+'data/derivatives/fmriprep/sourcedata/freesurfer/'+subj+'/'
    dest = proj_dir+'data/derivatives/fmriprep-fix/sourcedata/freesurfer/'
    cmd = 'cp -r '+source+' '+dest
    #print(cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()
    print(out, err)
    print('\tTime elapsed:', round(time() - start))

