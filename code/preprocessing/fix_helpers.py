# %%
'''
Contains several functions for performing fix-ica training, applying such training
to unseen data and organising the hand classifications.

Usage:
Training a classifier (edit lists):
    python -c "from fix_helpers import *; train_fix()"
Apply classifier to single subject:
    python -c "from fix_helpers import *; apply_fix('sub-patient01'); fix_to_bids('sub-patient01')"
'''

import subprocess
import numpy as np
import os
import shutil
import sys
sys.path.append("..")
from functions.data_helpers import get_computer

# paths
computer, proj_dir = get_computer()
fix_dir = proj_dir+'data/derivatives/fix/'

if computer == 'lucky2':
    fix_cmd = '/data1/fix/fix/fix'
elif computer == 'hpc':
    fix_cmd = '/software/fix/fix-1.06.15/fix'

# tasks
task_list = ['fearRev', 'rest']

# subjects used to train classifier
train_list = list(np.loadtxt('subject_list_fix.txt', dtype='str'))

# threshold used when applying classifier
# consult the LOO results to decide upon this threshold
# 20 is the default
thresh = 10


def concatenate_list_data(list):
    '''
    [summary]

    Parameters
    ----------
    list : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    result = ''
    for element in list:
        result += str(element)
    return result


def Reverse(lst):
    return [ele for ele in reversed(lst)]


def extract_fix_features(subj_list):
    '''
    [summary]
    '''

    # extract features (if needed)
    print('Extracting fix features...')
    for subj in subj_list:
        for task in task_list:
            # check if fix dir exists
            if not os.path.isdir(fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/fix/'):
                cmd = fix_cmd+' -f '+fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/'
                print('\t\t', cmd)
                process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
                out, err = process.communicate()
            else:
                print('\t', subj, task, 'fix dir exists')


def train_fix():
    '''
    [summary]
    '''

    # organise hand labeled files
    for subj in train_list:
        for task in task_list:

            # get the appropriate classification file
            class_file = fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/hand_classification_'
            if os.path.isfile(class_file+'review.txt'):
                class_file = class_file+'review.txt'

            elif os.path.isfile(class_file+'lh.txt'):
                class_file = class_file+'lh.txt'
    
            elif os.path.isfile(class_file+'cr.txt'):
                class_file = class_file+'cr.txt'

            # copy it to a file fix wants
            new_class_file = fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/hand_labels_noise.txt'
            shutil.copyfile(class_file, new_class_file)
    print('Hand classifications copied...')

    # run / check if fix features are extracted
    extract_fix_features(train_list)

    # run several variants of possible classifiers
    print('Running task-specific classifier training')
    for task in task_list:
        file_list = []
        for subj in train_list:
            # string of files to be included in the training
            file_list.append(fix_dir+task+'/'+subj+'/feat_'+subj+'.feat ')
        file_str = concatenate_list_data(file_list)

        out_file = fix_dir+'training_files/'+task
        cmd = fix_cmd+' -t '+out_file+' -l '+file_str
        print('\t\t', cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()

    print('Running across-task classifier training')
    for test_task, train_task in zip(task_list, Reverse(task_list)):

        file_list = []
        for subj in train_list:
            # string of files to be included in the test data
            file_list.append(fix_dir+test_task+'/'+subj+'/feat_'+subj+'.feat ')
        file_str = concatenate_list_data(file_list)

        out_file = fix_dir+'training_files/'+test_task+'By'+train_task
        train_file = fix_dir+'training_files/'+train_task+'.RData'
        cmd = fix_cmd+' -C '+train_file+' '+out_file+' '+file_str
        print('\t\t', cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()

    print('Running combined classifier training')
    file_list = []
    for task in task_list:
        for subj in train_list:
            # string of files to be included in the training
            file_list.append(fix_dir+task+'/'+subj+'/feat_'+subj+'.feat ')
    file_str = concatenate_list_data(file_list)

    out_file = fix_dir+'training_files/rest_and_fearRev'
    cmd = fix_cmd+' -t '+out_file+' -l '+file_str
    print('\t\t', cmd)
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()


def apply_fix(subj):
    '''
    [summary]

    Parameters
    ----------
    subj : [type]
        [description]
    '''
    # extract features
    extract_fix_features([subj])

    # running 'rest_and_fearRev' classifier
    print('Running classifier')
    for task in task_list:
        training_data = fix_dir+'training_files/rest_and_fearRev.RData'
        cmd = (fix_cmd+' -c '+fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/'+' '
               + training_data+' '+str(thresh))
        print('\t\t', cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()

    # apply cleanup
    print('Running cleanup')
    for task in task_list:
        class_file = 'fix4melview_rest_and_fearRev_thr'+str(thresh)+'.txt'
        cmd = (fix_cmd+' -a '+fix_dir+task+'/'+subj+'/feat_'+subj+'.feat/'+class_file+' -m')
        print('\t\t', cmd)
        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()


# %%
# run apply on lucky2
subj_list = list(np.loadtxt('/home/lukeh/projects/OCDbaseline/docs/code/subject_list.txt', dtype='str'))

for subj in subj_list[79:80]:
    print(subj)
    apply_fix(subj)

# %%
