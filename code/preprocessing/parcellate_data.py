# %%
'''
Downsamples fmriprep cifti data into parcellation
Assumes you have access to connectome workbench command line
'''

import numpy as np
import subprocess
import sys
sys.path.append("..")
from functions.data_helpers import get_computer

# paths
computer, _ = get_computer()
if computer == 'lucky2':
    prefix = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/data/'
    parc_prefix = '/home/lukeh/hpcworking/shared/parcellations/'

elif computer == 'hpc':
    prefix = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/'
    parc_prefix = '/mnt/lustre/working/lab_lucac/shared/parcellations/'


#prep_dir = prefix+'derivatives/fmriprep/'
prep_dir = prefix+'derivatives/fmriprep-fix/'

# parcellation info
img_space = 'fsLR_den-91k'  # HCP grayordinate space (cortex in fsLR, subcortical in MNI152NLin6Asym)

parcellation = 'glasser-tian'

if parcellation == 'glasser-tian':
    # "scale 2 " subcortical parcellation (32 regions)
    parc_file = (parc_prefix+'Tian2020MSA_v1.1/3T/Cortex-Subcortex/'
                 + 'Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors'
                 + '.32k_fs_LR_Tian_Subcortex_S2.dlabel.nii')

# subj list
#subj_list = list(np.loadtxt('../subject_list_tmp.txt', dtype='str'))
subj_list = ['sub-patient44', 'sub-patient45',
'sub-patient46',
'sub-patient47',
'sub-patient48',
'sub-patient49',
'sub-patient50']

# tasks
task_list = ['fearRev', 'rest']

# loop through subjects and downsample
for subj in subj_list:
    print(subj.upper())

    for task in task_list:
        print('\t', task)

        # input file
        img_file = (prep_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                    + img_space+'_bold.dtseries.nii')

        # output file
        out_img_file = (prep_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                        + img_space+'_parc-'+parcellation+'_bold.ptseries.nii')

        # downsample the data
        print('\t\tDownsampling grayordinate data to parcels')
        cmd = ('wb_command -cifti-parcellate '+img_file+' '+parc_file
               + ' COLUMN '+out_img_file+' -method MEAN')

        process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
        out, err = process.communicate()

# %%
