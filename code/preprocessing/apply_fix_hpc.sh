#!/bin/bash
# exit when any command fails
set -e

# load modules on HPC
module load fsl
module load R/4.0.2

# subject
subj=$1

## APPLY FIX
echo "APPLY FIX: subject: " ${subj}

# paths and parameters
fix_path='/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/derivatives/fix/'
fix_cmd='/software/fix/fix-1.06.15/fix'
train_data=${fix_path}training_files/rest_and_fearRev.RData
thresh=10
class_file='fix4melview_rest_and_fearRev_thr'${thresh}'.txt'

# FearREV
echo 'Applying classifier and cleanup: fearRev'
task='fearRev'
feat_path=${fix_path}${task}/${subj}/feat_${subj}.feat
# classifier
/software/fix/fix-1.06.15/fix -c ${feat_path} ${train_data} ${thresh}
# clean up
echo ${feat_path}/${class_file}
/software/fix/fix-1.06.15/fix -a ${feat_path}/${class_file} -m

echo 'Applying classifier and cleanup: rest'
task='rest'
feat_path=${fix_path}${task}/${subj}/feat_${subj}.feat
# classifier
/software/fix/fix-1.06.15/fix -c ${feat_path} ${train_data} ${thresh}
# clean up
/software/fix/fix-1.06.15/fix -a ${feat_path}/${class_file} -m

module purge