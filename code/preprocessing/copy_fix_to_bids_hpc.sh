#!/bin/bash
# exit when any command fails
set -e

# load modules on HPC
#module load fsl
#module load R/3.4.1

# subject
subj=$1

## COPY FILES TO NEW DIRECTORY
echo "COPY FILES: subject: " ${subj}

# paths and parameters
proj_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/
fix_path=${proj_dir}data/derivatives/fix/
bids_path=${proj_dir}data/bids/
bids_fix_path=${proj_dir}data/derivatives/bids-fix/

# copy original bids directory to bids-fix
srce=${bids_path}${subj}/
dest=${bids_fix_path}
cp -r ${srce} ${dest}

# replace old fMRI data with new (clean) data
task=fearRev
srce=${fix_path}${task}/${subj}/feat_${subj}.feat/filtered_func_data_clean.nii.gz
dest=${bids_fix_path}${subj}/func/${subj}_task-${task}_bold.nii.gz
rm ${dest}  # remove orig data
cp ${srce} ${dest}

task=rest
srce=${fix_path}${task}/${subj}/feat_${subj}.feat/filtered_func_data_clean.nii.gz
dest=${bids_fix_path}${subj}/func/${subj}_task-${task}_bold.nii.gz
rm ${dest}  # remove orig data
cp ${srce} ${dest}

# copy existing anatomical preprocessing
srce=${proj_dir}data/derivatives/fmriprep/sourcedata/freesurfer/${subj}/
dest=${proj_dir}data/derivatives/fmriprep-fix/sourcedata/freesurfer/
cp -r ${srce} ${dest}

echo "FINISHED COPYING CLEAN FIX DATA"