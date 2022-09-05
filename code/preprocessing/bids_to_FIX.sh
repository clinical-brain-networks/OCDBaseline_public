#!/bin/bash

subj=$1

# load modules
module load fsl
module load miniconda3/current
source activate fmriprep-env

# run python script to create 'pseudo-FSL' folders
python create_fix_folders.py ${subj}

## fearRev
fix_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/derivatives/fix/fearRev/

# run feat / melodic
feat ${fix_dir}${subj}/design.fsf -D ${fix_dir}${subj}/feat_${subj}.feat -I 1 -init
feat ${fix_dir}${subj}/design.fsf -D ${fix_dir}${subj}/feat_${subj}.feat -I 1 -prestats

# take care of motion parameters (for FIX feature extraction later on...)
mkdir ${fix_dir}${subj}/feat_${subj}.feat/mc/
cp ${fix_dir}${subj}/prefiltered_func_data_mcf.par ${fix_dir}${subj}/feat_${subj}.feat/mc/

# rest
fix_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/derivatives/fix/rest/

# run feat / melodic
feat ${fix_dir}${subj}/design.fsf -D ${fix_dir}${subj}/feat_${subj}.feat -I 1 -init
feat ${fix_dir}${subj}/design.fsf -D ${fix_dir}${subj}/feat_${subj}.feat -I 1 -prestats

# take care of motion parameters (for FIX feature extraction later on...)
mkdir ${fix_dir}${subj}/feat_${subj}.feat/mc/
cp ${fix_dir}${subj}/prefiltered_func_data_mcf.par ${fix_dir}${subj}/feat_${subj}.feat/mc/

# unload modules
module unload fsl
module unload miniconda3/current
echo 'MELODIC finished'