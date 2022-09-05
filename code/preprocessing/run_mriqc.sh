#!/bin/bash

subj=$1

# containers
mriqc_con=/mnt/lustre/working/lab_lucac/shared/x_mriqc_containers/mriqc-0.16.1.simg

# paths
project_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/
bids_dir=${project_dir}data/bids/
out_dir=${project_dir}data/derivatives/mriqc/
work_dir=${project_dir}data/scratch/mriqc/${subj}

# load modules
module load singularity/3.7.1

# run proxy script
source ~/.proxy

# run fmriprep through singularity
singularity run --cleanenv ${mriqc_con} ${bids_dir} ${out_dir} participant --participant-label ${subj} --no-sub --mem_gb 12 -w ${work_dir}

module unload singularity/3.7.1