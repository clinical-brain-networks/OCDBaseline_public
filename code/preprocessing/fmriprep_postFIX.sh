#!/bin/bash

module purge

### Run fmriprep for specific sub
subj=$1

# fmriprep container
fmrip_con=/mnt/lustre/working/lab_lucac/shared/x_fmriprep_versions/fmriprep-20.2.1.simg

# paths
project_dir=/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/
bids_dir=${project_dir}data/derivatives/bids-fix/
out_dir=${project_dir}data/derivatives/fmriprep-fix/
work_dir=${project_dir}data/scratch/fmriprep-fix/

# load modules
module load singularity/3.7.1

# run proxy script for internet access
source ~/.proxy

# run fmriprep through singularity
# note in this second pass we ignore slice timing and
# fieldmaps as that has already been done
singularity run --cleanenv ${fmrip_con} ${bids_dir} ${out_dir} participant \
--participant-label ${subj} \
--work-dir ${work_dir} \
--skip_bids_validation \
--ignore fieldmaps slicetiming \
--output-spaces MNI152NLin2009cAsym MNI152NLin6Asym:res-2 \
--output-layout bids \
--cifti-output 91k \
--write-graph \
--nthreads 8 \
--n_cpus 8 \
--mem-mb 28000 \
--resource-monitor \
--fs-license-file /software/freesurfer/freesurfer-6.0.1/license.txt

module unload singularity/3.7.1
