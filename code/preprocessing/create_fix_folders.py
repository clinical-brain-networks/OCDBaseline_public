'''
This code copies data from fmriprep output into FSL feat style
directories so that ICA FIX can be run.
'''

import os
import shutil
import pandas as pd
import numpy as np
import subprocess
import sys
import nibabel as nib

# input arguments
subj = sys.argv[1]
print(subj.upper())

# paths
prep_dir = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/derivatives/fmriprep/'
fix_dir = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/derivatives/fix/'

# task list
task_list = ['fearRev', 'rest']


def make_design_fsf_contents(func_for_melodic, func_mask, t1_for_melodic, nvol, TR, outputdir):
    '''
    Johan's crazy handcoded fsl feat design
    '''
    s_to_return = f'\
\n\
# FEAT version number\n\
set fmri(version) 6.00\n\
\n\
# Are we in MELODIC?\n\
set fmri(inmelodic) 0\n\
\n\
# Analysis level\n\
# 1 : First-level analysis\n\
# 2 : Higher-level analysis\n\
set fmri(level) 1\n\
\n\
# Which stages to run\n\
# 0 : No first-level analysis (registration and/or group stats only)\n\
# 7 : Full first-level analysis\n\
# 1 : Pre-processing\n\
# 2 : Statistics\n\
set fmri(analysis) 1\n\
\n\
# Use relative filenames\n\
set fmri(relative_yn) 0\n\
\n\
# Balloon help\n\
set fmri(help_yn) 1\n\
\n\
# Run Featwatcher\n\
set fmri(featwatcher_yn) 0\n\
\n\
# Cleanup first-level standard-space images\n\
set fmri(sscleanup_yn) 0\n\
\n\
# Output directory\n\
set fmri(outputdir) "{outputdir}"\n\
\n\
# TR(s)\n\
set fmri(tr) "{TR}" \n\
\n\
# Total volumes\n\
set fmri(npts) {nvol}\n\
\n\
# Delete volumes\n\
set fmri(ndelete) 0\n\
\n\
# Perfusion tag/control order\n\
set fmri(tagfirst) 1\n\
\n\
# Number of first-level analyses\n\
set fmri(multiple) 1\n\
\n\
# Higher-level input type\n\
# 1 : Inputs are lower-level FEAT directories\n\
# 2 : Inputs are cope images from FEAT directories\n\
set fmri(inputtype) 2\n\
\n\
# Carry out pre-stats processing?\n\
set fmri(filtering_yn) 1\n\
\n\
# Brain/background threshold, %\n\
set fmri(brain_thresh) 0\n\
\n\
# Critical z for design efficiency calculation\n\
set fmri(critical_z) 5.3\n\
\n\
# Noise level\n\
set fmri(noise) 0.66\n\
\n\
# Noise AR(1)\n\
set fmri(noisear) 0.34\n\
\n\
# Motion correction\n\
# 0 : None\n\
# 1 : MCFLIRT\n\
set fmri(mc) 0\n\
\n\
# Spin-history (currently obsolete)\n\
set fmri(sh_yn) 0\n\
\n\
# B0 fieldmap unwarping?\n\
set fmri(regunwarp_yn) 0\n\
\n\
# EPI dwell time (ms)\n\
set fmri(dwell) 0.7\n\
\n\
# EPI TE (ms)\n\
set fmri(te) 35\n\
\n\
# % Signal loss threshold\n\
set fmri(signallossthresh) 10\n\
\n\
# Unwarp direction\n\
set fmri(unwarp_dir) y-\n\
\n\
# Slice timing correction\n\
# 0 : None\n\
# 1 : Regular up (0, 1, 2, 3, ...)\n\
# 2 : Regular down\n\
# 3 : Use slice order file\n\
# 4 : Use slice timings file\n\
# 5 : Interleaved (0, 2, 4 ... 1, 3, 5 ... )\n\
set fmri(st) 0\n\
\n\
# Slice timings file\n\
set fmri(st_file) ""\n\
\n\
# BET brain extraction\n\
set fmri(bet_yn) 0\n\
\n\
# Spatial smoothing FWHM (mm)\n\
set fmri(smooth) 0\n\
\n\
# Intensity normalization\n\
set fmri(norm_yn) 0\n\
\n\
# Perfusion subtraction\n\
set fmri(perfsub_yn) 0\n\
\n\
# Highpass temporal filtering\n\
set fmri(temphp_yn) 1\n\
\n\
# Lowpass temporal filtering\n\
set fmri(templp_yn) 0\n\
\n\
# MELODIC ICA data exploration\n\
set fmri(melodic_yn) 1\n\
\n\
# Carry out main stats?\n\
set fmri(stats_yn) 0\n\
\n\
# Carry out prewhitening?\n\
set fmri(prewhiten_yn) 1\n\
\n\
# Add motion parameters to model\n\
# 0 : No\n\
# 1 : Yes\n\
set fmri(motionevs) 0\n\
set fmri(motionevsbeta) ""\n\
set fmri(scriptevsbeta) ""\n\
\n\
# Robust outlier detection in FLAME?\n\
set fmri(robust_yn) 0\n\
\n\
# Higher-level modelling\n\
# 3 : Fixed effects\n\
# 0 : Mixed Effects: Simple OLS\n\
# 2 : Mixed Effects: FLAME 1\n\
# 1 : Mixed Effects: FLAME 1+2\n\
set fmri(mixed_yn) 2\n\
\n\
# Number of EVs\n\
set fmri(evs_orig) 1\n\
set fmri(evs_real) 2\n\
set fmri(evs_vox) 0\n\
\n\
# Number of contrasts\n\
set fmri(ncon_orig) 1\n\
set fmri(ncon_real) 1\n\
\n\
# Number of F-tests\n\
set fmri(nftests_orig) 0\n\
set fmri(nftests_real) 0\n\
\n\
# Add constant column to design matrix? (obsolete)\n\
set fmri(constcol) 0\n\
\n\
# Carry out post-stats steps?\n\
set fmri(poststats_yn) 0\n\
\n\
# Pre-threshold masking?\n\
set fmri(threshmask) ""\n\
\n\
# Thresholding\n\
# 0 : None\n\
# 1 : Uncorrected\n\
# 2 : Voxel\n\
# 3 : Cluster\n\
set fmri(thresh) 3\n\
\n\
# P threshold\n\
set fmri(prob_thresh) 0.05\n\
\n\
# Z threshold\n\
set fmri(z_thresh) 2.3\n\
\n\
# Z min/max for colour rendering\n\
# 0 : Use actual Z min/max\n\
# 1 : Use preset Z min/max\n\
set fmri(zdisplay) 0\n\
\n\
# Z min in colour rendering\n\
set fmri(zmin) 2\n\
\n\
# Z max in colour rendering\n\
set fmri(zmax) 8\n\
\n\
# Colour rendering type\n\
# 0 : Solid blobs\n\
# 1 : Transparent blobs\n\
set fmri(rendertype) 1\n\
\n\
# Background image for higher-level stats overlays\n\
# 1 : Mean highres\n\
# 2 : First highres\n\
# 3 : Mean functional\n\
# 4 : First functional\n\
# 5 : Standard space template\n\
set fmri(bgimage) 1\n\
\n\
# Create time series plots\n\
set fmri(tsplot_yn) 1\n\
\n\
# Registration to initial structural\n\
set fmri(reginitial_highres_yn) 0\n\
\n\
# Search space for registration to initial structural\n\
# 0   : No search\n\
# 90  : Normal search\n\
# 180 : Full search\n\
set fmri(reginitial_highres_search) 90\n\
\n\
# Degrees of Freedom for registration to initial structural\n\
set fmri(reginitial_highres_dof) 3\n\
\n\
# Registration to main structural\n\
set fmri(reghighres_yn) 1\n\
\n\
# Search space for registration to main structural\n\
# 0   : No search\n\
# 90  : Normal search\n\
# 180 : Full search\n\
set fmri(reghighres_search) 90\n\
\n\
# Degrees of Freedom for registration to main structural\n\
set fmri(reghighres_dof) BBR\n\
\n\
# Registration to standard image?\n\
set fmri(regstandard_yn) 1\n\
\n\
# Use alternate reference images?\n\
set fmri(alternateReference_yn) 0\n\
\n\
# Standard image\n\
set fmri(regstandard) "/usr/share/fsl/5.0/data/standard/MNI152_T1_2mm_brain"\n\
\n\
# Search space for registration to standard space\n\
# 0   : No search\n\
# 90  : Normal search\n\
# 180 : Full search\n\
set fmri(regstandard_search) 90\n\
\n\
# Degrees of Freedom for registration to standard space\n\
set fmri(regstandard_dof) 12\n\
\n\
# Do nonlinear registration from structural to standard space?\n\
set fmri(regstandard_nonlinear_yn) 1\n\
\n\
# Control nonlinear warp field resolution\n\
set fmri(regstandard_nonlinear_warpres) 6 \n\
\n\
# High pass filter cutoff\n\
set fmri(paradigm_hp) 2000\n\
\n\
# Total voxels\n\
set fmri(totalVoxels) 437547360\n\
\n\
\n\
# Number of lower-level copes feeding into higher-level analysis\n\
set fmri(ncopeinputs) 0\n\
\n\
# 4D AVW data or FEAT directory (1)\n\
set feat_files(1) "{func_for_melodic}"\n\
\n\
# Add confound EVs text file\n\
set fmri(confoundevs) 0\n\
\n\
# Subject\'s structural image for analysis 1\n\
set highres_files(1) "{t1_for_melodic}"\n\
\n\
# EV 1 title\n\
set fmri(evtitle1) ""\n\
\n\
# Basic waveform shape (EV 1)\n\
# 0 : Square\n\
# 1 : Sinusoid\n\
# 2 : Custom (1 entry per volume)\n\
# 3 : Custom (3 column format)\n\
# 4 : Interaction\n\
# 10 : Empty (all zeros)\n\
set fmri(shape1) 0\n\
\n\
# Convolution (EV 1)\n\
# 0 : None\n\
# 1 : Gaussian\n\
# 2 : Gamma\n\
# 3 : Double-Gamma HRF\n\
# 4 : Gamma basis functions\n\
# 5 : Sine basis functions\n\
# 6 : FIR basis functions\n\
set fmri(convolve1) 2\n\
\n\
# Convolve phase (EV 1)\n\
set fmri(convolve_phase1) 0\n\
\n\
# Apply temporal filtering (EV 1)\n\
set fmri(tempfilt_yn1) 1\n\
\n\
# Add temporal derivative (EV 1)\n\
set fmri(deriv_yn1) 1\n\
\n\
# Skip (EV 1)\n\
set fmri(skip1) 0\n\
\n\
# Off (EV 1)\n\
set fmri(off1) 30\n\
\n\
# On (EV 1)\n\
set fmri(on1) 30\n\
\n\
# Phase (EV 1)\n\
set fmri(phase1) 0\n\
\n\
# Stop (EV 1)\n\
set fmri(stop1) -1\n\
\n\
# Gamma sigma (EV 1)\n\
set fmri(gammasigma1) 3\n\
\n\
# Gamma delay (EV 1)\n\
set fmri(gammadelay1) 6\n\
\n\
# Orthogonalise EV 1 wrt EV 0\n\
set fmri(ortho1.0) 0\n\
\n\
# Orthogonalise EV 1 wrt EV 1\n\
set fmri(ortho1.1) 0\n\
\n\
# Contrast & F-tests mode\n\
# real : control real EVs\n\
# orig : control original EVs\n\
set fmri(con_mode_old) orig\n\
set fmri(con_mode) orig\n\
\n\
# Display images for contrast_real 1\n\
set fmri(conpic_real.1) 1\n\
\n\
# Title for contrast_real 1\n\
set fmri(conname_real.1) ""\n\
\n\
# Real contrast_real vector 1 element 1\n\
set fmri(con_real1.1) 1\n\
\n\
# Real contrast_real vector 1 element 2\n\
set fmri(con_real1.2) 0\n\
\n\
# Display images for contrast_orig 1\n\
set fmri(conpic_orig.1) 1\n\
\n\
# Title for contrast_orig 1\n\
set fmri(conname_orig.1) ""\n\
\n\
# Real contrast_orig vector 1 element 1\n\
set fmri(con_orig1.1) 1\n\
\n\
# Contrast masking - use >0 instead of thresholding?\n\
set fmri(conmask_zerothresh_yn) 0\n\
\n\
# Do contrast masking at all?\n\
set fmri(conmask1_1) 0\n\
\n\
##########################################################\n\
# Now options that don\'t appear in the GUI\n\
\n\
# Alternative (to BETting) mask image\n\
set fmri(alternative_mask) "{func_mask}"\n\
\n\
# Initial structural space registration initialisation transform\n\
set fmri(init_initial_highres) ""\n\
\n\
# Structural space registration initialisation transform\n\
set fmri(init_highres) ""\n\
\n\
# Standard space registration initialisation transform\n\
set fmri(init_standard) ""\n\
\n\
# For full FEAT analysis: overwrite existing .feat output dir?\n\
set fmri(overwrite_yn) 0\n'
    return s_to_return


for task in task_list:
    print(task)
    fixtask_dir = fix_dir+task+'/'

    # paths for each scan we want to duplicate into the new FSL-style folder
    func = prep_dir+subj+'/func/'+subj+'_task-'+task+'_desc-preproc_bold.nii.gz'
    conf = prep_dir+subj+'/func/'+subj+'_task-'+task+'_desc-confounds_timeseries.tsv'
    fref = prep_dir+subj+'/func/'+subj+'_task-'+task+'_boldref.nii.gz'
    fmsk = prep_dir+subj+'/func/'+subj+'_task-'+task+'_desc-brain_mask.nii.gz'
    t1img = prep_dir+subj+'/anat/'+subj+'_desc-preproc_T1w.nii.gz'
    t1msk = prep_dir+subj+'/anat/'+subj+'_desc-brain_mask.nii.gz'

    # make a new subj directory
    os.makedirs(fixtask_dir+subj, exist_ok=True)

    # symlink images between fmriprep and new fix dir
    try:
        os.symlink(func, fixtask_dir+subj+'/func.nii.gz')
        os.symlink(fmsk, fixtask_dir+subj+'/func_mask.nii.gz')
        os.symlink(t1img, fixtask_dir+subj+'/t1.nii.gz')
        os.symlink(t1msk, fixtask_dir+subj+'/t1_mask.nii.gz')
    except FileExistsError:
        pass

    # manipulate and copy confounds from fmriprep
    df = pd.read_csv(conf, sep='\t')
    motion_params = df[['rot_x', 'rot_y', 'rot_z', 'trans_x', 'trans_y', 'trans_z']].values
    framewise_displacement = df['framewise_displacement'].values
    motion_file = fixtask_dir+subj+'/prefiltered_func_data_mcf.par'
    np.savetxt(motion_file, motion_params)

    # mask the T1 using fsl
    print('Masking T1')
    cmd = ('fslmaths '+fixtask_dir+subj+'/t1.nii.gz '
           + '-mas '+fixtask_dir+subj+'/t1_mask.nii.gz '
           + fixtask_dir+subj+'/t1_brain.nii.gz')
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    out, err = process.communicate()

    # copy the original data and rename it with filtered (not sure why)
    print('Copying func data')
    if not os.path.exists(fixtask_dir+subj+'/filtered_func_data.nii.gz'):
        shutil.copyfile(fixtask_dir+subj+'/func.nii.gz',
                        fixtask_dir+subj+'/filtered_func_data.nii.gz')

    # get image properties
    img = nib.load(fixtask_dir+subj+'/filtered_func_data.nii.gz')

    # make and write the design.fsf file:
    print('Writing design file')
    design = make_design_fsf_contents(fixtask_dir+subj+'/filtered_func_data.nii.gz',
                                      fixtask_dir+subj+'/func_mask.nii.gz',
                                      fixtask_dir+subj+'/t1_brain.nii.gz',
                                      img.shape[3],
                                      img.header.get_zooms()[3],
                                      fixtask_dir+subj+'/feat_'+subj)

    if not os.path.exists(fixtask_dir+subj+'/design.fsf'):
        with open(fixtask_dir+subj+'/design.fsf', 'w') as f:
            f.write(design)

    # create feat  folders
    if not os.path.exists(fixtask_dir+subj+'/feat_'+subj+'.feat'):
        os.mkdir(fixtask_dir+subj+'/feat_'+subj+'.feat')
    if not os.path.exists(fixtask_dir+subj+'/feat_'+subj+'.feat/logs'):
        os.mkdir(fixtask_dir+subj+'/feat_'+subj+'.feat/logs')
