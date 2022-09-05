'''
Functions to run ppi analyses using nilearn and afni

This code relies upon AFNI functions, if these cannot be found by calling them
in bash it will fail.

see:
    - AFNI documentation: https://afni.nimh.nih.gov/CD-CorrAna
    - Di et al., (working with Biswal) has a number of papers that were useful for this code:
        - https://doi.org/10.1007/s11682-020-00304-8
        - https://doi.org/10.3389/fnins.2017.00573 (BSC and gPPI are very similar)
        - https://doi.org/10.1002/hbm.23413 (demean your psychological variables in gPPI)
'''

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix, run_glm
from nilearn.glm.contrasts import compute_contrast
import numpy as np
import pandas as pd
import subprocess
import nibabel as nib
import os


def create_afni_ppi_term(seed, events, tr, up_rate=2, out_dir='afni_batch/'):
    '''
    Creates PPI interaction term using AFNI's deconvolution

    Parameters
    ----------
    seed : numpy.ndarray
        flat array of the seed timeseries
    events : pandas.dataframe
        pandas dataframe with 'onset', 'duration' and 'trial_type' columns as per
        bids and nilearn custom
    tr : float
        time repetition in seconds
    up_rate : int, optional
        how much to upsample the data at the 'neural' level. Be warned - this
        slows the computation down exponentially, by default 2
    out_dir : str, optional
        path where AFNI intermediate files will be stored, by default 'afni_batch/'

    Returns
    -------
    numpy.ndarray
        a TR by condition array of PPI interaction regressors
    '''

    # upsampling rate
    sub_TR = tr / up_rate

    # preallocate results array (TRs x conditions)
    ppi_array = np.zeros((len(seed), len(events.trial_type.unique())))

    # create Gamma impulse response function using AFNI
    cmd = 'waver -dt '+str(sub_TR) + \
        ' -GAM -peak 1 -inline 1@1 > '+out_dir+'GammaHR.1D'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # save the seed for AFNI
    np.savetxt(out_dir+'seed_ts.1D', seed.reshape(-1, 1))

    # upsample the seed
    if up_rate > 1:
        cmd = '1dUpsample '+str(up_rate)+' '+out_dir + \
            'seed_ts.1D > '+out_dir+'seed_ts_up.1D'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()
    else:
        np.savetxt(out_dir+'seed_ts_up.1D', seed.reshape(-1, 1))

    # run the deconvolution
    cmd = '3dTfitter -RHS '+out_dir+'seed_ts_up.1D -FALTUNG ' + \
        out_dir+'GammaHR.1D '+out_dir+'seed_neur_up 012 -1'
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    output, error = process.communicate()

    # create condition specific ppis
    for i, cond_label in enumerate(events.trial_type.unique()):

        # create condition timeseries with 1's and 0s
        # This should be demeaned, according to Biswal
        cond_events = events[events.trial_type == cond_label]
        cond_events = cond_events.reset_index(drop=True)
        cond_stim = generate_stimfunction(onsets=cond_events.onset,
                                          event_durations=cond_events.duration,
                                          total_time=len(seed) * tr,
                                          temporal_resolution=(1/tr) * up_rate)
      
        # demean and save to 1D (commented out)
        # cond_stim = cond_stim - np.mean(cond_stim)
        np.savetxt(out_dir+'condition_up.1D', np.ravel(cond_stim))

        # create interaction regressor
        cmd = "1deval -a "+out_dir+"seed_neur_up.1D\\' -b "+out_dir + \
            "condition_up.1D -expr 'a*b' > "+out_dir+"interaction_neural.1D"
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        # reconvolve
        cmd = 'waver -GAM -peak 1 -dt ' + \
            str(sub_TR)+' -input '+out_dir+'interaction_neural.1D -numout ' + \
            str(len(cond_stim))+' > '+out_dir+'ppi_up.1D'
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        output, error = process.communicate()

        # downsample the interaction regressor (e.g. by slicing or meaning)
        tmp = np.loadtxt(out_dir+'ppi_up.1D')
        tmp = tmp.reshape(len(seed), -1)
        ppi_ds = np.mean(tmp, axis=1)

        # append for later use
        ppi_array[:, i] = ppi_ds.copy()

        # remove the condition specific files (but keep deconvolved data)
        os.remove(out_dir+'condition_up.1D')
        os.remove(out_dir+'interaction_neural.1D')
        os.remove(out_dir+'ppi_up.1D')

        if process.returncode != 0:
            print('AFNI error has likely occured')
            print(output, error)

    # remove all files
    os.remove(out_dir+'GammaHR.1D')
    os.remove(out_dir+'seed_ts.1D')
    os.remove(out_dir+'seed_ts_up.1D')
    os.remove(out_dir+'seed_neur_up.1D')
    return ppi_array


def perform_ppi(ts,
                events,
                tr,
                confounds=None,
                method='seed-to-seed',
                up_rate=2,
                target_img=None,
                hrf_model='spm',
                noise_model='ar1',
                slice_time_ref=0.5,
                smooth=None,
                n_jobs=1,
                afni_dir='/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/data/scratch/afni_batch/'):
    '''
    [summary]

    Parameters
    ----------
    ts : numpy.ndarray
        a TR x ROI array
    events : pandas.dataframe
        pandas dataframe with 'onset', 'duration' and 'trial_type' columns as per
        bids and nilearn custom
    tr : float
        time repetition in seconds
    confounds : pandas.dataframe, optional
        pandas dataframe of confound signals to include in the glm, 
        by default None
    method : str, optional
        str describing methodology, either 'seed-to-seed' or 'seed-to-voxel',
        by default 'seed-to-seed'
    up_rate : int, optional
        how much to upsample the data at the 'neural' level. Be warned - this
        slows the computation down exponentially, by default 2
    target_img : str, optional
        str path to nifti target image, by default None
    hrf_model : str, optional
        hrf_model used by nilearn, by default 'spm'
    noise_model : str, optional
        noise_model used by nilearn, by default 'ar1'
    slice_time_ref : float, optional
        slice time reference used by nilearn between 0 and 1,
        by default 0.5
    smooth : int, optional
        smoothing fwhm, by default None
    n_jobs : int, optional
        number of jobs for nilearn, by default 1

    Returns
    -------
    dict or numpy.ndarray
        if seed-to-seed a (n ROI x n ROI x n Condition) matrix is returned
        if seed-to-voxel a dict is returned with n Condition levels, each of
            which contains nifti statistical maps
    '''

    # num of conditions
    n_conds = len(events.trial_type.unique())

    # num of TRs
    n_scans = len(ts)

    # preallocate results
    if method == 'seed-to-seed':
        ppi_mat = np.zeros((ts.shape[1], ts.shape[1], n_conds))
    elif method == 'seed-to-voxel':
        ppi_maps = {}

    for seed_idx in range(ts.shape[1]):
        # define seed and targets
        seed = ts[:, seed_idx]
        if method == 'seed-to-seed':
            targets = ts.copy()
            targets = np.delete(targets, seed_idx, axis=1)

        elif method == 'seed-to-voxel':
            targets = nib.load(target_img)

        # get PPI interaction terms
        ppi_array = create_afni_ppi_term(
            seed, events, tr, up_rate=up_rate,
            out_dir=afni_dir)

        # start constructing design matrix
        design_matrix = seed.reshape(-1, 1)

        # add ppi regressors
        design_matrix = np.hstack((design_matrix, ppi_array))

        # generate task evoked design elements
        # and anything else, e.g., high pass
        frame_times = tr * (np.arange(n_scans) + slice_time_ref)
        task_dm = make_first_level_design_matrix(frame_times,
                                                 events=events,
                                                 hrf_model=hrf_model,
                                                 add_regs=confounds,
                                                 high_pass=None,
                                                 drift_model=None)
        design_matrix = np.hstack((design_matrix, task_dm.values))

        # do the regression
        if method == 'seed-to-seed':
            # use the run_glm function
            labels, estimates = run_glm(
                targets, design_matrix, noise_model=noise_model, n_jobs=n_jobs)

        elif method == 'seed-to-voxel':
            # use the firstlevelglm object
            glm = FirstLevelModel(t_r=tr,
                                  slice_time_ref=slice_time_ref,
                                  hrf_model=None,
                                  drift_model=None,
                                  high_pass=None,
                                  smoothing_fwhm=smooth,
                                  signal_scaling=0,
                                  noise_model=noise_model,
                                  n_jobs=n_jobs)
            glm = glm.fit(
                targets, design_matrices=pd.DataFrame(data=design_matrix))

        # get betas for the ppi interactions
        # Remember! this relies on the fact that the design matrix
        # always has the seed first, followed by the ppi regressors
        for i in range(n_conds):
            contrast = np.zeros((1, design_matrix.shape[1]))
            contrast[0, i+1] = 1  # +1 accounts for seed timeseries

            if method == 'seed-to-seed':
                # put betas into connectivity matrix
                res_map = compute_contrast(labels, estimates, contrast)
                betas = res_map.stat()
                ppi_mat[seed_idx, seed_idx, i] = np.nan
                idx = np.isnan(ppi_mat[seed_idx, :, i])
                ppi_mat[seed_idx, ~idx, i] = betas.copy()

            elif method == 'seed-to-voxel':
                res_map = glm.compute_contrast(contrast, output_type='all')
                ppi_maps[i] = {}
                for map_type in res_map.keys():
                    ppi_maps[i][map_type] = res_map[map_type]

    if method == 'seed-to-seed':
        return ppi_mat
    elif method == 'seed-to-voxel':
        return ppi_maps

# fmrisim / brainiak is not compatible with newer version of python,
# so I have copied this small function here
# https://github.com/brainiak/brainiak/blob/master/brainiak/utils/fmrisim.py


def generate_stimfunction(onsets,
                          event_durations,
                          total_time,
                          weights=[1],
                          timing_file=None,
                          temporal_resolution=100.0,
                          ):
    """Return the function for the timecourse events
    When do stimuli onset, how long for and to what extent should you
    resolve the fMRI time course. There are two ways to create this, either
    by supplying onset, duration and weight information or by supplying a
    timing file (in the three column format used by FSL).
    Parameters
    ----------
    onsets : list, int
        What are the timestamps (in s) for when an event you want to
        generate onsets?
    event_durations : list, int
        What are the durations (in s) of the events you want to
        generate? If there is only one value then this will be assigned
        to all onsets
    total_time : int
        How long (in s) is the experiment in total.
    weights : list, float
        What is the weight for each event (how high is the box car)? If
        there is only one value then this will be assigned to all onsets
    timing_file : string
        The filename (with path) to a three column timing file (FSL) to
        make the events. Still requires total_time to work
    temporal_resolution : float
        How many elements per second are you modeling for the
        timecourse. This is useful when you want to model the HRF at an
        arbitrarily high resolution (and then downsample to your TR later).
    Returns
    ----------
    stim_function : 1 by timepoint array, float
        The time course of stimulus evoked activation. This has a temporal
        resolution of temporal resolution / 1.0 elements per second
    """

    # If the timing file is supplied then use this to acquire the
    if timing_file is not None:

        # Read in text file line by line
        with open(timing_file) as f:
            text = f.readlines()  # Pull out file as a an array

        # Preset
        onsets = list()
        event_durations = list()
        weights = list()

        # Pull out the onsets, weights and durations, set as a float
        for line in text:
            onset, duration, weight = line.strip().split()

            # Check if the onset is more precise than the temporal resolution
            upsampled_onset = float(onset) * temporal_resolution

            # Because of float precision, the upsampled values might
            # not round as expected .
            # E.g. float('1.001') * 1000 = 1000.99
            if np.allclose(upsampled_onset, np.round(upsampled_onset)) == 0:
                warning = 'Your onset: ' + str(onset) + ' has more decimal ' \
                                                        'points than the ' \
                                                        'specified temporal ' \
                                                        'resolution can ' \
                                                        'resolve. This means' \
                                                        ' that events might' \
                                                        ' be missed. ' \
                                                        'Consider increasing' \
                                                        ' the temporal ' \
                                                        'resolution.'
                logger.warning(warning)

            onsets.append(float(onset))
            event_durations.append(float(duration))
            weights.append(float(weight))

    # If only one duration is supplied then duplicate it for the length of
    # the onset variable
    if len(event_durations) == 1:
        event_durations = event_durations * len(onsets)

    if len(weights) == 1:
        weights = weights * len(onsets)

    # Check files
    if np.max(onsets) > total_time:
        raise ValueError('Onsets outside of range of total time.')

    # Generate the time course as empty, each element is a millisecond by
    # default
    stimfunction = np.zeros((int(round(total_time * temporal_resolution)), 1))

    # Cycle through the onsets
    for onset_counter in list(range(len(onsets))):
   
        # Adjust for the resolution
        onset_idx = int(np.floor(onsets[onset_counter] * temporal_resolution))

        # Adjust for the resolution
        offset_idx = int(np.floor((onsets[onset_counter] + event_durations[
            onset_counter]) * temporal_resolution))

        # Store the weights
        stimfunction[onset_idx:offset_idx, 0] = [weights[onset_counter]]

    return stimfunction
