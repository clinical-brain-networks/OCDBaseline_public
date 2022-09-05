# %%
'''
A collection of python functions to help perform GLM analyses in the OCD baseline project.
Designed to interact with a nipype workflow
'''

import numpy as np
import pandas as pd
import nibabel as nib


def get_firstlevel_design(event_file, design):
    '''
    Generates a pandas df containing onsets, durations and conditions for the
    first level experiment design

    Parameters
    ----------
    subj : str
        bids-style subject label, e.g., 'sub-control01'
    task : str
        'fearRev' or 'rest'
    design : str, optional
        a str indicating what first level design should be used, by default 'Savage'

    Returns
    -------
    pandas dataframe
        a df that can be directly imported into nilearn as a design matrix
    '''
    event_df = pd.read_csv(event_file, delimiter='\t')
    if design == 'Savage':
        '''
        Savage et al., 1st level design
        1. Baseline_CS+
        2. Baseline_CS-
        3. Conditioning_CS+
        4. Conditioning_CS-
        5. Conditioning_CS+paired
        6. Reversal_CS+
        7. Reversal_CS-
        8. Reversal_CS+paired
        '''

        # manipulate conditions to match Savage et al.
        event_df = event_df[event_df.rating_measure.isna()]  # limit to actual trials
        event_df['new_trial_type'] = event_df.phase+'_'+event_df.trial_type
        for i, row in event_df.iterrows():
            if row.paired is True:
                event_df.loc[i, 'new_trial_type'] = event_df.new_trial_type[i]+'paired'
        event_df = event_df[['onset', 'duration', 'new_trial_type']]
        event_df = event_df.rename(columns={'new_trial_type': 'trial_type'})

    elif design == 'early-late':
        '''
        Similiar to Savage et al., , but split into early and late trials in each phase
        1. Baseline_CS+
        2. Baseline_CS-
        3. Early Conditioning_CS+
        4. Late Conditioning_CS+
        5. Early Conditioning_CS-
        6. Late Conditioning_CS-
        7. Conditioning_CS+paired
        8. Early Reversal_CS+
        9. Late Reversal_CS+
        10. Early Reversal_CS-
        11. Late Reversal_CS-
        12. Reversal_CS+paired
        '''
        # manipulate conditions
        # copy original savage design
        event_df = event_df[event_df.rating_measure.isna()]  # limit to actual trials
        event_df['new_trial_type'] = event_df.phase+'_'+event_df.trial_type
        for i, row in event_df.iterrows():
            if row.paired is True:
                event_df.loc[i, 'new_trial_type'] = event_df.new_trial_type[i]+'paired'
        event_df = event_df[['onset', 'duration', 'new_trial_type']]
        event_df = event_df.rename(columns={'new_trial_type': 'trial_type'})
        event_df.reset_index(inplace=True)
        del event_df['index']

        # assign trials as early or late
        for trial_type in ['CS+', 'CS-']:
            for phase in ['conditioning', 'reversal']:
                idx = event_df[event_df['trial_type'] == phase+'_'+trial_type]
                early_trials = idx.index[0:5]
                late_trials = idx.index[5::]

                event_df.iloc[early_trials, 2] = 'early_'+phase+'_'+trial_type
                event_df.iloc[late_trials, 2] = 'late_'+phase+'_'+trial_type
    return event_df


def get_firstlevel_contrasts(design, events_len, confounds_len, padding=4, n_fir=None):
    '''
    Generates a dict of contrasts that can be used in nilearn

    Parameters
    ----------
    design : str, optional
        a str indicating what first level design should be used, by default 'Savage'
    events_len : int
        number of task related regressors
    confounds_len : int
        number of confound related regressors
    padding : int, optional
        number of additional regressors (e.g., drift), by default 4

    Returns
    -------
    dict
        a dict of contrast labels and corresponding arrays
    '''
    # create dict of contrasts to be computed
    cond_len = events_len+confounds_len+padding
    if design == 'Savage':
        # preallocate zeros based on inputs
        conditions = {
            'habituation_CS+': np.zeros(cond_len),
            'habituation_CS-': np.zeros(cond_len),
            'conditioning_CS+': np.zeros(cond_len),
            'conditioning_CS-': np.zeros(cond_len),
            'reversal_CS+': np.zeros(cond_len),
            'reversal_CS-': np.zeros(cond_len)}

        # add conditions of interest
        conditions['habituation_CS+'][3] = 1.0
        conditions['habituation_CS-'][4] = 1.0
        conditions['conditioning_CS+'][0] = 1.0
        conditions['conditioning_CS-'][2] = 1.0
        conditions['reversal_CS+'][5] = 1.0
        conditions['reversal_CS-'][7] = 1.0

        # create contrasts
        contrasts = {
            'conditioning_CS+': (conditions['conditioning_CS+']
                                 - conditions['habituation_CS+']),
            'conditioning_CS-': (conditions['conditioning_CS-']
                                 - conditions['habituation_CS-']),
            'reversal_CS+': (conditions['reversal_CS+']
                             - conditions['habituation_CS-']),
            'reversal_CS-': (conditions['reversal_CS-']
                             - conditions['habituation_CS+'])}

    if design == 'early-late':
        # preallocate zeros based on inputs
        conditions = {
            'habituation_CS+': np.zeros(cond_len),
            'habituation_CS-': np.zeros(cond_len),
            'early_conditioning_CS+': np.zeros(cond_len),
            'early_conditioning_CS-': np.zeros(cond_len),
            'early_reversal_CS+': np.zeros(cond_len),
            'early_reversal_CS-': np.zeros(cond_len),
            'late_conditioning_CS+': np.zeros(cond_len),
            'late_conditioning_CS-': np.zeros(cond_len),
            'late_reversal_CS+': np.zeros(cond_len),
            'late_reversal_CS-': np.zeros(cond_len)}

        # add conditions of interest
        conditions['habituation_CS+'][5] = 1.0
        conditions['habituation_CS-'][6] = 1.0

        conditions['early_conditioning_CS+'][1] = 1.0
        conditions['early_conditioning_CS-'][2] = 1.0

        conditions['early_reversal_CS+'][3] = 1.0
        conditions['early_reversal_CS-'][4] = 1.0

        conditions['late_conditioning_CS+'][7] = 1.0
        conditions['late_conditioning_CS-'][8] = 1.0

        conditions['late_reversal_CS+'][9] = 1.0
        conditions['late_reversal_CS-'][10] = 1.0

        # create contrasts
        # note the habituation CS+/- are flipped for reversal phase to match color
        contrasts = {
            'early_conditioning_CS+': (conditions['early_conditioning_CS+']
                                       - conditions['habituation_CS+']),

            'early_conditioning_CS-': (conditions['early_conditioning_CS-']
                                       - conditions['habituation_CS-']),

            'late_conditioning_CS+': (conditions['late_conditioning_CS+']
                                      - conditions['habituation_CS+']),

            'late_conditioning_CS-': (conditions['late_conditioning_CS-']
                                      - conditions['habituation_CS-']),

            'early_reversal_CS+': (conditions['early_reversal_CS+']
                                   - conditions['habituation_CS-']),

            'early_reversal_CS-': (conditions['early_reversal_CS-']
                                   - conditions['habituation_CS+']),

            'late_reversal_CS+': (conditions['late_reversal_CS+']
                                  - conditions['habituation_CS-']),

            'late_reversal_CS-': (conditions['late_reversal_CS-']
                                  - conditions['habituation_CS+'])}

    if design == 'Savage-FIR':

        # add conditions of interest
        conditions = {}
        for i in range(n_fir):
            # pre allocate empty vec
            conditions['conditioning_CS+_'+str(i)] = np.zeros(cond_len)
            conditions['conditioning_CS+_'+str(i)][i] = 1.0  # add one for each fir

            conditions['conditioning_CS-_'+str(i)] = np.zeros(cond_len)
            conditions['conditioning_CS-_'+str(i)][2*n_fir+i] = 1.0

            conditions['habituation_CS+_'+str(i)] = np.zeros(cond_len)
            conditions['habituation_CS+_'+str(i)][3*n_fir+i] = 1.0

            conditions['reversal_CS+_'+str(i)] = np.zeros(cond_len)
            conditions['reversal_CS+_'+str(i)][5*n_fir+i] = 1.0

        # conditions['conditioning_CS+'][0] = 1.0
        # conditions['conditioning_CS-'][2] = 1.0
        # conditions['habituation_CS+'][3] = 1.0
        # conditions['habituation_CS-'][4] = 1.0
        # conditions['reversal_CS+'][5] = 1.0
        # conditions['reversal_CS-'][7] = 1.0

        # create contrasts
        contrasts = conditions

    return contrasts
# def run_firstlevel(subj):
#     '''
#     Wrapper function to run the first level GLM with nilearn

#     Parameters
#     ----------
#     subj : str
#         bids-style subject label, e.g., 'sub-control01'
#     '''
#     print(subj.upper())

#     # get 4d task bold data
#     bold_img, mask_img = get_bold_data(subj, task, bold_dir, img_space=img_space, parc=parc)

#     # get ts confounds
#     if confound_model is not None:
#         conf_df = get_bold_conf(subj, task, confound_model=confound_model)
#         n_confs = conf_df.shape[1]
#     else:
#         conf_df = None
#         n_confs = 0
    
#     # get first level events df
#     events_df = get_firstlevel_design(subj, task, firstlevel_design)

#     # get first level contrasts
#     contrasts = get_firstlevel_contrasts(firstlevel_design,
#                                          len(events_df.trial_type.unique()),
#                                          n_confs, padding=16)
#     # run glm
#     # init nilearn glm object
#     model = FirstLevelModel(t_r=bold_img.header.get_zooms()[3],
#                             slice_time_ref=0.5,
#                             hrf_model=hrf_model,
#                             drift_model=drift_model,
#                             noise_model=noise_model,
#                             smoothing_fwhm=smoothing_fwhm,
#                             high_pass=high_pass,
#                             n_jobs=8,
#                             verbose=0,
#                             mask_img=mask_img)
#     # run the glm
#     model.fit(bold_img, events_df, conf_df)

#     # create output directory
#     try:
#         os.mkdir(firstlevel_dir+subj)
#     except FileExistsError:
#         print("Directory already exists")

#     # compute contrasts
#     for i in contrasts:
#         res_map = model.compute_contrast(contrasts[i], output_type='all')

#         # save each map
#         for map_type in res_map.keys():
#             filename = subj+'_task-fearRev'+'_contrast-'+i+'_'+map_type+'.nii.gz'
#             nib.save(res_map[map_type], firstlevel_dir+subj+'/'+filename)

#     # save glm report as html
#     html_report = model.generate_report(contrasts, title=subj,
#                                         threshold=3.09, alpha=0.01,
#                                         cluster_threshold=100,
#                                         height_control='fpr',
#                                         plot_type='glass',
#                                         report_dims=(1600, 800))
#     html_report.save_as_html(firstlevel_dir+subj+'/'+subj+'_report.html')


# def get_bold_data(subj, task, bold_dir, img_space='MNI152NLin2009cAsym', parc=None):
#     '''
#     Fetch (preprocessed) bold data for a given task

#     Parameters
#     ----------
#     subj : str
#         bids-style subject label, e.g., 'sub-control01'
#     task : str
#         'fearRev' or 'rest'
#     img_space : str
#         space/image type, by default 'MNI152NLin2009cAsym'
#     parc : str, optional
#         fetch already parcellated data, by default None

#     Returns
#     -------
#     nibabel image
#         The complete nibabel image with header etc.,
#     '''
#     if img_space == 'MNI152NLin2009cAsym' and parc is None:
#         img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                     + img_space+'_desc-preproc_bold.nii.gz')
#         msk_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                     + img_space+'_desc-brain_mask.nii.gz')

#     if img_space == 'fsLR_den-91k' and parc is None:
#         img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                             + img_space+'_bold.dtseries.nii')
#         msk_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                     + img_space+'_desc-brain_mask.nii.gz')
#     elif parc == 'glasser-tian':
#         img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                     + img_space+'_parc-'+parc+'_bold.ptseries.nii')
#         msk_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
#                     + img_space+'_desc-brain_mask.nii.gz')

#     img = nib.load(img_file)
#     msk = nib.load(msk_file)
#     return img, msk


# def get_bold_conf(subj, task, confound_model='WM+CSF+6MP+FD', conf_dir=proj_dir+'data/derivatives/fmriprep/'):
#     '''
#     Reads in fmriprep generated confound .tsv file and trims it to the
#     selected confound model

#     Parameters
#     ----------
#     subj : str
#         bids-style subject label, e.g., 'sub-control01'
#     task : str
#         'fearRev' or 'rest'
#     confound_model : str, optional
#         str deliniating confounds to include, by default 'WM+CSF+6MP+FD'
#         Options include:
#             'WM+CSF+6MP+FD'
#                 - mean white matter
#                 - mean csf
#                 - 6 head motion parameters
#                 - framewise displacement
#             '24pXaCompCorXVolterra+GS'
#                 - 6 head motion parameters + derivatives + quadratics (24)
#                 - 5 anatomical comp cor components + derivs + quads (20)
#                 - Usually would include global signal but have not done that here
#     Returns
#     -------
#     pandas dataframe
#         df with confounds as columns of height nTRs
#     '''
#     # get confounds
#     conf_loc = conf_dir+subj+'/func/'+subj+'_task-'+task+'_desc-confounds_timeseries.tsv'
#     conf_df = pd.read_csv(conf_loc, delimiter='\t')

#     # select confounds
#     if confound_model == 'WM+CSF+6MP+FD':

#         confound_labels = ['white_matter', 'csf', 'framewise_displacement',
#                            'trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']
    
#     elif confound_model == '24p':
#         # create confound list
#         confound_labels = []
#         for i in ['x', 'y', 'z']:
#             confound_labels.append('trans_'+i)
#             confound_labels.append('trans_'+i+'_derivative1')
#             confound_labels.append('rot_'+i)
#             confound_labels.append('rot_'+i+'_derivative1')

#         # add quadratics
#         result = [i+'_power2' for i in confound_labels]
#         confound_labels = confound_labels + result

#         # get confounds
#         conf_df = conf_df[confound_labels]

#     elif confound_model == '24pXaCompCorXVolterra+GS':
#         # calculate the extra confounds not in fmripreps output
#         for i in range(5):
#             label = 'a_comp_cor_0'+str(i)
#             data = conf_df[label].values

#             # derivative
#             deriv = np.zeros(data.shape)
#             deriv[1::] = data[1:] - data[:-1]
#             conf_df[label+'_derivative1'] = deriv

#             # quadratic /^2
#             conf_df[label+'_power2'] = data**2

#             # derivative quadratic
#             conf_df[label+'_derivative1_power2'] = deriv**2

#         # create confound list
#         confound_labels = []
#         for i in ['x', 'y', 'z']:
#             confound_labels.append('trans_'+i)
#             confound_labels.append('trans_'+i+'_derivative1')
#             confound_labels.append('rot_'+i)
#             confound_labels.append('rot_'+i+'_derivative1')

#         for i in range(5):
#             confound_labels.append('a_comp_cor_0'+str(i))
#             confound_labels.append('a_comp_cor_0'+str(i)+'_derivative1')

#         # add quadratics
#         result = [i+'_power2' for i in confound_labels]
#         confound_labels = confound_labels + result

#     conf_df = conf_df[confound_labels]

#     # fill nans as 0s (e.g., first value in FD)
#     conf_df = conf_df.fillna(0)
#     return conf_df







# def get_secondlevel_design(secondlevel_design, subj, firstlevel_dir, input_type='stat'):
#     '''
#     Gets second level design, contrast and input files. This does not include any group level
#     contrast information
    
#     Parameters
#     ----------
#     secondlevel_design : [type]
#         [description] : [type]
#     firstlevel_dir
#         [description]
#     input_type : str, optional
#         [description], by default 'stat'

#     Returns
#     -------
#     [type]
#         [description]
#     '''

#     if secondlevel_design == 'Savage':
#         # list of *first level* contrasts we want to enter into the
#         # second level model
#         firstlevel_contrast_list = ['conditioning_CS+', 'conditioning_CS-', 'reversal_CS+', 'reversal_CS-']
#         n_contrasts = len(firstlevel_contrast_list)

#         # second level conditions in the context of a second level design matrix
#         conditions = {
#                     'conditioning_CS+': np.zeros((n_contrasts)),
#                     'conditioning_CS-': np.zeros((n_contrasts)),
#                     'reversal_CS+': np.zeros((n_contrasts)),
#                     'reversal_CS-': np.zeros((n_contrasts))}

#         # add conditions of interest
#         conditions['conditioning_CS+'][0] = 1.0
#         conditions['conditioning_CS-'][1] = 1.0
#         conditions['reversal_CS+'][2] = 1.0
#         conditions['reversal_CS-'][3] = 1.0

#         # create contrasts
#         condition_contrasts = {
#             'Threat': (conditions['conditioning_CS+'] - conditions['conditioning_CS-']
#                     + conditions['reversal_CS+'] - conditions['reversal_CS-']),
#             'ThreatRev': (conditions['reversal_CS+']
#                         - conditions['conditioning_CS-']),
#             'SafetyRev': (conditions['reversal_CS-']
#                         - conditions['conditioning_CS+'])}

#         # create design matrices
#         # subj_effect = np.tile(np.eye(n_subjs), (n_contrasts, 1))
#         design_matrix_list = []
#         secondlevel_contrast_list = []

#         # THREAT
#         design_matrix = pd.DataFrame(condition_contrasts['Threat'],
#                                     columns=['Threat'])
#         design_matrix_list.append(design_matrix)

#         # create contrast that matches the design matrix
#         contrast = np.zeros((np.shape(design_matrix)[1]))
#         contrast[0] = 1.0  # we are interested in the first col only
#         secondlevel_contrast_list.append(contrast)

#         # THREATREV
#         # create design matrix dataframe and append to list
#         design_matrix = pd.DataFrame(condition_contrasts['ThreatRev'],
#                                     columns=['ThreatRev'])
#         design_matrix_list.append(design_matrix)

#         # create contrast that matches the design matrix
#         contrast = np.zeros((np.shape(design_matrix)[1]))
#         contrast[0] = 1.0  # we are interested in the first col only
#         secondlevel_contrast_list.append(contrast)

#         # SAFETYREV
#         design_matrix = pd.DataFrame(condition_contrasts['SafetyRev'],
#                                     columns=['SafetyRev'])
#         design_matrix_list.append(design_matrix)

#         # create contrast that matches the design matrix
#         contrast = np.zeros((np.shape(design_matrix)[1]))
#         contrast[0] = 1.0  # we are interested in the first col only
#         secondlevel_contrast_list.append(contrast)

#         # get input files in the same order as the design matrix
#         secondlevel_input = []
#         for contrast in firstlevel_contrast_list:
#             filename = subj+'_task-'+task+'_contrast-'+contrast+'_'+input_type+'.nii.gz'
#             secondlevel_input.append(firstlevel_dir+subj+'/'+filename)
#     return design_matrix_list, secondlevel_contrast_list, secondlevel_input


# def run_secondlevel():
#     '''
#     Runs second level analysis in Nilearn. Second level models threat, threat reversal
#     and safety reversal. This is a single-subject contrast that makes the third-level
#     group level contrasts simplier (no within subject contrasts).
#     '''
#     from time import time

#     for subj in subj_list[86::]:
#         print(subj.upper())
#         start = time()
#         # get design details
#         dm_list, con_list, input = get_secondlevel_design(secondlevel_design, subj,
#                                                           firstlevel_dir, input_type='stat')
#         # create output directory
#         try:
#             os.mkdir(secondlevel_dir+subj)
#         except FileExistsError:
#             print("Directory already exists")

#         for design in range(len(dm_list)):

#             # get analysis label
#             label = dm_list[design].columns[con_list[design] == 1][0]

#             # create/fit second level model
#             model = SecondLevelModel(smoothing_fwhm=smoothing_fwhm, verbose=0)
#             model.fit(input, design_matrix=dm_list[design])
#             res_map = model.compute_contrast(con_list[design], output_type='all')

#             # save each map
#             for map_type in res_map.keys():
#                 filename = subj+'_task-fearRev'+'_contrast-'+label+'_'+map_type+'.nii.gz'
#                 nib.save(res_map[map_type], secondlevel_dir+subj+'/'+filename)

#             # # save glm report as html
#             # html_report = model.generate_report(con_list[design], title=label,
#             #                                     threshold=3.09, alpha=0.01,
#             #                                     cluster_threshold=100,
#             #                                     height_control='fpr',
#             #                                     plot_type='glass',
#             #                                     report_dims=(1600, 800))
#             # html_report.save_as_html(secondlevel_dir+subj+'/'+label+'_report.html')
#             print(time()-start)

