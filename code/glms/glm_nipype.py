'''
Nipype functions and workflows to run glms in SPM12
This must be run within the docker/singularity container

'''
import numpy as np
from nipype import Node, Function, Workflow, IdentityInterface
from nipype.interfaces.io import SelectFiles, DataSink
from nipype.interfaces import spm
from nipype.algorithms.misc import Gunzip
from nipype.algorithms.modelgen import SpecifySPMModel
from glm_helpers import get_firstlevel_design
from shutil import copytree

# Analysis settings
# directories
bids_dir = '/data/bids/'
prep_dir = '/data/fmriprep/'
fix_dir = '/data/fmriprep-fix/'
post_fix_dir = '/data/post-fmriprep-fix/'
work_dir = '/data/work_dir/'

# subject list
subj_list = list(np.loadtxt('../subject_list.txt', dtype='str'))
subj_list = ['sub-control28']
subj_list_secondlvl = list(np.loadtxt('../subject_list_exclusions.txt', dtype='str'))

# task to analyze
task = 'fearRev'

# space of data
space = 'MNI152NLin2009cAsym'

# Smoothing widths to apply
fwhm = [6, 6, 6]
smooth_label = '6mm'

# analysis label
analysis_label = '6mm_despike'

# glm designs
firstlevel_design = 'Savage'
secondlevel_design = 'Savage'

# number of processes
n_procs = 24

# mask
group_mask = '/home/code/glms/spm_mask.nii'

# Condition / contrast information
if firstlevel_design == 'Savage':
    condition_names = ['conditioning_CS+',
                       'conditioning_CS+paired',
                       'conditioning_CS-',
                       'habituation_CS+',
                       'habituation_CS-',
                       'reversal_CS+',
                       'reversal_CS+paired',
                       'reversal_CS-']

    # Contrasts
    cont01 = ['conditioning_CS+', 'T',
              condition_names, [1, 0, 0, -1, 0, 0, 0, 0]]
    cont02 = ['conditioning_CS-', 'T',
              condition_names, [0, 0, 1, 0, -1, 0, 0, 0]]
    cont03 = ['reversal_CS+',     'T',
              condition_names, [0, 0, 0, 0, -1, 1, 0, 0]]
    cont04 = ['reversal_CS-',     'T',
              condition_names, [0, 0, 0, -1, 0, 0, 0, 1]]
    cont05 = ['CS+ > CS-',        'T',
              condition_names, [1, 0, -1, 0, 0, 1, 0, -1]]
    cont06 = ['ThreatRev',        'T',
              condition_names, [0, 0, -1, 0, 0, 1, 0, 0]]
    cont07 = ['SafetyRev',        'T',
              condition_names, [-1, 0, 0, 0, 0, 0, 0, 1]]
    cont08 = ['conditio_CS+>CS-', 'T',
              condition_names, [1, 0, -1, -1, 1, 0, 0, 0]]
    cont09 = ['reversal_CS+>CS-', 'T',
              condition_names, [0, 0, 0, 1, -1, 1, 0, -1]]

    contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08,
                     cont09]


if secondlevel_design == 'Savage':
    second_level_contrast_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004',
        'con_0005', 'con_0006', 'con_0007', 'con_0008', 'con_0009']

# Second level stats
cluster = 0.05  # cluster threshold in p-value
height = 0.001  # height threshold in p-value
extent = 10  # extent in voxels


def get_nipype_TR(func):
    '''
    nipype-friendly function to get the TR of a BOLD image

    Parameters
    ----------
    func : path to bold img
    '''
    import nibabel as nib
    imgfile = (func)
    bold_img = nib.load(imgfile)
    TR = bold_img.header.get_zooms()[3]
    return TR


def get_nipype_subject_info(event_file, conf_file, design):
    '''
    Nipype-friendly function to take a given event file ('.tsv')
    manipulate in accordance with the given design and then
    translate into a nipype Bunch format for spm
    Includes a regressor for time point scrubbing - consider the 
    threshold carefully, see https://onlinelibrary.wiley.com/doi/epdf/10.1002/hbm.22307

    Parameters
    ----------
    event_file : path to .tsv file
        bids-style task event file
    design : str
        e.g., 'Savage', or 'early-late'
    '''

    import pandas as pd
    from glm_helpers import get_firstlevel_design
    from nipype.interfaces.base import Bunch

    # get events organised according to firstlevel_design
    events_df = get_firstlevel_design(event_file, design)

    # get a few minimal confounds
    # get motion regressors (censoring)
    fd = pd.read_csv(conf_file, delimiter='\t').framewise_displacement.values
    spike_reg = (fd > 0.5) * 1

    # white matter / csf
    # white_matter = pd.read_csv(conf_file, delimiter='\t').white_matter.values
    # csf = pd.read_csv(conf_file, delimiter='\t').csf.values

    # translate to nipype / spm
    conditions = []
    onsets = []
    durations = []

    for group in events_df.groupby('trial_type'):
        conditions.append(group[0])
        onsets.append(list(group[1].onset))
        durations.append(group[1].duration.tolist())

    subject_info = [Bunch(conditions=conditions,
                          onsets=onsets,
                          durations=durations,
                          regressor_names=['spike_reg'],
                          regressors=[list(spike_reg)]
                          )]
    return subject_info


def first_level_wf():
    '''
    [summary]

    Returns
    -------
    [type]
        [description]
    '''
    # iterable node
    info_source = Node(IdentityInterface(fields=['subject_id', 'contrasts'],
                                         contrasts=contrast_list),
                       name='info_source')
    info_source.iterables = [('subject_id', subj_list)]

    # Select functional data node
    # String template with {}-based strings
    templates = {'func': ('{subject_id}/func/{subject_id}_task-'+task+'_'
                          + 'space-'+space+'_desc-detrend.nii.gz')}

    # Create SelectFiles node
    select_files = Node(SelectFiles(templates, base_directory=post_fix_dir),
                        name='select_files')

    # unzip .nii.gz data for SPM
    gunzip = Node(Gunzip(), name='gunzip')

    # smooth the data
    smooth = Node(spm.Smooth(fwhm=fwhm), name='smooth')

    # save smoothed data via datasink
    sinker = Node(DataSink(base_directory=work_dir+'smooth'), name='sinker')
    substitutions = [('_subject_id_', '')]  # Define substitution strings
    sinker.inputs.substitutions = substitutions

    # First Level
    # get subject TR
    getsubjectTR = Node(Function(input_names=['func'],
                                 output_names=['TR'],
                                 function=get_nipype_TR),
                        name='getsubjectTR')

    # Select event files
    event_template = {'event': ('{subject_id}/func/{subject_id}_task-'+task
                                + '_events.tsv')}

    # Select confound files
    conf_template = {'conf': ('{subject_id}/func/{subject_id}_task-'+task
                              + '_desc-confounds_timeseries.tsv')}

    # Create SelectFiles node (event files)
    select_event_files = Node(SelectFiles(event_template, base_directory=bids_dir),
                              name='select_event_files')

    # Create SelectFiles node (confounds file)
    select_conf_files = Node(SelectFiles(conf_template, base_directory=prep_dir),
                             name='select_conf_files')

    # Get Subject Info - get subject specific condition information
    getsubjectinfo = Node(Function(input_names=['event_file', 'conf_file', 'design'],
                                   output_names=['subject_info'],
                                   function=get_nipype_subject_info),
                          name='getsubjectinfo')
    getsubjectinfo.inputs.design = firstlevel_design

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     high_pass_filter_cutoff=128),
                     name="modelspec")

    # Level1Design - Generates an SPM design matrix
    level1design = Node(spm.Level1Design(bases={'hrf': {'derivs': [1, 0]}},
                                         timing_units='secs',
                                         model_serial_correlations='FAST'),
                        name="level1design")
    level1design.inputs.mask_image = group_mask
    level1design.inputs.mask_threshold = 0.5

    # EstimateModel - estimate the parameters of the model
    level1estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                          name="level1estimate")

    # EstimateContrast - estimates contrasts
    level1con = Node(spm.EstimateContrast(), name="level1con")

    # save first level files
    sinker1level = Node(DataSink(base_directory=work_dir+'1stlvl/'),
                        name='sinker1level')
    substitutions = [('_subject_id_', '')]
    sinker1level.inputs.substitutions = substitutions

    # workflow construction for first level analysis
    # create workflow
    wf = Workflow(name='postPrep', base_dir=work_dir)

    # link file selector and gunzipping
    wf.connect(info_source, 'subject_id', select_files, 'subject_id')
    wf.connect(select_files, 'func', gunzip, 'in_file')
    wf.connect(select_files, 'func', getsubjectTR, 'func')

    # do smoothing
    wf.connect(gunzip, 'out_file', smooth, 'in_files')
    wf.connect(smooth, 'smoothed_files', sinker, 'smooth-'+smooth_label)

    # link subject ids and get subject information
    wf.connect(info_source, 'subject_id', select_event_files, 'subject_id')
    wf.connect(info_source, 'subject_id', select_conf_files, 'subject_id')
    wf.connect(select_event_files, 'event', getsubjectinfo, 'event_file')
    wf.connect(select_conf_files, 'conf', getsubjectinfo, 'conf_file')

    # link subject info and data to SPM model
    wf.connect(getsubjectTR, 'TR', modelspec, 'time_repetition')
    wf.connect(getsubjectinfo, 'subject_info', modelspec, 'subject_info')
    wf.connect(sinker, 'out_file', modelspec, 'functional_runs')
    wf.connect(getsubjectTR, 'TR', level1design, 'interscan_interval')
    wf.connect(modelspec, 'session_info', level1design, 'session_info')
    wf.connect(info_source, 'contrasts', level1con, 'contrasts')

    # link design to estimate to contrasts
    wf.connect(level1design, 'spm_mat_file', level1estimate, 'spm_mat_file')
    wf.connect(level1estimate, 'spm_mat_file', level1con, 'spm_mat_file')
    wf.connect(level1estimate, 'beta_images', level1con, 'beta_images')
    wf.connect(level1estimate, 'residual_image', level1con, 'residual_image')

    # same the images out
    wf.connect(level1con, 'spm_mat_file', sinker1level, '@spm_mat')
    wf.connect(level1con, 'spmT_images', sinker1level, '@T')
    wf.connect(level1con, 'con_images', sinker1level, '@con')
    wf.connect(level1con, 'ess_images', sinker1level, '@ess')
    return wf


def move_firstlevel():
    # small helper function to avoid renaming things in datasink
    for subj in subj_list:
        source = work_dir+'1stlvl/'+subj
        dest = post_fix_dir+subj+'/spm/glm_'+analysis_label
        copytree(source, dest)


def secondlevel_files_1samp(contrast_id, subj_list, post_fix_dir, analysis_label):
    # returns list of subject list specific files
    cons = []
    for subj in subj_list:
        f = post_fix_dir+subj+'/spm/glm_'+analysis_label+'/'+contrast_id+'.nii'
        cons.append(f)
    return cons


def secondlevel_files_2samp(contrast_id, subj_list, post_fix_dir, analysis_label):
    # returns list of subject list specific files
    control_cons = []
    patient_cons = []
    for subj in subj_list:
        f = post_fix_dir+subj+'/spm/glm_'+analysis_label+'/'+contrast_id+'.nii'
        if f.find('control') > 0:
                control_cons.append(f)
        elif f.find('patient') > 0:
                patient_cons.append(f)
    return control_cons, patient_cons


def secondlevel_1samp_wf():
    # Infosource - a function free node to iterate over contrasts
    info_source = Node(IdentityInterface(fields=['contrast_id']),
                       name="info_source")
    info_source.iterables = [('contrast_id', second_level_contrast_list)]

    # Select files with custom function
    select_files = Node(Function(input_names=['contrast_id',
                                              'subj_list',
                                              'post_fix_dir',
                                              'analysis_label'],
                        output_names=['cons'],
                        function=secondlevel_files_1samp),
               name='select_files')
    select_files.inputs.subj_list = subj_list_secondlvl
    select_files.inputs.post_fix_dir = post_fix_dir
    select_files.inputs.analysis_label = analysis_label

    # OneSampleTTestDesign - creates one sample T-Test Design
    onesamplettestdes = Node(spm.OneSampleTTestDesign(),
                             name="onesampttestdes")
    onesamplettestdes.inputs.explicit_mask_file = group_mask

    # EstimateModel - estimates the model
    level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    # EstimateContrast - estimates group contrast
    level2con = Node(spm.EstimateContrast(group_contrast=True),
                     name="level2con")
    level2con.inputs.contrasts = [['Group', 'T', ['mean'], [1]]]

    # Threshold - thresholds contrasts
    level2thresh = Node(spm.Threshold(contrast_index=1,
                                      extent_fdr_p_threshold=0.01,
                                      extent_threshold=10,
                                      height_threshold=0.001,
                                      height_threshold_type='p-value',
                                      use_fwe_correction=False,
                                      use_topo_fdr=True),
                        name="level2thresh")

    # Save output (datasink)
    sinker2level = Node(DataSink(base_directory=post_fix_dir+'spm_group/'),
                        name='sinker2level')
    substitutions = [('_contrast_id_', '')]  # Define substitution strings
    sinker2level.inputs.substitutions = substitutions

    # create workflow and connections
    # Initiation of the 2nd-level analysis workflow
    wf = Workflow(name='postPrep2ndlevel', base_dir=work_dir)

    # Connect up the 2nd-level analysis components
    wf.connect([(info_source, select_files, [('contrast_id', 'contrast_id')]),
                (select_files, onesamplettestdes, [('cons', 'in_files')]),
                (onesamplettestdes, level2estimate, [
                 ('spm_mat_file', 'spm_mat_file')]),
                (level2estimate, level2con, [('spm_mat_file', 'spm_mat_file'),
                                             ('beta_images', 'beta_images'),
                                             ('residual_image', 'residual_image')]),
                (level2con, level2thresh, [('spm_mat_file', 'spm_mat_file'),
                                           ('spmT_images', 'stat_image')]),
                (level2con, sinker2level, [('spm_mat_file', 'glm_'+analysis_label+'.1samp'+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@spm_mat'),
                                           ('spmT_images', 'glm_'+analysis_label+'.1samp'+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@T'),
                                           ('con_images', 'glm_'+analysis_label+'.1samp'+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@con')]),
                (level2thresh, sinker2level, [
                 ('thresholded_map', 'glm_'+analysis_label+'.1samp'+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@threshold')])
                ])
    return wf


def secondlevel_2samp_wf(con='HC>PAT'):

    # I couldn't figure a way to iterate over the two sample contrast switch
    if con == 'HC>PAT':
        con_input = ('Controls>Patients', 'T', ['Group_{1}', 'Group_{2}'], [1, -1])
    elif con == 'PAT>HC':
        con_input = ('Patients>Controls', 'T', ['Group_{1}', 'Group_{2}'], [-1, 1])

    # Infosource - a function free node to iterate over contrasts
    info_source = Node(IdentityInterface(fields=['contrast_id']),
                       name="info_source")
    info_source.iterables = [('contrast_id', second_level_contrast_list)]

    # Select files with custom function
    select_files = Node(Function(input_names=['contrast_id',
                                              'subj_list',
                                              'post_fix_dir',
                                              'analysis_label'],
                        output_names=['control_cons', 'patient_cons'],
                        function=secondlevel_files_2samp),
               name='select_files')
    select_files.inputs.subj_list = subj_list_secondlvl
    select_files.inputs.post_fix_dir = post_fix_dir
    select_files.inputs.analysis_label = analysis_label

    # OneSampleTTestDesign - creates one sample T-Test Design
    twosamplettestdes = Node(
        spm.TwoSampleTTestDesign(), name="twosampttestdes")
    twosamplettestdes.inputs.explicit_mask_file = group_mask
    twosamplettestdes.inputs.unequal_variance = True

    # EstimateModel - estimates the model
    level2estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                          name="level2estimate")

    # EstimateContrast - estimates group contrast
    level2con = Node(spm.EstimateContrast(group_contrast=True),
                     name="level2con")
    level2con.inputs.contrasts = [con_input]

    level2thresh = Node(spm.Threshold(contrast_index=1,
                                      extent_fdr_p_threshold=cluster,
                                      extent_threshold=extent,
                                      height_threshold=height,
                                      height_threshold_type='p-value',
                                      use_fwe_correction=False,
                                      use_topo_fdr=True),
                        name="level2thresh")

    # Save output (datasink)
    sinker2level = Node(DataSink(base_directory=post_fix_dir+'spm_group/'),
                        name='sinker2level')
    substitutions = [('_contrast_id_', '')]  # Define substitution strings
    sinker2level.inputs.substitutions = substitutions

    # create workflow and connections
    # Initiation of the 2nd-level analysis workflow
    wf = Workflow(name='postPrep2ndlevel', base_dir=work_dir)

    # Connect up the 2nd-level analysis components
    wf.connect([(info_source, select_files, [('contrast_id', 'contrast_id')]),
                (select_files, twosamplettestdes, [
                 ('control_cons', 'group1_files')]),
                (select_files, twosamplettestdes, [
                 ('patient_cons', 'group2_files')]),
                (twosamplettestdes, level2estimate, [
                 ('spm_mat_file', 'spm_mat_file')]),
                (level2estimate, level2con, [('spm_mat_file', 'spm_mat_file'),
                                             ('beta_images', 'beta_images'),
                                             ('residual_image', 'residual_image')]),
                (level2con, level2thresh, [('spm_mat_file', 'spm_mat_file'),
                                           ('spmT_images', 'stat_image')]),
                (level2con, sinker2level, [('spm_mat_file', 'glm_'+analysis_label+'.2samp_'+con+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@spm_mat'),
                                           ('spmT_images', 'glm_'+analysis_label+'.2samp_'+con+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@T'),
                                           ('con_images', 'glm_'+analysis_label+'.2samp_'+con+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@con')]),
                (level2thresh, sinker2level, [
                 ('thresholded_map', 'glm_'+analysis_label+'.2samp_'+con+'_h-'+str(height)[2::]+'_c-'+str(cluster)[2::]+'.@threshold')])
                ])
    return wf


def run_firstlevel():
    # python -c "from glm_nipype import *; run_firstlevel()"
    wf = first_level_wf()
    wf.run('MultiProc', plugin_args={'n_procs': n_procs})
    move_firstlevel()


def run_firstlevel_nosmooth():
    # python -c "from glm_nipype import *; run_firstlevel_nosmooth()"
    wf = first_level_nosmooth_wf()
    wf.run('MultiProc', plugin_args={'n_procs': n_procs})
    move_firstlevel()


def run_secondlevel():
    # python -c "from glm_nipype import *; run_secondlevel()"
    wf = secondlevel_1samp_wf()
    wf.run('MultiProc', plugin_args={'n_procs': n_procs})

    wf = secondlevel_2samp_wf(con='HC>PAT')
    wf.run('MultiProc', plugin_args={'n_procs': n_procs})

    wf = secondlevel_2samp_wf(con='PAT>HC')
    wf.run('MultiProc', plugin_args={'n_procs': n_procs})


def first_level_nosmooth_wf():
    '''
    [summary]

    Returns
    -------
    [type]
        [description]
    '''
    # iterable node
    info_source = Node(IdentityInterface(fields=['subject_id', 'contrasts'],
                                         contrasts=contrast_list),
                       name='info_source')
    info_source.iterables = [('subject_id', subj_list)]

    # Select functional data node
    # String template with {}-based strings
    templates = {'func': ('{subject_id}/func/{subject_id}_task-'+task+'_'
                          + 'space-'+space+'_desc-detrend.nii.gz')}

    # Create SelectFiles node
    select_files = Node(SelectFiles(templates, base_directory=post_fix_dir),
                        name='select_files')

    # unzip .nii.gz data for SPM
    gunzip = Node(Gunzip(), name='gunzip')

    # # smooth the data
    # smooth = Node(spm.Smooth(fwhm=fwhm), name='smooth')

    # # save smoothed data via datasink
    # sinker = Node(DataSink(base_directory=work_dir+'smooth'), name='sinker')
    # substitutions = [('_subject_id_', '')]  # Define substitution strings
    # sinker.inputs.substitutions = substitutions

    # First Level
    # get subject TR
    getsubjectTR = Node(Function(input_names=['func'],
                                 output_names=['TR'],
                                 function=get_nipype_TR),
                        name='getsubjectTR')

    # Select event files
    event_template = {'event': ('{subject_id}/func/{subject_id}_task-'+task
                                + '_events.tsv')}

    # Select confound files
    conf_template = {'conf': ('{subject_id}/func/{subject_id}_task-'+task
                              + '_desc-confounds_timeseries.tsv')}

    # Create SelectFiles node (event files)
    select_event_files = Node(SelectFiles(event_template, base_directory=bids_dir),
                              name='select_event_files')

    # Create SelectFiles node (confounds file)
    select_conf_files = Node(SelectFiles(conf_template, base_directory=prep_dir),
                             name='select_conf_files')

    # Get Subject Info - get subject specific condition information
    getsubjectinfo = Node(Function(input_names=['event_file', 'conf_file', 'design'],
                                   output_names=['subject_info'],
                                   function=get_nipype_subject_info),
                          name='getsubjectinfo')
    getsubjectinfo.inputs.design = firstlevel_design

    # SpecifyModel - Generates SPM-specific Model
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     high_pass_filter_cutoff=128),
                     name="modelspec")

    # Level1Design - Generates an SPM design matrix
    level1design = Node(spm.Level1Design(bases={'hrf': {'derivs': [1, 0]}},
                                         timing_units='secs',
                                         model_serial_correlations='FAST'),
                        name="level1design")
    level1design.inputs.mask_image = group_mask
    level1design.inputs.mask_threshold = 0.5

    # EstimateModel - estimate the parameters of the model
    level1estimate = Node(spm.EstimateModel(estimation_method={'Classical': 1}),
                          name="level1estimate")

    # EstimateContrast - estimates contrasts
    level1con = Node(spm.EstimateContrast(), name="level1con")

    # save first level files
    sinker1level = Node(DataSink(base_directory=work_dir+'1stlvl/'),
                        name='sinker1level')
    substitutions = [('_subject_id_', '')]
    sinker1level.inputs.substitutions = substitutions

    # workflow construction for first level analysis
    # create workflow
    wf = Workflow(name='postPrep', base_dir=work_dir)

    # link file selector and gunzipping
    wf.connect(info_source, 'subject_id', select_files, 'subject_id')
    wf.connect(select_files, 'func', gunzip, 'in_file')
    wf.connect(select_files, 'func', getsubjectTR, 'func')

    # do smoothing
    #wf.connect(gunzip, 'out_file', smooth, 'in_files')
    #wf.connect(smooth, 'smoothed_files', sinker, 'smooth-'+smooth_label)

    # link subject ids and get subject information
    wf.connect(info_source, 'subject_id', select_event_files, 'subject_id')
    wf.connect(info_source, 'subject_id', select_conf_files, 'subject_id')
    wf.connect(select_event_files, 'event', getsubjectinfo, 'event_file')
    wf.connect(select_conf_files, 'conf', getsubjectinfo, 'conf_file')

    # link subject info and data to SPM model
    wf.connect(getsubjectTR, 'TR', modelspec, 'time_repetition')
    wf.connect(getsubjectinfo, 'subject_info', modelspec, 'subject_info')
    wf.connect(gunzip, 'out_file', modelspec, 'functional_runs')
    wf.connect(getsubjectTR, 'TR', level1design, 'interscan_interval')
    wf.connect(modelspec, 'session_info', level1design, 'session_info')
    wf.connect(info_source, 'contrasts', level1con, 'contrasts')

    # link design to estimate to contrasts
    wf.connect(level1design, 'spm_mat_file', level1estimate, 'spm_mat_file')
    wf.connect(level1estimate, 'spm_mat_file', level1con, 'spm_mat_file')
    wf.connect(level1estimate, 'beta_images', level1con, 'beta_images')
    wf.connect(level1estimate, 'residual_image', level1con, 'residual_image')

    # save the images out
    wf.connect(level1con, 'spm_mat_file', sinker1level, '@spm_mat')
    wf.connect(level1con, 'spmT_images', sinker1level, '@T')
    wf.connect(level1con, 'con_images', sinker1level, '@con')
    wf.connect(level1con, 'ess_images', sinker1level, '@ess')
    return wf
