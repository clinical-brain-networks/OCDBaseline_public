'''
Collection of functions for loading and organising data.
'''
import platform
import os
import pandas as pd
import numpy as np
import nibabel as nib
import pingouin as pg

def get_computer():
    '''
    Detects which computer the code is being run on using platform.
    Useful for changing paths on the hpc vs. lucky2 automatically
    Returns
    -------
    str
        'hpc' or 'lucky' identifier
    str
        project directory, e.g., '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'
    '''
    if platform.release() == '3.10.0-693.11.6.el7.x86_64':
        computer = 'hpc'
        project_directory = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/'

    elif platform.release() == '4.15.0-43-generic':
        computer = 'lucky2'
        project_directory = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'
    
    elif platform.release() == '5.8.0-61-generic':
        computer = 'lucky3'
        project_directory = '/home/lukeh/hpcworking/lukeH/projects/OCDbaseline/'

    else:
        print('Unknown computer! Assuming on the cluster...')
        print(platform.release())
        computer = 'Unknown'
        project_directory = '/mnt/lustre/working/lab_lucac/lukeH/projects/OCDbaseline/'
    return computer, project_directory


# global variables (after get computer)
# paths
computer, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
prep_dir = proj_dir+'data/derivatives/fmriprep/'
bold_dir = proj_dir+'data/derivatives/fmriprep-fix/'


def get_phenotype(subj_list, label_list=['participants']):
    '''
    returns a pandas df of demographic / phenotypic data

    Parameters
    ----------
    subj_list : list
        list of subjects (str)
    label_list : list
        list of phenotypic variables to return, e.g., 'ybocs'

    Returns
    -------
    pandas dataframe
        df of requested phenotypic data
    '''
    df = pd.DataFrame()
    for i, label in enumerate(label_list):

        # special case for basic demographics or participants file
        # these are kept in the upper directory
        if label == 'participants':
            ss_file = bids_dir+label+'.tsv'
        else:
            ss_file = bids_dir+'phenotype/'+label+'.tsv'

        tmp = pd.read_csv(ss_file, delimiter='\t')

        # apply any additional manipulation to phenotype data (e.g., calculate subscales)
        # these should really be included in the original raw -> bids code...
        if label == 'ocir':
            # calculate the subscales
            # factor 1: washing
            tmp['ocir_washing'] = tmp.ocir5 + tmp.ocir11 + tmp.ocir17
            # factor 2: obsessing
            tmp['ocir_obsess'] = tmp.ocir6 + tmp.ocir12 + tmp.ocir18
            # factor 3: hoarding
            tmp['ocir_hoarding'] = tmp.ocir1 + tmp.ocir7 + tmp.ocir13
            # factor 4: ordering
            tmp['ocir_ordering'] = tmp.ocir3 + tmp.ocir9 + tmp.ocir15
            # factor 5: checking
            tmp['ocir_checking'] = tmp.ocir2 + tmp.ocir8 + tmp.ocir14
            # factor 6: neutralizing
            tmp['ocir_neutral'] = tmp.ocir4 + tmp.ocir10 + tmp.ocir16

        # create or merge the dataframe
        if i == 0:
            df = tmp.copy()
        else:
            df = pd.merge(df, tmp, on='participant_id')

    # trim df to subject list
    df = df[df['participant_id'].isin(subj_list)]
    return df


def get_mean_fd(subj_list, task_list):

    confound_df = pd.DataFrame()

    for subj in subj_list:
        for task in task_list:

            # get confounds
            conf_loc = prep_dir+subj+'/func/'+subj+'_task-'+task+'_desc-confounds_timeseries.tsv'
            df = pd.read_csv(conf_loc, delimiter='\t')
            # subject info
            subj_df = pd.DataFrame()
            subj_df['participant_id'] = [subj]
            subj_df['task'] = [task]
            subj_df['framewise_displacement_avg'] = np.nanmean(df['framewise_displacement'])
            # concat
            confound_df = pd.concat([confound_df, subj_df])
    return confound_df


def get_subj_group(subj):
    '''
    [summary]

    Parameters
    ----------
    subj : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    '''
    df = get_phenotype([subj], label_list=['participants'])
    return df.group.values[0]


def get_task_beh_data(subj_list, ratings_only=False):
    '''
    [summary]

    Parameters
    ----------
    ratings_only : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    '''

    # get task data
    task_df = pd.DataFrame()
    for subj in subj_list:
        # event file
        event_file = bids_dir+subj+'/func/'+subj+'_task-fearRev_events.tsv'

        # load dataframe
        tmp = pd.read_csv(event_file, delimiter='\t')

        # add subj id
        tmp['participant_id'] = subj

        # get the group
        if 'patient' in subj:
            tmp['group'] = 'patient'
        else:
            tmp['group'] = 'control'

        # remove all non rating info
        task_df = pd.concat([task_df, tmp])
    if ratings_only:
        task_df = task_df.dropna(subset=['rating_measure'])
        task_df = task_df.dropna(axis=1, how='all')

    return task_df


def make_dirs(dirs_list):
    '''
    Creates dir if it doesn't exist

    Parameters
    ----------
    dirs_list : [type]
        [description]
    '''

    for dirName in dirs_list:
        try:
            os.makedirs(dirName)
            print("Directory ", dirName, " created ")
        except FileExistsError:
            # print("Directory ", dirName, " already exists")
            pass


def get_bold_data(subj, task, bold_dir=bold_dir, img_space='MNI152NLin2009cAsym', parc=None):
    '''
    Fetch (preprocessed) bold data for a given task

    Parameters
    ----------
    subj : str
        bids-style subject label, e.g., 'sub-control01'
    task : str
        'fearRev' or 'rest'
    img_space : str
        space/image type, by default 'MNI152NLin2009cAsym'
    parc : str, optional
        fetch already parcellated data, by default None

    Returns
    -------
    nibabel image
        The complete nibabel image with header etc.,
    '''
    if img_space == 'MNI152NLin2009cAsym' and parc is None:
        img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                    + img_space+'_desc-preproc_bold.nii.gz')
        msk_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                    + img_space+'_desc-brain_mask.nii.gz')

    if img_space == 'fsLR_den-91k' and parc is None:
        img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                            + img_space+'_bold.dtseries.nii')
        msk_file = np.nan

    elif parc == 'glasser-tian':
        img_file = (bold_dir+subj+'/func/'+subj+'_task-'+task+'_space-'
                    + img_space+'_parc-'+parc+'_bold.ptseries.nii')
        msk_file = np.nan

    img = nib.load(img_file)

    try:
        msk = nib.load(msk_file)
    except Exception:
        msk = np.nan

    return img, msk


def sym_by_avg(data):
    data_all = np.dstack((data, data.T))
    data_avg = np.mean(data_all, axis=2)

    mask = np.eye(data.shape[0], dtype=bool)
    data_avg[mask] = 0
    return data_avg


def get_awareness_labels(subj_list):
    df = get_phenotype(subj_list, ['participants', 'post_scan'])

    # calculate manipulation accuracies
    df['conditioning_manipulation_acc'] = df['conditioning_CS+'] == df['conditioning_manipulation']
    df['reversal_manipulation_acc'] = df['reversal_CS+'] == df['reversal_manipulation']

    # all controls scored accurately
    df.loc[df.group == 'control', 'conditioning_manipulation_acc'] = True
    df.loc[df.group == 'control', 'reversal_manipulation_acc'] = True

    df['aware'] = df[['reversal_manipulation_acc', 'conditioning_manipulation_acc']].all(axis='columns')
    df['group_aware'] = np.nan

    for group in ['control', 'patient']:
        df.loc[(df.group == group) & (df.aware == True), 'group_aware'] = group+'-aware'
        df.loc[(df.group == group) & (df.aware == False), 'group_aware'] = group+'-unaware'
    return df['group_aware']


def get_reversal_behaviour(subj_list):
    task_df = get_task_beh_data(subj_list, ratings_only=True)
    df = get_phenotype(subj_list, ['participants'])

    for label in ['arousal', 'valence']:
        tmp = task_df[task_df.rating_measure == label]

        # safety reversal
        srev = ((tmp.loc[(tmp.phase == 'reversal')
                         & (tmp.trial_type == 'CS-')].rating.values)
                - (tmp.loc[(tmp.phase == 'conditioning')
                           & (tmp.trial_type == 'CS+')].rating.values))

        # threat reversal
        trev = ((tmp.loc[(tmp.phase == 'reversal')
                         & (tmp.trial_type == 'CS+')].rating.values)
                - (tmp.loc[(tmp.phase == 'conditioning')
                           & (tmp.trial_type == 'CS-')].rating.values))
        df[label+'_safety_reversal'] = srev.copy()
        df[label+'_threat_reversal'] = trev.copy()

    return df


def run_behav_stats(df, groupvar='group', group1='control', group2='patient'):
    stat_df = pd.DataFrame()
    for measure in ['arousal', 'valence']:
        for contrast in ['safety_reversal', 'threat_reversal']:

            # one sample t-test
            tmp = pg.ttest(df[measure+'_'+contrast], y=0)
            tmp['measure'] = measure
            tmp['contrast'] = contrast
            tmp['test'] = 'One sample'
            stat_df = pd.concat([stat_df, tmp])

            # two sample ttest
            tmp = pg.ttest(df.loc[df[groupvar] == group1, measure+'_'+contrast],
                        df.loc[df[groupvar] == group2, measure+'_'+contrast])

            tmp['measure'] = measure
            tmp['contrast'] = contrast
            tmp['test'] = 'Two sample'
            stat_df = pd.concat([stat_df, tmp])

    # multiple comparison correction
    stat_df['pval_corr'] = np.nan
    stat_df['sig_corr'] = np.nan
    for label in ['arousal', 'valence']:
        idx = stat_df.measure == label
        tmp = stat_df[idx]
        h1, pval_corr = pg.multicomp(tmp['p-val'].values, alpha=0.05, method='fdr_by')
        stat_df.loc[idx, 'pval_corr'] = np.real(pval_corr)
        stat_df.loc[idx, 'sig_corr'] = h1
            
    return stat_df