# %%
'''
Code to take individual subject parameter estimates from SPM in
regions of interest.

We use the mean within rois / spheres.

Also an 'eroded' version where ROI size has been controlled for

'''
import sys
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.masking import apply_mask
from nilearn.input_data import NiftiSpheresMasker
from nilearn.image import coord_transform
import warnings
sys.path.insert(0, '..')
from functions.data_helpers import get_computer, get_subj_group

# turn off a warning
warnings.simplefilter(action='ignore', category=UserWarning)

# global variables
# paths
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
roi_dir = proj_dir+'data/derivatives/masks/'

# glm analysis stream to use
glm_label = 'smooth-6mm_despike'

# contrast information
# list of SPM contrasts that match the four conditions of interest
# see 'glms' folder and the nipype code
# these correspond to:
# conditioning_CS+, conditioning_CS-, reversal_CS+, reversal_CS-
# (after subtracting the habituation condition)
con_list = ['con_0001', 'con_0002', 'con_0003', 'con_0004']


def save_roi_signals(values, method, roi_labels, subj, results_df):
    # a short helper function to save the data to a df
    c = 0
    for phase in ['conditioning', 'reversal']:
        for trial in ['CS+', 'CS-']:
            df = pd.DataFrame()
            df['value'] = values[c, :]
            df['method'] = method
            df['roi'] = roi_labels
            df['phase'] = phase
            df['trial_type'] = trial
            df['subj'] = subj
            df['group'] = get_subj_group(subj)
            results_df = pd.concat([results_df, df])
            c += 1

    # save threat and safety contrast values specifically
    df = pd.DataFrame()

    # rev CS- vs. cond CS+ (safety)
    df['value'] = values[3, :] - values[0, :]
    df['method'] = method
    df['roi'] = roi_labels
    df['contrast'] = 'Safety reversal'
    df['subj'] = subj
    df['group'] = get_subj_group(subj)
    results_df = pd.concat([results_df, df])

    df = pd.DataFrame()
    # rev CS+ vs. cond CS- (threat)
    df['value'] = values[2, :] - values[1, :]
    df['method'] = method
    df['roi'] = roi_labels
    df['contrast'] = 'Threat reversal'
    df['subj'] = subj
    df['group'] = get_subj_group(subj)
    results_df = pd.concat([results_df, df])
    return results_df


def extract_betas(subj, mask_input='Savage', radius=4, eroded=False):

    # load analysis information from the .csv
    df_spm = pd.read_csv(roi_dir+mask_input+'_coords.csv')

    if mask_input == 'tian':  # no need for safety contrast
        df_spm = df_spm.loc[df_spm.spm == 'tian_threat']

    # roi files can be derived from spm csv information:
    roi_labels = df_spm.label.values
    roi_files = []
    for i, row in df_spm.iterrows():
        if mask_input == 'spm':
            roi_files.append(roi_dir+'spm_'+row.spm+'_'+row.label+'.nii.gz')
        else:
            roi_files.append(roi_dir+row.spm+'_'+row.label+'.nii.gz')

    if eroded:  # choose different rois
        new_roi_files = []
        for roi in roi_files:
            filename = roi.split('.nii.gz')[0]+'_eroded.nii.gz'
            new_roi_files.append(filename)
        roi_files = new_roi_files

    # seed coordinates
    seeds = df_spm[['x', 'y', 'z']].to_numpy()
    results_df = pd.DataFrame()
    coords_MNI_df = pd.DataFrame()

    # get first level contrast niftis
    con_files = []
    for con in con_list:
        con_files.append(deriv_dir+subj+'/spm/glm_'+glm_label+'/'+con+'.nii')

    # extract roi signals via the MEAN across all voxels
    # roi signals is shape (condition x roi)
    roi_signals = np.zeros((4, len(roi_files)))
    for i, roi in enumerate(roi_files):
        roi_signals[:, i] = np.mean(apply_mask(con_files, roi), axis=1)

    # save
    results_df = save_roi_signals(roi_signals, 'region',
                                  roi_labels, subj, results_df)
    results_df.drop_duplicates(
        subset=None, keep='first', inplace=True, ignore_index=False)

    # save the results
    if eroded:
        results_out = (deriv_dir+'spm_group/glm_'+glm_label
                       + '/extracted_betas/'+subj+'_'+mask_input+'_'+str(radius)+'mm_eroded.csv')
    else:
        results_out = (deriv_dir+'spm_group/glm_'+glm_label
                       + '/extracted_betas/'+subj+'_'+mask_input+'_'+str(radius)+'mm.csv')
    results_df.to_csv(results_out, index=False)


if __name__ == '__main__':
    # define subj
    subj = sys.argv[1]
    print(subj)

    extract_betas(subj, mask_input='Savage', radius=4, eroded=False)
    extract_betas(subj, mask_input='tian', radius=4, eroded=False)

    extract_betas(subj, mask_input='Savage', radius=4, eroded=True)
    extract_betas(subj, mask_input='tian', radius=4, eroded=True)

    print('Subj finished')
