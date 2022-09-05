# %%
'''
Extracts timeseries for roi masks, and spheres
Performs a single subject at time, designed to be used on the cluster
'''
import sys, os
import numpy as np
import pandas as pd
import h5py
import nibabel as nib
from nilearn.input_data import NiftiLabelsMasker, NiftiSpheresMasker
from time import time
sys.path.insert(0, '..')
from functions.data_helpers import get_computer

# global variables
# paths
_, proj_dir = get_computer()
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
scratch_dir = proj_dir+'data/scratch/nilearn/'
roi_dir = proj_dir+'data/derivatives/masks/'

# which data?
# task
task_list = ['fearRev', 'rest']

# preprocessed image space
img_space = 'MNI152NLin2009cAsym'

# denoising label (no smoothing needed)
denoise_label = {'fearRev': 'detrend',
                 'rest': 'detrend_filtered_scrub'}

# sphere radius
radius = 4
eroded = False
# glm analysis stream to use to extract coords
glm_label = 'smooth-6mm_despike'

# region of interest information
# edit these as appropriate
roi_files = [roi_dir+'Savage_threat_leftInsula.nii.gz',
             roi_dir+'Savage_threat_rightInsula.nii.gz',
             roi_dir+'Savage_threat_acc.nii.gz',
             roi_dir+'Savage_safety_pcc.nii.gz',
             roi_dir+'Savage_safety_vmpfc.nii.gz',
             roi_dir+'tian_threat_leftPUT.nii.gz',
             roi_dir+'tian_threat_rightPUT.nii.gz',
             roi_dir+'tian_threat_leftCAU.nii.gz',
             roi_dir+'tian_threat_rightCAU.nii.gz',
             roi_dir+'tian_threat_leftGP.nii.gz',
             roi_dir+'tian_threat_rightGP.nii.gz']

if eroded:  # choose different rois
    new_roi_files = []
    for roi in roi_files:
        filename = roi.split('.nii.gz')[0]+'_eroded.nii.gz'
        new_roi_files.append(filename)
    roi_files = new_roi_files


roi_labels = ['leftInsula',
              'rightInsula',
              'acc',
              'pcc',
              'vmpfc',
              'leftPUT',
              'rightPUT',
              'leftCAU',
              'rightCAU',
              'leftGP',
              'rightGP']

# group seed coordinates
df_spm = pd.concat([pd.read_csv(roi_dir+'Savage_coords.csv'),
                    pd.read_csv(roi_dir+'tian_coords.csv')])
df_spm = df_spm[df_spm.spm != 'tian_safety']  
df_spm.reset_index(inplace=True, drop=True)


def extract_ts(subj, method='region', label='region', df_seed=None, eroded=eroded):
    start = time()

    for task in task_list:
        print('\t', task)

        # get input image
        bold_file = deriv_dir+subj+'/func/'+subj+'_task-' + \
            task+'_space-'+img_space+'_desc-'+denoise_label[task]+'.nii.gz'
        bold_img = nib.load(bold_file)

        # init h5 file
        try:
            os.makedirs(deriv_dir+subj+'/timeseries/')
            print("Directory ", deriv_dir+subj+'/timeseries/', " created ")
        except FileExistsError:
            print("Directory ", deriv_dir+subj+'/timeseries/', " already exists")
            pass

        if eroded:
            out_file = deriv_dir+subj+'/timeseries/'+subj+'_task-' + \
                task+'_method-'+label+'_desc-'+denoise_label[task]+'_eroded.h5'
        else:
            out_file = deriv_dir+subj+'/timeseries/'+subj+'_task-' + \
                task+'_method-'+label+'_desc-'+denoise_label[task]+'.h5'
        hf = h5py.File(out_file, 'w')

        # extract roi signals
        if method == 'region':
            for i, roi in enumerate(roi_files):
                masker = NiftiLabelsMasker(labels_img=roi)
                ts = np.squeeze(masker.fit_transform(bold_img))
                hf.create_dataset(roi_labels[i], data=ts)  # save

        elif method == 'sphere':
            # get seed coords
            seeds = df_seed[['x', 'y', 'z']].to_numpy()

            # extract sphere signals
            masker = NiftiSpheresMasker(seeds, radius=radius, allow_overlap=True)
            ts = masker.fit_transform(bold_img)

            # save each ts seperately
            for i in range(len(df_seed)):
                hf.create_dataset(df_seed.label[i], data=ts[:, i])

            # save the df for reference as well
            df_spm.to_csv(deriv_dir+subj+'/timeseries/'+subj+'_task-' +
                          task+'_method-'+label+'_coords.csv', index=False)
        # close h5
        hf.close()

    finish = time()
    print('\tTime elapsed (s):', round(finish-start))


if __name__ == '__main__':
    
    # define subj
    subj = sys.argv[1]
    print(subj)

    # extract nii masks
    extract_ts(subj, method='region', label='region', eroded=True)
    extract_ts(subj, method='region', label='region', eroded=False)

    print('Finished ts extraction')
