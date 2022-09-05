# %%
'''
Extracts regions based on SPM results. Saves ROI masks and a list of centroids.

Looks at con_0006 ('threat reversal') and con_0007 ('safety reversal') specifically.

Be mindful that this script needs 'supervision', e.g., to hand label each ROI which 
may change depending on the exact parameters (namely cluster size)
'''
from nilearn.regions import connected_regions
from nilearn.glm import threshold_stats_img
from nilearn import plotting
from nilearn.image import new_img_like, resample_to_img, load_img, coord_transform
from nilearn.image import index_img, threshold_img
from scipy.ndimage import center_of_mass
from functions.data_helpers import get_computer
import nibabel as nib
import numpy as np
import pandas as pd


def get_centroid(roi_data, template_img):
    coords = center_of_mass(roi_data)
    coords_MNI = coord_transform(
        coords[0], coords[1], coords[2], nib.load(template_img).affine)
    return np.round(coords_MNI)


# global variables
_, proj_dir = get_computer()
roi_dir = proj_dir+'data/derivatives/masks/'
template_img = (
    proj_dir+'data/derivatives/spm/1stLevel-6mm/sub-control01/spmT_0001.nii')

# you need to look at each roi to label them (hence a supervised script!)
roi_labels_dict = {'con_0006': ['left tpj',
                                'left insula',
                                'acc',
                                'right striatum',
                                'right insula',
                                'right pmc',
                                'right tpj'],
                   'con_0007': ['vmpfc',
                                'pcc']}

# voxel size threshold
min_region_size = (100*(2**2**2))  # size (100) * voxel sizes

# pre allocate
df_coords = pd.DataFrame(columns=['spm', 'label', 'x', 'y', 'z'])

# THREAT / Threat reversal learning contrast - con_0006 in SPM
for spm in ['con_0006', 'con_0007']:
    spm_file = (proj_dir+'data/derivatives/spm/2ndLevel-6mm/1samp/'+spm
                + '/spmT_0001.nii')
    spm_img = nib.load(spm_file)
    spm_file = (proj_dir+'data/derivatives/spm/2ndLevel-6mm/1samp/'+spm
                + '/spmT_0001_thr.nii')
    spm_thresh_img = nib.load(spm_file)

    # plot spm
    plotting.plot_stat_map(spm_img, draw_cross=False, title=spm)
    plotting.plot_stat_map(spm_thresh_img, draw_cross=False, title=spm)

    # Seperate the results
    regions_value_img, index = connected_regions(spm_thresh_img,
                                                 min_region_size=min_region_size,
                                                 extract_type='connected_components')

    plotting.plot_prob_atlas(regions_value_img, view_type='contours')
    print(regions_value_img.shape[3], 'ROIs isolated')

    # Choose the ROIs you would like to include (all of them)
    roi_list = range(regions_value_img.shape[3])
    roi_labels = roi_labels_dict[spm]

    for roi_i in roi_list:

        # isolate and binarize
        roi_data = index_img(regions_value_img, roi_i).get_fdata() > 0

        # convert scipy back into nifti
        roi_img = new_img_like(template_img, roi_data)

        # visualize roi
        plotting.plot_roi(roi_img, title=roi_labels[roi_i])
        plotting.show()

        # save roi
        nib.save(roi_img, roi_dir+'spm_'+spm+'_'+roi_labels[roi_i]+'.nii.gz')

        # save the coordinates
        coords_MNI = get_centroid(roi_data, template_img)
        row = {'spm': spm,
               'label': roi_labels[roi_i],
               'x': coords_MNI[0],
               'y': coords_MNI[1],
               'z': coords_MNI[2]}
        df_coords = df_coords.append(row, ignore_index=True)
df_coords.to_csv(roi_dir+'spm_coords.csv', index=False)
