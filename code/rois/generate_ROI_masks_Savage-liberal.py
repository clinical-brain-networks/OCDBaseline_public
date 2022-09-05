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
from nilearn.image import index_img, threshold_img, resample_to_img
from scipy.ndimage import center_of_mass
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import stats
import sys
sys.path.insert(0, '..')
from functions.data_helpers import get_computer

def get_centroid(roi_data, template_img):
    coords = center_of_mass(roi_data)
    coords_MNI = coord_transform(
        coords[0], coords[1], coords[2], nib.load(template_img).affine)
    return np.round(coords_MNI)


def t_to_z(t_scores, deg_of_freedom):
    p_values = stats.t.sf(t_scores, df=deg_of_freedom)
    z_values = stats.norm.isf(p_values)
    return z_values


# global variables
_, proj_dir = get_computer()
roi_dir = proj_dir+'data/derivatives/masks/'
template_img = (proj_dir+'data/derivatives/post-fmriprep-fix/'
                + 'spm_group/glm_smooth-6mm_despike/1samp/'
                + 'con_0001/spmT_0001.nii')
bg_img = (proj_dir + 'data/derivatives/masks/'
          + 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz')

# deg of freedom for the t-to-z conversion
deg_of_freedom = 92  # From Savage et al.,

# threshold for z score
z_thresh = 2.41

# voxel size threshold
min_region_size = (200*(2**2**2))  # size (100) * voxel sizes

# you need to look at each roi to label them (hence a supervised script!)
roi_labels_dict = {'Savage_threat': {'rois': [1, 2, 3],
                                     'labels': ['leftInsula',
                                                'acc',
                                                'rightInsula']},
                   'Savage_safety': {'rois': [3, 4],
                                     'labels': ['vmpfc',
                                                'pcc']}}

# pre allocate
df_coords = pd.DataFrame(columns=['spm', 'label', 'x', 'y', 'z'])

# THREAT / Threat reversal learning contrast - con_0006 in SPM
for spm in ['Savage_threat', 'Savage_safety']:
    spm_file = (proj_dir+'data/derivatives/masks/'+spm+'.nii')
    spm_img = nib.load(spm_file)

    # resample the spm image
    spm_img_resamp = resample_to_img(spm_img, template_img,
                                     interpolation='nearest')

    # plot spm
    plotting.plot_stat_map(spm_img_resamp, draw_cross=False, title=spm, bg_img=bg_img)

    # threshold
    spm_img_thresh, threshold = threshold_stats_img(stat_img=spm_img_resamp,
                                                    threshold=z_thresh,
                                                    height_control=None,
                                                    cluster_threshold=300,
                                                    two_sided=False)

    plotting.plot_stat_map(spm_img_thresh, draw_cross=False, title=spm, bg_img=bg_img)

    # Seperate the results
    regions_value_img, index = connected_regions(spm_img_thresh,
                                                 min_region_size=10,
                                                 extract_type='connected_components')

    plotting.plot_prob_atlas(regions_value_img, view_type='contours')
    print(regions_value_img.shape[3], 'ROIs isolated')

    # Choose the ROIs you would like to include (all of them)
    roi_list = range(regions_value_img.shape[3])
    roi_labels = roi_labels_dict[spm]['labels']

    for i, roi_i in enumerate(roi_labels_dict[spm]['rois']):

        # isolate and binarize
        roi_data = index_img(regions_value_img, roi_i).get_fdata() > 0
 
        # convert scipy back into nifti
        roi_img = new_img_like(template_img, roi_data)

        # visualize roi
        plotting.plot_roi(roi_img, title=roi_labels[i])
        plotting.show()

        # save roi
        nib.save(roi_img, roi_dir+spm+'_'+roi_labels[i]+'-liberal.nii.gz')

        # save the coordinates
        coords_MNI = get_centroid(roi_data, template_img)
        row = {'spm': spm,
               'label': roi_labels[i],
               'x': coords_MNI[0],
               'y': coords_MNI[1],
               'z': coords_MNI[2]}
        df_coords = df_coords.append(row, ignore_index=True)
df_coords.to_csv(roi_dir+'Savage_coords-liberal.csv', index=False)

# %%
