# %%
'''
Extracts regions based on Tian SC parc. Saves ROI masks and a list of centroids.

Looks at con_0006 ('threat reversal') and con_0007 ('safety reversal') specifically.
'''
from nilearn import plotting
from nilearn.image import new_img_like
from scipy.ndimage import center_of_mass
import nibabel as nib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '..')
from functions.data_helpers import get_computer, get_subj_group

# global variables
_, proj_dir = get_computer()
roi_dir = proj_dir+'data/derivatives/masks/'

# tian's parcellation
sc_img = ('/home/lukeh/hpcworking/shared/parcellations/Tian2020MSA_v1.1/'
          + '3T/Subcortex-Only/Tian_Subcortex_S1_3T_2009cAsym.nii.gz')
sc_coords = ('/home/lukeh/hpcworking/shared/parcellations/Tian2020MSA_v1.1/'
             + '3T/Subcortex-Only/Tian_Subcortex_S1_3T_COG.txt')
coords = np.loadtxt(sc_coords)

# list of the ROIs - remember these are indexed from 1 onwards in the nifti
# file, i.e., not pythonic!
'''
1. HIP-rh
2. AMY-rh
3. pTHA-rh
4. aTHA-rh
5. NAc-rh
6. GP-rh
7. PUT-rh
8. CAU-rh
9. HIP-lh
10. AMY-lh
11. pTHA-lh
12. aTHA-lh
13. NAc-lh
14. GP-lh
15. PUT-lh
16. CAU-lh
'''
rois = [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16]
roi_labels = ['rightHipp',
              'rightAmyg',
              'rightpTHA',
              'rightaTHA',
              'rightGP',
              'rightPUT',
              'rightCAU',
              'leftHipp',
              'leftAmyg',
              'leftpTHA',
              'leftaTHA',
              'leftGP',
              'leftPUT',
              'leftCAU']

# pre allocate
df_coords = pd.DataFrame(columns=['spm', 'label', 'x', 'y', 'z'])

img = nib.load(sc_img)

# plot atlas
plotting.plot_stat_map(img, draw_cross=False)
for con in ['threat', 'safety']:
    for i, roi_i in enumerate(rois):

        # isolate and binarize
        roi_data = img.get_fdata() == roi_i
        print(roi_labels[i], i, roi_i, np.sum(roi_data))

    #     # convert scipy back into nifti
    #     roi_img = new_img_like(sc_img, roi_data)

    #     # visualize roi
    #     plotting.plot_roi(roi_img, title=roi_labels[i])
    #     plotting.show()

    #     # save roi
    #     nib.save(roi_img, roi_dir+'tian_'+con+'_'+roi_labels[i]+'.nii.gz')

    #     # save the coordinates
    #     row = {'spm': 'tian_'+con,
    #            'label': roi_labels[i],
    #            'x': coords[roi_i-1, 0],
    #            'y': coords[roi_i-1, 1],
    #            'z': coords[roi_i-1, 2]}
    #     df_coords = df_coords.append(row, ignore_index=True)
    # df_coords.to_csv(roi_dir+'tian_coords.csv', index=False)

# %%
