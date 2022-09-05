# %%
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import sys
from nilearn.image import new_img_like, coord_transform
from scipy.ndimage import binary_erosion, center_of_mass
from nilearn.image import new_img_like
from nilearn import plotting
sys.path.insert(0, '..')
from functions.data_helpers import get_computer


def get_centroid(roi_data, template_img):
    coords = center_of_mass(roi_data)
    coords_MNI = coord_transform(
        coords[0], coords[1], coords[2], nib.load(template_img).affine)
    return np.round(coords_MNI)


# global variables
_, proj_dir = get_computer()
roi_dir = proj_dir+'data/derivatives/masks/'

roi_list = ['Savage_safety_pcc.nii.gz',
            'Savage_safety_vmpfc.nii.gz',
            'Savage_threat_acc.nii.gz',
            'Savage_threat_leftInsula.nii.gz',
            'Savage_threat_rightInsula.nii.gz',
            'tian_threat_leftCAU.nii.gz',
            'tian_threat_leftGP.nii.gz',
            'tian_threat_leftPUT.nii.gz',
            'tian_threat_rightCAU.nii.gz',
            'tian_threat_rightGP.nii.gz',
            'tian_threat_rightPUT.nii.gz',
            'tian_threat_rightHipp.nii.gz',
            'tian_threat_rightAmyg.nii.gz',
            'tian_threat_rightpTHA.nii.gz',
            'tian_threat_rightaTHA.nii.gz',
            'tian_threat_leftHipp.nii.gz',
            'tian_threat_leftAmyg.nii.gz',
            'tian_threat_leftpTHA.nii.gz',
            'tian_threat_leftaTHA.nii.gz'
            ]

roi_sizes = []
eroded_roi_sizes = []
for roi in roi_list:
    roi_data = nib.load(roi_dir+roi).get_fdata()
    roi_size = np.sum(roi_data)
    print(roi, ':', roi_size) 
    roi_sizes.append(roi_size)

    eroded_roi_data = roi_data.copy()
    eroded_roi_size = roi_size.copy()
    while eroded_roi_size > 350:  # 350 seems to provide a good limit
        eroded_roi_data = binary_erosion(eroded_roi_data, iterations=1)
        eroded_roi_size = np.sum(eroded_roi_data)
        print(roi, ':', eroded_roi_size)
    eroded_roi_sizes.append(eroded_roi_size)
 
    # visualize and save eroded mask
    filename = roi.split('.nii.gz')[0]+'_eroded.nii.gz'
    eroded_roi_img = new_img_like(roi_dir+roi, eroded_roi_data)
    # plotting.plot_roi(eroded_roi_img)
    # plotting.show()

    nib.save(eroded_roi_img, roi_dir+filename)

plt.bar(range(len(roi_list)), roi_sizes)
plt.bar(range(len(roi_list)), eroded_roi_sizes)
plt.show()


# %%
