# %% Plot spm glm results

import numpy as np
from nilearn import plotting
import nibabel as nib
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '..')
from functions.data_helpers import get_computer


# Ensure MNI152NLin2009cAsym template is available
# can be downloaded from https://www.templateflow.org/browse/
# paths
_, proj_dir = get_computer()
analysis_label = 'smooth-6mm_despike'
secondlevel_dir = proj_dir+'data/derivatives/post-fmriprep-fix/spm_group/glm_'+analysis_label+'/1samp/'
between_dir = proj_dir+'data/derivatives/post-fmriprep-fix/spm_group/glm_'+analysis_label+'/2samp/'

bg_img = (proj_dir + 'data/derivatives/masks/'
          + 'tpl-MNI152NLin2009cAsym_res-01_desc-brain_T1w.nii.gz')
fig_dir = '../../figures/'

# global plotting params
cmap1 = plt.cm.inferno
cmap2 = plt.cm.winter
annotate = False

# plot raw images for inspections
filename = 'con_0006/spmT_0001'
img = nib.load(secondlevel_dir+filename+'.nii')
plotting.plot_glass_brain(
    img, threshold=0, display_mode='lyrz', colorbar=True, plot_abs=False)

filename = 'con_0007/spmT_0001'
img = nib.load(secondlevel_dir+filename+'.nii')
plotting.plot_glass_brain(
    img, threshold=0, display_mode='lyrz', colorbar=True, plot_abs=False)

# plot the thresholded images
vmax = 7

fig, axs = plt.subplot_mosaic("""
                              AAB
                              FCC
                              DEE
                              """, figsize=(7, 5))
plt.rc('axes', labelsize=12)

# get threat image
filename = 'con_0006/spmT_0001'
img = nib.load(secondlevel_dir+filename+'_thr.nii')

cut_coords = [8, 60]
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='x',
                       cut_coords=cut_coords,
                       axes=axs['A'],
                       cmap=cmap1,
                       colorbar=False,
                       threshold=3,
                       vmax=vmax)

cut_coords = [-40]
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='y',
                       cut_coords=cut_coords,
                       axes=axs['B'],
                       cmap=cmap1,
                       colorbar=True,
                       threshold=3,
                       vmax=vmax)

cut_coords = [8, 31]  # -51, -2, 68
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='z',
                       cut_coords=cut_coords,
                       colorbar=True,
                       cmap=cmap1,
                       threshold=3,
                       axes=axs['C'],
                       vmax=vmax)


cut_coords = [-37]
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='x',
                       cut_coords=cut_coords,
                       axes=axs['F'],
                       cmap=cmap1,
                       colorbar=False,
                       threshold=3,
                       vmax=vmax)

filename = 'con_0007/spmT_0001'
img = nib.load(secondlevel_dir+filename+'_thr.nii')
cut_coords = [-4]
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='x',
                       cut_coords=cut_coords,
                       colorbar=False,
                       cmap=cmap2,
                       threshold=3,
                       axes=axs['D'],
                       vmax=vmax)

cut_coords = [-7, 25]
plotting.plot_stat_map(img,
                       bg_img=bg_img,
                       black_bg=False,
                       annotate=annotate,
                       display_mode='z',
                       cut_coords=cut_coords,
                       axes=axs['E'],
                       cmap=cmap2,
                       colorbar=True,
                       threshold=3,
                       vmax=vmax)

# Text
# headings
font = {'color':  'black',
        'weight': 'bold',
        'size': 10}
fig.text(0.1, 0.9, "A", fontdict=font)

font = {'color':  'black',
        'weight': 'bold',
        'size': 10}
fig.text(0.1, 0.36, "B", fontdict=font)

# consider adding arrows and labels to plot

# draw a grid guide
grid = False
if grid:
    font = {'color':  'red',
            'weight': 'normal',
            'size': 20}
    for x in np.arange(0, 1, 0.05):
        for y in np.arange(0, 1, 0.05):
            fig.text(x, y, ".", fontdict=font)
plt.savefig(fig_dir+'glm_wholebrain_thresholded.svg', dpi=300, pad_inches=0.2)
plt.show()

# Raw statistical comparison plot

# plot Savage, current results and current two sample results
vmax = 7

fig, axs = plt.subplot_mosaic("""
                              A
                              B
                              C
                              """, figsize=(7, 5), constrained_layout=True)
plt.rc('axes', labelsize=12)

# plot our threat image
filename = 'con_0006/spmT_0001'
img = nib.load(secondlevel_dir+filename+'.nii')
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['A'])

# get Savage image
filename = (proj_dir+'data/derivatives/masks/Savage_threat.nii')
img = nib.load(filename)
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['B'])

# plot our between groups threat image
filename = 'con_0006/spmT_0001'
img = nib.load(between_dir+filename+'.nii')
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['C'])

# Text
# headings
font = {'color':  'black',
        'weight': 'bold',
        'size': 10}
fig.text(0.1, 0.90, "A.", fontdict=font)
fig.text(0.1, 0.62, "B", fontdict=font)
fig.text(0.1, 0.35, "C", fontdict=font)

font = {'color':  'black',
        'weight': 'normal',
        'size': 9}
fig.text(0.14, 0.90, "Threat reversal (current results)", fontdict=font)
fig.text(0.14, 0.62, "Threat reversal (Savage et al., 2020)", fontdict=font)
fig.text(0.14, 0.35, "HC > OCD", fontdict=font)

plt.savefig(fig_dir+'glm_wholebrain_threat_sup.jpeg', dpi=300, pad_inches=0.2)
plt.show()

# plot Safety results
vmax = 7

fig, axs = plt.subplot_mosaic("""
                              A
                              B
                              C
                              """, figsize=(7, 5), constrained_layout=True)
plt.rc('axes', labelsize=12)

# plot our threat image
filename = 'con_0007/spmT_0001'
img = nib.load(secondlevel_dir+filename+'.nii')
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['A'])

# get Savage image
filename = (proj_dir+'data/derivatives/masks/Savage_safety.nii')
img = nib.load(filename)
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['B'])

# plot our between groups threat image
filename = 'con_0007/spmT_0001'
img = nib.load(between_dir+filename+'.nii')
plotting.plot_glass_brain(img,
                          threshold=0,
                          cmap=plotting.cm.bwr,
                          vmax=vmax,
                          display_mode='lyrz',
                          colorbar=True,
                          plot_abs=False,
                          axes=axs['C'])

# Text
# headings
font = {'color':  'black',
        'weight': 'bold',
        'size': 10}
fig.text(0.1, 0.90, "A.", fontdict=font)
fig.text(0.1, 0.62, "B", fontdict=font)
fig.text(0.1, 0.35, "C", fontdict=font)

font = {'color':  'black',
        'weight': 'normal',
        'size': 9}
fig.text(0.14, 0.90, "Safety reversal (current results)", fontdict=font)
fig.text(0.14, 0.62, "Safety reversal (Savage et al., 2020)", fontdict=font)
fig.text(0.14, 0.35, "HC > OCD", fontdict=font)
plt.savefig(fig_dir+'glm_wholebrain_safety_sup.jpeg', dpi=300, pad_inches=0.2)
plt.show()
# %%
