'''
Performs PLS on brain activation and behavioural data in the OCDbaseline
project.

Due to the relatively small sample size (N=~90) the data input into the PLSC
undergoes dimensionality reduction (PCA). However, the number of components
to use is abitrary. We've chosen four per side, resulting in slighty
more than 10 subjects per feature (a good rule of thumb). But to make sure 
this isn't biasing the results we test from 2 - 6 just to make sure the 
results are somewhat robust.

'''
# %%
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import pyls
import pickle
import numpy as np
import pandas as pd
from glms.glm_beta_stats import load_betas
from functions.data_helpers import get_computer, get_phenotype
from functions.data_helpers import get_task_beh_data, get_awareness_labels

## Parameters, paths etc.,
# paths
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
deriv_dir = proj_dir+'data/derivatives/post-fmriprep-fix/'
out_flag = ''

# Supplemental analysis flags :/
exclude_unaware = False
exclude_subjective_ratings = False

# get subjects
subj_list = list(np.loadtxt('subject_list_exclusions.txt', dtype='str'))

if exclude_unaware:
    df = get_phenotype(subj_list)
    df['awareness'] = get_awareness_labels(subj_list)
    df = df.loc[df.awareness != 'patient-unaware']
    subj_list = df['participant_id'].to_list()
    out_flag = 'aware-only/'

# timeseries extraction method
ts_method = 'region'
eroded = False  # which set of ROIs to use

# sphere radius size (doesn't mean anything)
radius = 4

# glm analysis stream to use
glm_label = 'smooth-6mm_despike'

# roi information for plotting and statistics
roi_df = pd.read_csv('../code/roi_details.csv')

## PLS parameters
# these are the range of components to test across
comp_range = np.arange(2, 7)
n_perms = 20000
n_boots = 10000

## Organise input data:
# prepare activation betas (X)
beta_df = load_betas(subj_list, deriv_dir,
                     glm_label=glm_label, radius=radius, eroded=eroded)
beta_df = beta_df[beta_df.method == ts_method]

# Collect betas into a single matrix (subj x beta)
# (seperately for threat and safety)
X_raw = np.zeros((len(beta_df.subj.unique()), 1))
contrast_df = beta_df[beta_df.contrast == 'Safety reversal']
for roi in ['vmpfc', 'pcc']:
    b = contrast_df[contrast_df.roi == roi].value.values
    X_raw = np.hstack((X_raw, b.reshape(-1, 1)))

contrast_df = beta_df[beta_df.contrast == 'Threat reversal']
for roi in ['leftInsula', 'rightInsula', 'acc', 'leftPUT',
            'rightPUT', 'leftCAU', 'rightCAU', 'leftGP', 'rightGP']:
    b = contrast_df[contrast_df.roi == roi].value.values
    X_raw = np.hstack((X_raw, b.reshape(-1, 1)))

X_raw = np.delete(X_raw, 0, axis=1)
X_scaled = StandardScaler().fit_transform(X_raw)

# put all the raw data into a nice df to be saved
# (this is hard coded so be wary)
df = pd.DataFrame(columns=['vmpfc',
                           'pcc',
                           'leftInsula', 
                           'rightInsula', 
                           'acc', 
                           'leftPUT',
                           'rightPUT', 
                           'leftCAU', 
                           'rightCAU', 
                           'leftGP', 
                           'rightGP'], data=X_raw)
df['subj'] = subj_list
df.to_csv('../results/PLSC/'+out_flag+'X_raw.csv', index=False)

# Katerberg et al., symptom factors
# get yboc symptoms
ybocs_df = get_phenotype(subj_list, ['participants', 'ybocs_symptoms'])

# get factor information
factor_df = pd.read_csv('Katerberg_factors.csv')
factor_df = factor_df.dropna(axis=0)
factor_data = factor_df[factor_df.columns[factor_df.columns.str.contains(
    'factor_')]].values

item_list = factor_df.item_num.to_list()
item_list = ['item_' + str(int(i)) for i in item_list]  # 'add item'

factor_scores = np.zeros((len(ybocs_df.participant_id.unique()), 5))
for s, subj in enumerate(ybocs_df.participant_id.unique()):

    # get subject specific item data
    tmp = ybocs_df[ybocs_df.participant_id == subj]
    data = (tmp.loc[tmp.symptom_time == 'current', item_list].values
            + tmp.loc[tmp.symptom_time == 'past', item_list].values)
    # multiply subject data by factor weightings
    factor_scores[s, :] = np.sum(data.T * factor_data, axis=0)

# get task response information
# this data reflect subjective 'sensitivity' to the task manipulation
task_df = get_task_beh_data(subj_list, ratings_only=True)
task_df = task_df[task_df.rating_measure == 'arousal']

# anxious arousal
safe_data_a = ((task_df.loc[(task_df.phase == 'reversal')
                        & (task_df.trial_type == 'CS-')].rating.values)
            - (task_df.loc[(task_df.phase == 'conditioning')
                            & (task_df.trial_type == 'CS+')].rating.values))
threat_data_a = ((task_df.loc[(task_df.phase == 'reversal')
                        & (task_df.trial_type == 'CS+')].rating.values)
            - (task_df.loc[(task_df.phase == 'conditioning')
                            & (task_df.trial_type == 'CS-')].rating.values))
data_a = (threat_data_a - safe_data_a) * -1  # same direction as data_v

# do the same for valence
task_df = get_task_beh_data(subj_list, ratings_only=True)
task_df = task_df[task_df.rating_measure == 'valence']

# valence
safe_data_v = ((task_df.loc[(task_df.phase == 'reversal')
                        & (task_df.trial_type == 'CS-')].rating.values)
            - (task_df.loc[(task_df.phase == 'conditioning')
                            & (task_df.trial_type == 'CS+')].rating.values))
threat_data_v = ((task_df.loc[(task_df.phase == 'reversal')
                        & (task_df.trial_type == 'CS+')].rating.values)
            - (task_df.loc[(task_df.phase == 'conditioning')
                            & (task_df.trial_type == 'CS-')].rating.values))
data_v = (threat_data_v - safe_data_v) 
task_ratings = np.hstack((data_a.reshape(-1, 1), data_v.reshape(-1, 1)))
task_ratings = task_ratings * -1  # this ensures coding such that higher values = larger differences between conditions

y_raw = np.hstack((factor_scores, task_ratings))

if exclude_subjective_ratings:
    y_raw = factor_scores.copy()
    out_flag = 'no_ratings/'

y_scaled = StandardScaler().fit_transform(y_raw)

# put all the raw data into a nice df to be saved
# (this is hard coded so be wary)
if exclude_subjective_ratings:
    df = pd.DataFrame(columns=['factor_taboo',
                               'factor_contamination_cleaning',
                               'factor_doubts',
                               'factor_rituals_superstition',
                               'factor_hoarding_symmetry'], data=y_raw)
else:
    df = pd.DataFrame(columns=['factor_taboo',
                               'factor_contamination_cleaning',
                               'factor_doubts',
                               'factor_rituals_superstition',
                               'factor_hoarding_symmetry',
                               'subjective_anxious',
                               'subjective_valence'], data=y_raw)

df['subj'] = subj_list
df.to_csv('../results/PLSC/'+out_flag+'y_raw.csv', index=False)

## Perform looped PLSC
r_matrix = np.zeros((len(comp_range), len(comp_range)))
p_matrix = np.zeros((len(comp_range), len(comp_range)))

for n_X in comp_range:
    for n_y in comp_range:
        print('Running:',n_X, n_y)

        # perform PCA on X
        pca = PCA(n_components=n_X).fit(X_scaled)
        X = pca.transform(X_scaled)

        # perform PCA on y
        pca = PCA(n_components=n_y).fit(y_scaled)
        y = pca.transform(y_scaled)

        # run PLSC
        bpls = pyls.behavioral_pls(X, y, n_perm=n_perms, n_boot=n_boots)

        # run the reverse - lazy mans way of getting
        # CIs in the original X variable
        bpls_rev = pyls.behavioral_pls(y, X, n_perm=n_perms, n_boot=n_boots)

        # run correlations between the original variables and
        # PLSC results

        out_file = ('../results/PLSC/'
                    + out_flag
                    + 'eroded-'+str(eroded)
                    + '_Xnum-'+str(n_X)
                    + '_ynum-'+str(n_y)
                    + '.pkl')
        with open(out_file, 'wb') as outp:  # Overwrites any existing file.
                pickle.dump(bpls, outp, pickle.HIGHEST_PROTOCOL)
                pickle.dump(bpls_rev, outp, pickle.HIGHEST_PROTOCOL)



# %%
