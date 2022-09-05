# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import sys
sys.path.insert(0, '..')

# import my own code
from functions.data_helpers import get_phenotype,  get_computer, get_mean_fd

# parameters
_, proj_dir = get_computer()
subj_list = list(np.loadtxt('../subject_list_exclusions.txt', dtype='str'))

demo_df = pd.DataFrame(columns=['measure',
                                'HC mean',
                                'HC std',
                                'HC low',
                                'HC high',
                                'OCD mean',
                                'OCD std',
                                'OCD low',
                                'OCD high',
                                'stat',
                                'es',
                                'p'])

# get head motion data
motion = get_mean_fd(subj_list, ['rest', 'fearRev'])

# get data
df = get_phenotype(subj_list, ['participants', 'wasi', 'ybocs', 'mini','hads',
                                    'hama', 'madrs', 'obq', 'ocir'])
N = df['group'].value_counts()

# put motion in df
df['fd_rest'] = motion[motion.task == 'rest'].framewise_displacement_avg.values.copy()
df['fd_task'] = motion[motion.task == 'fearRev'].framewise_displacement_avg.values.copy()


# continious variables
for label in ['age',
              'fsiq-4_comp_score',
              'obsession_subtotal',
              'compulsion_subtotal',
              'ybocs_total',
              'obq_total',
              'ocir_total',
              'hads_anxiety_total',
              'hads_depression_total',
              'hama_total',
              'madrs_total',
              'fd_rest',
              'fd_task',
              ]:

    means = df.groupby(by='group')[label].mean()
    stds = df.groupby(by='group')[label].std()
    mins = df.groupby(by='group')[label].min()
    maxs = df.groupby(by='group')[label].max()

    # do a stat
    res = pg.ttest(df[df.group == 'control'][label].values,
                   df[df.group == 'patient'][label].values)
    # create new row
    row = {'measure': label,
           'HC mean': means.control,
           'HC std': stds.control,
           'HC low': mins.control,
           'HC high': maxs.control,
           'OCD mean': means.patient,
           'OCD std': stds.patient,
           'OCD low': mins.patient,
           'OCD high': maxs.patient,
           'stat': res['T'].values[0],
           'es': res['cohen-d'].values[0],
           'p': res['p-val'].values[0],
           }
    # add to row
    demo_df = demo_df.append(row, ignore_index=True)


# ordinal variables
label = 'gender'
counts = df.groupby(by='group')[label].value_counts()
mins = df.groupby(by='group')[label].min()
maxs = df.groupby(by='group')[label].max()

# do chi square
exp, obs, res = pg.chi2_independence(df, x=label, y='group')

# n male and %
perc_control = (counts.control.male /
                (counts.control.male+counts.control.female))*100
perc_patient = (counts.patient.male /
                (counts.patient.male+counts.patient.female))*100

# create new row
row = {'measure': label,
       'HC mean': counts.control.male,
       'HC std': perc_control,
       'OCD mean': counts.patient.male,
       'OCD std': perc_patient,
       'stat': res['chi2'].values[0],
       'es': res['cramer'].values[0],
       'p': res['pval'].values[0],
       }
# add to row
demo_df = demo_df.append(row, ignore_index=True)

# handedness
label = 'handedness'
counts = df.groupby(by='group')[label].value_counts()
mins = df.groupby(by='group')[label].min()
maxs = df.groupby(by='group')[label].max()

# do chi square
exp, obs, res = pg.chi2_independence(df, x=label, y='group')

# n righthanded and %
perc_control = (counts.control.right /
                (counts.control.right+counts.control.left))*100
perc_patient = (counts.patient.right /
                (counts.patient.right+counts.patient.left))*100

# create new row
row = {'measure': label,
       'HC mean': counts.control.right,
       'HC std': perc_control,
       'OCD mean': counts.patient.right,
       'OCD std': perc_patient,
       'stat': res['chi2'].values[0],
       'es': res['cramer'].values[0],
       'p': res['pval'].values[0],
       }

# add to row
demo_df = demo_df.append(row, ignore_index=True)

# manually curate final table for paper


def make_neat_cont_row(new_label, old_label, demo_df, HC_label, OCD_label):
    index = demo_df[demo_df.measure == old_label].index[0]
    row = {'_': new_label,
           HC_label: str(demo_df[demo_df.measure == old_label]
                         ['HC mean'][index].round(r))
           + ' ('+str(demo_df[demo_df.measure == old_label]
                      ['HC std'][index].round(r))+')',
           OCD_label: str(demo_df[demo_df.measure == old_label]
                          ['OCD mean'][index].round(r))
           + ' ('+str(demo_df[demo_df.measure == old_label]
                      ['OCD std'][index].round(r))+')',
           'p': str(demo_df[demo_df.measure == old_label]['p'][index].round(r))
           }
    return row


def make_neat_cat_row(new_label, old_label, demo_df, HC_label, OCD_label):
    index = demo_df[demo_df.measure == old_label].index[0]
    row = {'_': new_label,
           HC_label: str(demo_df[demo_df.measure == old_label]['HC std'][index].round(r)),
           OCD_label: str(demo_df[demo_df.measure == old_label]['OCD std'][index].round(r)),
           'p': str(demo_df[demo_df.measure == old_label]['p'][index].round(3))
           }
    return row


r = 2  # how many decimal rounding?
HC_label = 'Control (N='+str(N.control)+')'
OCD_label = 'OCD (N='+str(N.patient)+')'
table = pd.DataFrame(columns=['_',
                              HC_label,
                              OCD_label,
                              'p'])

row = make_neat_cont_row('Age', 'age', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cat_row('Gender (\% Male)', 'gender', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cat_row('Handedness (\% Right)', 'handedness', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('IQ (WASI)', 'fsiq-4_comp_score', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('YBOCS', 'ybocs_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('\quad Obsessions', 'obsession_subtotal', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('\quad Compulsions', 'compulsion_subtotal', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('OBQ', 'obq_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('OCIR', 'ocir_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('HADS: anxiety', 'hads_anxiety_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('HADS: depression', 'hads_depression_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('HAMA', 'hama_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('MADRS', 'madrs_total', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('FD: Resting state', 'fd_rest', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

row = make_neat_cont_row('FD: Task', 'fd_task', demo_df, HC_label, OCD_label)
table = table.append(row, ignore_index=True)

display(table.head(20))
table.to_csv('../../results/demographics.csv', index=False)


# %%
