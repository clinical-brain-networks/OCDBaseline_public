# %%
'''
Want to find how long occured between MRI and behavioural session
'''

import numpy as np
import pandas as pd
import datetime
import glob
from tqdm import tqdm
import sys
import pydicom as dicom
import nibabel.nicom.csareader as csareader
sys.path.insert(0, '..')

# import my own code
from functions.data_helpers import get_phenotype, get_subj_group, get_computer

# parameters
_, proj_dir = get_computer()
bids_dir = proj_dir+'data/bids/'
subj_list = list(np.loadtxt('../subject_list_exclusions.txt', dtype='str'))
df = get_phenotype(subj_list, ['participants', 'wasi'])

# Assume the WASI test date is the neuropsyc date
date1 = []
for subj in tqdm(subj_list):
    date = df.loc[df.participant_id == subj].wasi_test_date.values
    date1.append(date)

# oh god you have to look at the raw DICOMS jfc
# this is very slow
date2 = []
raw_dir = '/home/lukeh/LabData/Lab_LucaC/Luke/OCD_CT_BIDS/data/'
for subj in tqdm(subj_list):
    
    if get_subj_group(subj) == 'control':
        n = subj.split('l')[1].zfill(2)
        p = raw_dir+'rawdata_amended_HC/*'+n+'_*'
    elif get_subj_group(subj) == 'patient':
        n = subj.split('nt')[1].zfill(2)
        p = raw_dir+'rawdata_amended_P/1_Pre_TMS/*'+n+'_*'
    
    path_to_dicoms = glob.glob(p)
    files = glob.glob(path_to_dicoms[0]+ '/**/*.dcm', recursive=True)
    if len(files) == 0:
        files = glob.glob(path_to_dicoms[0]+ '/**/*.IMA', recursive=True)
    dcm = dicom.read_file(files[0])
    # print(subj)
    # print('\t', n)
    # print('\t', p)
    # print('\t', files[0])
    date = dcm[int('00080022', 16)].value
    date2.append(date)


# %% convert using datetime
df = pd.DataFrame(columns=['subj','date1', 'date2'])

for i, subj in enumerate(subj_list):
    
    d = date1[i][0]
    if subj == 'sub-control09':
        d = '2018-11-22'  # mistake in data (verified via caitlin ss)

    if subj == 'sub-control14':
        d = '2018-11-22'  # mistake in data (verified via caitlin ss)

    if subj == 'sub-control21':
        d = '2019-02-08'  # mistake in data (verified via caitlin ss)
    
    if subj == 'sub-patient07':
        d = '2018-06-15'  # poor formatting in spreadsheet
    
    year = d[0:4]
    month = d[5:7]
    day = d[8:10]
    d1 = datetime.date(int(year), int(month), int(day))

    d = date2[i]
    if subj == 'sub-patient22':
        d = '20190910'  # mistake in data (verified via DICOM zip file name)
    year = d[0:4]
    month = d[4:6]
    day = d[6:8]
    d2 = datetime.date(int(year), int(month), int(day))

    
    row = {'subj': subj,
           'date1': d1,
           'date2': d2}
    df = df.append(row, ignore_index=True)

df['difference'] = (df['date2'] - df['date1'])
df['diff_days'] = df['difference'].dt.days
df.head(10)

print('Abs mean:')
print(np.mean(abs(df['diff_days'].values)))
print('Abs std:')
print(np.std(abs(df['diff_days'].values)))
# %%
df.to_csv('../../results/time_between.csv', index=False)
# %%
