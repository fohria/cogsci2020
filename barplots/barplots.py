from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# define datadirs

import os
os.chdir("../../")

# EXPA: random version from pilot
exp_path = Path.cwd() / 'data/pilot/'
exp_path2 = ''
exp_name = 'Experiment A (random version fixed sequence)'
total_trials = 270
save = True

# SIMA
exp_path = Path.cwd() / 'cogsci_paper/simA'
exp_path2 = ''
exp_name = 'Simulation A (random bags using BQL)'
total_trials = 270
save = True

# SIMA REPRO
exp_path = Path.cwd() / 'cogsci_paper/test'
exp_path2 = ''
exp_name = 'test repro simulation a (random BQL?)'
total_trials = 270
save = False

# SIMA STATIC
exp_path = Path.cwd() / 'cogsci_paper/simAstatic'
exp_path2 = ''
exp_name = 'Simulation A (random version fixed sequence using BQL)'
total_trials = 270
save = True

# EXPB
exp_path = Path.cwd() / 'data/b0b-nr_additional/'
exp_path2 = Path.cwd() / 'data/prolific_pilot_b0b-nr'
exp_name = 'Experiment B (bag of bags without repeat)'
total_trials = 99
save = True

# SIMB
exp_path = Path.cwd() / 'cogsci_paper/simB'
exp_path2 = ''
exp_name = 'Simulation B1 (bag of bags without repeat using BQL)'
total_trials = 99
save = True

# SIMB2
exp_path = Path.cwd() / 'cogsci_paper/simB2'
exp_path2 = ''
exp_name = 'Simulation B2 (bag of bags without repeat using SEQL)'
total_trials = 99
save = True

# EXPC
exp_path = Path.cwd() / 'data/b0b_additional'
exp_path2 = Path.cwd() / 'data/prolific_pilot_b0b'
exp_name = 'Experiment C (bag of bags)'
total_trials = 99
save = False

# SIMC1
exp_path = Path.cwd() / 'cogsci_paper/simC1'
exp_path2 = ''
exp_name = 'Simulation C1 (bag of bags using BQL)'
total_trials = 99
save = False

# SIMC2
exp_path = Path.cwd() / 'cogsci_paper/simC2'
exp_path2 = ''
exp_name = 'Simulation C2 (bag of bags using SEQL)'
total_trials = 99
save = False

# EXP D -- random prolific
exp_path = Path.cwd() / 'data/random_additional'
exp_path2 = Path.cwd() / 'data/prolific_pilot_randombags'
exp_name = 'Experiment D (randomly seeded sequence)'
total_trials = 99
save = False

# SIM D A
exp_path = Path.cwd() / 'cogsci_paper/simDa'
exp_path2 = ''
exp_name = 'Simulation D random task SEQL 1'
total_trials = 99
save = False

#%%

"""
    copy csvs to tempdir; for using multiple data_dirs
"""

def copy_to_temp(data_dir, data_dir2='', remove_old=True):

    import os
    import shutil
    tempdir = Path.cwd() / 'analysis/datatemp'
    if remove_old:
        for _file in tempdir.glob('*.csv'):
            os.remove(_file)
    last_filename = ''
    for _file in data_dir.glob('*.csv'):
        shutil.copy(_file, tempdir / _file.name)
        last_filename = _file.name
    last_filename = int(last_filename[8:10])
    if data_dir2 != '':
        for _file in data_dir2.glob('*.csv'):
            last_filename += 1
            filename = 'subject_' + str(last_filename).zfill(2) + '.csv'
            shutil.copy(_file, tempdir / filename)

#%%

import numpy as np

nbr_of_bags = int(total_trials / 3)
nbr_of_bobs = int(nbr_of_bags / 3)
shape_pos = np.tile([1,2,3], nbr_of_bags)

workdir = Path.cwd() / 'analysis/datatemp'
copy_to_temp(exp_path / 'csv')

# bla = pd.read_csv(workdir / 'subject_02.csv')

# just do below for all 4 columns we want;
# 'pred_correct', 'shift_predict', 'winstay', 'loseshift'
# yes, but do below for each shape position as we have for linegraphs

pred_correct = pd.DataFrame()
winstay = pd.DataFrame()
loseshift = pd.DataFrame()
shiftpredict = pd.DataFrame()

for subject in workdir.glob('*.csv'):

    subj_df = pd.read_csv(subject)
    # subj_df['shape_pos'] = shape_pos
    pred_correct[subject.name] = subj_df.pred_correct
    winstay[subject.name] = subj_df.winstay
    loseshift[subject.name] = subj_df.loseshift
    shiftpredict[subject.name] = subj_df.shift_predict

pred_correct['shape_pos'] = shape_pos
winstay['shape_pos'] = shape_pos
loseshift['shape_pos'] = shape_pos
shiftpredict['shape_pos'] = shape_pos
# pred_correct.mean()
# pred_correct['subject_23.csv'].mean()
# pred_correct['subject_23.csv'].std()

#%%
# seaborn setup

import seaborn as sns
sns.set(rc={'figure.facecolor':'white'})
sns.set_palette("colorblind")

#%%

# all_scores = pd.DataFrame()
shape1 = pd.DataFrame()

shape1['accuracy'] = pred_correct.query('shape_pos == 1').drop(columns=['shape_pos']).mean()
shape1['winstay'] = winstay.query('shape_pos == 1').drop(columns=['shape_pos']).mean()
shape1['loseshift'] = loseshift.query('shape_pos == 1').drop(columns=['shape_pos']).mean()
shape1['shiftpredict'] = shiftpredict.query('shape_pos == 1').drop(columns=['shape_pos']).mean()

sns.set(rc={'figure.facecolor':'white'})
sns.set_palette("colorblind")
sns.barplot(x='variable', y='value', data=shape1.melt())
plt.title(f'S1 for {exp_name}')
plt.xlabel('')
plt.ylabel('')
plt.ylim(0, 1)
if save:
    plt.savefig(f'cogsci_paper/barplots/{exp_name}_ball1.png', dpi=300)
plt.show()


#%%

shape2 = pd.DataFrame()

shape2['accuracy'] = pred_correct.query('shape_pos == 2').drop(columns=['shape_pos']).mean()
shape2['winstay'] = winstay.query('shape_pos == 2').drop(columns=['shape_pos']).mean()
shape2['loseshift'] = loseshift.query('shape_pos == 2').drop(columns=['shape_pos']).mean()
shape2['shiftpredict'] = shiftpredict.query('shape_pos == 2').drop(columns=['shape_pos']).mean()

sns.barplot(x='variable', y='value', data=shape2.melt())
plt.title(f'S2 for {exp_name}')
plt.xlabel('')
plt.ylabel('')
plt.ylim(0, 1)
if save:
    plt.savefig(f'cogsci_paper/barplots/{exp_name}_ball2.png', dpi=300)
plt.show()

#%%

shape3 = pd.DataFrame()

shape3['accuracy'] = pred_correct.query('shape_pos == 3').drop(columns=['shape_pos']).mean()
shape3['winstay'] = winstay.query('shape_pos == 3').drop(columns=['shape_pos']).mean()
shape3['loseshift'] = loseshift.query('shape_pos == 3').drop(columns=['shape_pos']).mean()
shape3['shiftpredict'] = shiftpredict.query('shape_pos == 3').drop(columns=['shape_pos']).mean()

sns.barplot(x='variable', y='value', data=shape3.melt())
plt.title(f'S3 for {exp_name}')
plt.xlabel('')
plt.ylabel('')
plt.ylim(0, 1)
if save:
    plt.savefig(f'cogsci_paper/barplots/{exp_name}_ball3.png', dpi=300)
plt.show()

#%%

print(f'S1 accuracy mean: {shape1.accuracy.mean()}')
print(f'S1 accuracy std: {shape1.accuracy.std()}')
print(f'S1 winstay mean: {shape1.winstay.mean()}')
print(f'S1 winstay std: {shape1.winstay.std()}')
print(f'S1 loseshift mean: {shape1.loseshift.mean()}')
print(f'S1 loseshift std: {shape1.loseshift.std()}')
print(f'S1 shiftpredict mean: {shape1.shiftpredict.mean()}')
print(f'S1 shiftpredict std: {shape1.shiftpredict.std()}')

print(f'S2 accuracy mean: {shape2.accuracy.mean()}')
print(f'S2 accuracy std: {shape2.accuracy.std()}')
print(f'S2 winstay mean: {shape2.winstay.mean()}')
print(f'S2 winstay std: {shape2.winstay.std()}')
print(f'S2 loseshift mean: {shape2.loseshift.mean()}')
print(f'S2 loseshift std: {shape2.loseshift.std()}')
print(f'S2 shiftpredict mean: {shape2.shiftpredict.mean()}')
print(f'S2 shiftpredict std: {shape2.shiftpredict.std()}')

print(f'S3 accuracy mean: {shape3.accuracy.mean()}')
print(f'S3 accuracy std: {shape3.accuracy.std()}')
print(f'S3 winstay mean: {shape3.winstay.mean()}')
print(f'S3 winstay std: {shape3.winstay.std()}')
print(f'S3 loseshift mean: {shape3.loseshift.mean()}')
print(f'S3 loseshift std: {shape3.loseshift.std()}')
print(f'S3 shiftpredict mean: {shape3.shiftpredict.mean()}')
print(f'S3 shiftpredict std: {shape3.shiftpredict.std()}')

#%%
# ttest expc & simc2

expc_s3shift = shape3.shiftpredict
simc2_s3shift = shape3.shiftpredict

len(simc2_s3shift)
len(expc_s3shift)

from scipy.stats import ttest_ind, ttest_rel

# degrees of freedom independent ttest: (N1 - 1) + (N2 - 1)
print(f'dof: {len(simc2_s3shift) + len(expc_s3shift) - 2}')
ttest_ind(simc2_s3shift, expc_s3shift)

# ttest expa
# uses paired ttest since we measure from same subjects

ttest_rel(shape3.shiftpredict, shape2.shiftpredict)
