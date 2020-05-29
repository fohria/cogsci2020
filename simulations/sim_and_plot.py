# %% md
"""
# # shapetask
# ## - simulate and plot -

hello and welcome to the fabulous shape sequence task

paths may need to be changed depending on your environment. we've mostly used atom editor + hydrogen plugin
"""

# %%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# environment settings
%load_ext autoreload
%autoreload 2
sns.set()  # resets to dark background
# sns.set_palette("pastel6")  # light fresh graphs
sns.set_palette("colorblind")  # darker but colorblind friendly

# %%
"""
# ## simulation
"""

# create parameter set
from classes import Parameters
params = Parameters(alpha = 0.5,
                    beta = 1,
                    gamma = 0.5,
                    task = 'b0b-nr',
                    # task = 'b0b',
                    # task = 'random',
                    trials = 99)

# generate human readable sequence
from functions import generate_sequence
sequence = generate_sequence(params)

# simulate subject with sequence
from functions import simulate_subject
actions, shape_sequence = simulate_subject(params, sequence)

# %%
"""
# ## collate data and plot results
"""

from functions import create_dataframe
subject_data = create_dataframe(actions, shape_sequence)

# reshape data for plotting (a bit ugly right now but whatever)
from functions import reshape_data
df_dict = reshape_data(subject_data, params)

# barplots for each shape position
from functions import barplot_shape
barplot_shape(1, df_dict, params)
barplot_shape(2, df_dict, params)
barplot_shape(3, df_dict, params)

# %%
"""
# line plots to visualise learning rates
"""

from functions import reshape_data, lineplot
df_dict = reshape_data(subject_data, params)
lineplot(df_dict, params)

# %%
"""
# special plot to visualise choices against actual sequence
"""

# shape sequence for subject
from functions import prep_data_for_choiceplot
df_sequence = prep_data_for_choiceplot(sequence, yval = 1)

# action selection sequence for subject
actionmap = {
    0: 'circle',
    1: 'square',
    2: 'triangle'
}
actionshapes = [actionmap[x] for x in actions]
df_actions = prep_data_for_choiceplot(actionshapes, yval = 2)

from functions import choiceplot
choiceplot(df_actions, df_sequence)
