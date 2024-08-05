# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from pathlib import Path
import pandas as pd
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma
from scipy.signal import convolve2d
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import train_test_split
# %%
## Generating Tester Data ###############################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################################
NUM_NEURONS = 50
NUM_TRIALS = 80
NUM_COND = 8
conditions = []
stim_dep = []
dec_dep = []
TIME_PTS = 200 
NUM_STIM = 4
NUM_DEC = 2
NUM_TIME = 100
np.random.seed(42)
base_rates = np.random.rand(NUM_NEURONS, NUM_COND) * 10  

stimulus_dependence = np.random.randint(1, 5, size=(NUM_TRIALS))
decision_dependence = np.random.randint(1, 3, size=(NUM_NEURONS, NUM_COND))

spiking_data = np.zeros((NUM_NEURONS, NUM_TRIALS, TIME_PTS))
spiking_data_array = []
for trial in range(NUM_TRIALS):
    condition = trial % NUM_COND  

    if condition % 2 == 0:
        dec = 2
    else:
        dec = 1


    stim = stimulus_dependence[trial]
    stim_dep.append(stim)
    # conditions.append(condition)

    dec_dep.append(dec)

    for neuron in range(NUM_NEURONS):
        rate = base_rates[neuron, condition] + stim * np.sin(trial / NUM_TRIALS * 2 * np.pi)
        rate += dec * np.cos(trial / NUM_TRIALS * 2 * np.pi)
        rate = max(rate, 0)
        if np.isnan(rate):
            rate = 0
        
        spikes = np.random.poisson(rate, TIME_PTS)
        spiking_data[neuron, trial, :] = spikes

# %%
## Finding indices decision and tone specific indices ################################################################################################################################################################################
#################################################################################################################################################################################
dec_1 = np.where(np.array(dec_dep) == 1)[0]
dec_2 = np.where(np.array(dec_dep) == 2)[0]
tone_1 = np.where(np.array(stim_dep) == 1)[0]
tone_2 = np.where(np.array(stim_dep) == 2)[0]
tone_3 = np.where(np.array(stim_dep) == 3)[0]
tone_4 = np.where(np.array(stim_dep) == 4)[0]

# %%
## Binning our data, as well as plotting rasters and PSTHs ################################################################################################################################################################################
#################################################################################################################################################################################

bin_count = 2
binned_spiking_data = np.zeros((NUM_NEURONS, NUM_TRIALS, TIME_PTS // bin_count))
# smoothed_spiking_data = np.zeros((NUM_TRIALS, NUM_NEURONS, TIME_PTS // bin_count))

for trial in range(NUM_TRIALS):
    for neuron in range(NUM_NEURONS):
        bin_raw_data = np.sum(spiking_data[neuron][trial].reshape(-1, bin_count), axis=1)
        binned_spiking_data[neuron, trial] = bin_raw_data
    

# plt.figure(figsize=(15, 10))
# for neuron in range(5):
#     plt.subplot(5, 1, neuron + 1)
#     for trial in range(spiking_data.shape[0]):
#         spikes = np.where(spiking_data[trial, :, neuron, :] > 0)[1]
#         plt.scatter(spikes, np.ones_like(spikes) * trial, s=1, color='black')
#     plt.title(f'Neuron {neuron}')
#     plt.xlabel('Time Bin')
#     plt.ylabel('Trial')
# plt.tight_layout()
# plt.show()

# colors = ['k', 'k', 'r', 'r', 'g', 'g', 'b', 'b']


# for neuron in range(min(num_neurons, 5)):  
#     plt.figure(figsize=(15, 10))

#     for condition in range(num_conditions):
#         if condition % 2 == 0:
#             linetype = 'solid'
#         else:
#             linetype = 'dashed'

#         plt.plot(np.mean(binned_data[:, condition, neuron, :], axis=0), label=f'Condition {condition}', linestyle = linetype, color = colors[condition])
#     plt.title(f'Neuron {neuron + 1} - Binned Spike Counts')
#     plt.xlabel('Time')
#     plt.ylabel('Rate')
#     # plt.legend()
# plt.tight_layout()
# plt.show()

# %%

# %%
## Marginalization Procedure ####################################################################################################################################################
#################################################################################################################################################################################

# dPCA_full = binned_spiking_data.transpose(1, 0, 2)
NUM_TIME = 100

neuron_1 = binned_spiking_data[0]

x = np.mean(neuron_1)
x_t = np.mean((neuron_1 - x), axis = 0) 
x_s = np.zeros((NUM_STIM))
x_d = np.zeros((NUM_DEC))

x_s_pad = np.zeros((NUM_TRIALS, 1))
x_ts_pad = np.zeros((NUM_TRIALS, NUM_TIME))

x_d_pad = np.zeros((NUM_TRIALS, 1))
x_td_pad = np.zeros((NUM_TRIALS, NUM_TIME))

x_ts = np.zeros((NUM_STIM, NUM_TIME)) 
x_td = np.zeros((NUM_DEC, NUM_TIME))

x_sd = np.zeros((NUM_STIM, NUM_DEC))
x_sd_temp_pad = []
x_sd_pad = np.zeros((NUM_TRIALS, NUM_TIME))

x_tsd = np.zeros((NUM_STIM, NUM_DEC, NUM_TIME))

x_tsdk = np.zeros((NUM_STIM, NUM_DEC, NUM_TRIALS, NUM_TIME))

for stim in range(NUM_STIM):
    stim_coord = np.where(np.array(stim_dep) == (stim + 1))[0]
    x_s[stim] = (np.mean(neuron_1[stim_coord] - x))
    x_s_pad[stim_coord] = np.mean(neuron_1[stim_coord] - x)
    x_ts_pad[stim_coord] = np.mean((neuron_1[stim_coord] - x), axis = 0)

for dec in range(NUM_DEC):

    dec_coor = np.where(np.array(dec_dep) == (dec + 1))[0]

    x_d[dec] = np.mean(neuron_1[dec_coor] - x)
    x_d_pad[dec_coor] = np.mean(neuron_1[dec_coor] - x)
    x_td_pad[dec_coor] = np.mean((neuron_1[dec_coor] - x), axis = 0) 

x_arr_subtracted = neuron_1 - x - x_t - x_s_pad - x_d_pad

for stim in range(NUM_STIM):
    stim_coord = np.where(np.array(stim_dep) == (stim + 1))[0]
    x_ts[stim] = np.mean(x_arr_subtracted[stim_coord], axis = 0)

    for dec in range(NUM_DEC):
        
        dec_coor = np.where(np.array(dec_dep) == (dec + 1))[0]
        intersect_coord = np.intersect1d(stim_coord, dec_coor)
        x_sd[stim, dec] = np.mean(x_arr_subtracted[intersect_coord])
        x_sd_pad[intersect_coord, :] = np.mean(x_arr_subtracted[intersect_coord], axis = 0)

for dec in range(NUM_DEC):
    x_td[dec] = np.mean(x_arr_subtracted[dec_coor], axis = 0)

x_tsd_subtracted = x_arr_subtracted - x_ts_pad - x_td_pad - x_sd_pad


# %%
concat_by_time = x_tsd_subtracted.flatten()
concat_by_time = concat_by_time.reshape(1, concat_by_time.shape[0])

sigma_tsdk = np.stack([concat_by_time] * NUM_TRIALS, axis=-1)
sigma_tsdk = sigma_tsdk.reshape(1, 640000)
# %%
print(x_sd_pad)

# %%
