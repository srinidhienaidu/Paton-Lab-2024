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
NUM_NEURONS = 2
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


# for neuron in range(min(NUM_NEURONS, 5)):  
#     plt.figure(figsize=(15, 10))

#     for condition in range(NUM_COND):
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

X_T = np.zeros((NUM_NEURONS, (NUM_TIME * NUM_DEC * NUM_TRIALS * NUM_STIM)))
X_TS = np.zeros((NUM_NEURONS, (NUM_TIME * NUM_DEC * NUM_TRIALS * NUM_STIM)))
X_TD = np.zeros((NUM_NEURONS, (NUM_TIME * NUM_DEC * NUM_TRIALS * NUM_STIM)))
X_TSD = np.zeros((NUM_NEURONS, (NUM_TIME * NUM_DEC * NUM_TRIALS * NUM_STIM)))
X_NOISE = np.zeros((NUM_NEURONS, (NUM_TIME * NUM_DEC * NUM_TRIALS * NUM_STIM)))

for cell in range(NUM_NEURONS):

    x_tsdk = binned_spiking_data[cell]

    x = np.mean(x_tsdk)
    x_t = np.mean((x_tsdk - x), axis = 0) 
    x_s = np.zeros((NUM_STIM))
    x_d = np.zeros((NUM_DEC))

    x_s_pad = np.zeros((NUM_TRIALS, 1))
    x_ts_pad = np.zeros((NUM_TRIALS, NUM_TIME))

    x_d_pad = np.zeros((NUM_TRIALS, 1))
    x_td_pad = np.zeros((NUM_TRIALS, NUM_TIME))

    x_ts = np.zeros((NUM_STIM, NUM_TIME)) 
    x_td = np.zeros((NUM_DEC, NUM_TIME))

    x_sd = np.zeros((NUM_STIM, NUM_DEC))
    x_sd_pad = np.zeros((NUM_TRIALS, NUM_TIME))

    for stim in range(NUM_STIM):
        stim_coord = np.where(np.array(stim_dep) == (stim + 1))[0]
        x_s[stim] = (np.mean(x_tsdk[stim_coord] - x))
        x_s_pad[stim_coord] = np.mean(x_tsdk[stim_coord] - x)

    for dec in range(NUM_DEC):

        dec_coor = np.where(np.array(dec_dep) == (dec + 1))[0]
        x_d[dec] = np.mean(x_tsdk[dec_coor] - x)
        x_d_pad[dec_coor] = np.mean(x_tsdk[dec_coor] - x)

    x_arr_subtracted = x_tsdk - x - x_t - x_s_pad - x_d_pad

    for stim in range(NUM_STIM):
        stim_coord = np.where(np.array(stim_dep) == (stim + 1))[0]
        x_ts[stim] = np.mean(x_arr_subtracted[stim_coord], axis = 0)
        x_ts_pad[stim_coord] = np.mean(x_arr_subtracted[stim_coord], axis = 0)

        for dec in range(NUM_DEC):
            dec_coor = np.where(np.array(dec_dep) == (dec + 1))[0]
            intersect_coord = np.intersect1d(stim_coord, dec_coor)
            sd_val = np.mean(x_arr_subtracted[intersect_coord])
            x_sd[stim, dec] = sd_val

            x_sd_pad[intersect_coord, :] = np.stack([sd_val] * NUM_TIME, axis = 0) 

    for dec in range(NUM_DEC):
        dec_coor = np.where(np.array(dec_dep) == (dec + 1))[0]
        x_td[dec] = np.mean(x_arr_subtracted[dec_coor], axis = 0)
        x_td_pad[dec_coor] = np.mean(x_arr_subtracted[dec_coor], axis = 0) 

    x_tsd = (x_arr_subtracted - x_ts_pad - x_td_pad - x_sd_pad)
    sigma_tsdk = x_tsdk - np.mean(x_tsdk, axis = 0)

    x_bar_reshaped = np.full((NUM_STIM * NUM_DEC * NUM_TRIALS * NUM_TIME), x)
    X_T[cell] = np.stack([x_t] * (NUM_TRIALS * NUM_STIM * NUM_DEC), axis = 0).flatten()
    X_TS[cell] = np.stack([x_ts_pad.flatten()] * (NUM_STIM * NUM_DEC), axis = 0).flatten()
    X_TD[cell] = np.stack([x_td_pad.flatten()] * (NUM_STIM * NUM_DEC), axis = 0).flatten()
    X_TSD[cell] = np.stack([x_tsd.flatten()] * (NUM_STIM * NUM_DEC), axis = 0).flatten()
    X_NOISE[cell] = np.stack([sigma_tsdk.flatten()] * (NUM_STIM * NUM_DEC), axis = 0).flatten()

    # reconstruction = x_reshaped + x_t_reshaped + x_ts_reshaped + x_td_reshaped + x_tsd_reshaped + sigma_reshaped
    # X[cell] = reconstruction
    # print(reconstruction) 

X = X_T + X_TS + X_TD + X_TSD + X_NOISE
# %%
## Core dPCA ####################################################################################################################################################
#################################################################################################################################################################################

# First testing out for X_T
NUM_EVAL = 1

A_OLS = X_TS @ X.T @ np.linalg.inv(X @ X.T)
A_cov = np.cov(A_OLS @ X)
evec, eval = np.linalg.eigh(A_cov)

idx = eval.argsort()[::-1]   
eigenValues = eval[idx]
eigenVectors = evec[idx]

U_q = eigenVectors[:, :NUM_EVAL]
A_q = U_q @ U_q.T @ A_OLS 

F = U_q
D = U_q.T @ A_OLS
# %%
dec_1_spikes = np.mean(binned_spiking_data[:, dec_1], axis = 1)
mean_1 = np.mean(dec_1_spikes, axis = 0)

demean_1 = dec_1_spikes - mean_1
dec_2_spikes = np.mean(binned_spiking_data[:, dec_2], axis = 1)

plt.scatter(dec_1_spikes[0], dec_1_spikes[1], label = 'Decision 1')
plt.scatter(dec_2_spikes[0], dec_2_spikes[1], label = 'Decision 2')
plt.xlabel("FR Neuron 1")
plt.ylabel("FR Neuron 2")
plt.legend()

# %%

# %%
projected_points = np.zeros((NUM_TIME, NUM_NEURONS))
dim_1 = np.dot(demean_1.T, D.T)
for idx, val in enumerate(dim_1):
    print(((val * D.T) + mean_1).shape)
    # projected_points[idx] = 
# %%

# %%

# %%
