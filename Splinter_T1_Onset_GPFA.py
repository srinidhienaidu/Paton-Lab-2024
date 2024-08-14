# %%
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import matplotlib.cm as cm
from pathlib import Path
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from matplotlib.gridspec import GridSpec
from scipy.stats import gamma
from scipy.signal import convolve2d
from scipy.optimize import minimize
from sklearn.cross_decomposition import PLSRegression
from elephant.gpfa import GPFA
import neo
import quantities as pq
from einops import rearrange
import scipy.ndimage
import torch
from mpl_toolkits.mplot3d import Axes3D

# %%
# base_dir = r'C:\Users\srini\Documents\GitHub\Paton-Lab--Summer-2024\Splinter' # Champalimaud PC dir

base_dir = r'/Users/srinidhienaidu/Downloads/GitHub/Paton-Lab--Summer-2024/Splinter' # Srinidhi Mac dir
session = 0 # Change based on which trial you want
date = sorted(os.listdir(base_dir))[session + 1]
path = os.path.join(base_dir, date)
os.chdir(path)
print(path)

# %%
N_T1_Onset_bins = np.load('n_GPFA_bins.npy', allow_pickle= True)
N_array_bins = np.stack((N_T1_Onset_bins), axis = -1)
S_T1_Onset_bins = np.load('s_GPFA_bins.npy', allow_pickle= True)
S_array_bins = np.stack((S_T1_Onset_bins), axis = -1)
C_T1_Onset_bins = np.load('c_GPFA_bins.npy', allow_pickle= True)
C_array_bins = np.stack((C_T1_Onset_bins), axis = -1)

# %%
def array_to_spiketrains(array, bin_size):
    """Convert B x T x N spiking array to list of list of SpikeTrains"""
    stList = []

    for trial in range(array.shape[0]):
        trialList = []
        for channel in range(array.shape[2]):
            times = np.nonzero(array[trial, :, channel])[0]
            counts = array[trial, times, channel].astype(int)
            times = np.repeat(times, counts)
            st = neo.SpikeTrain(times*bin_size, t_stop=array.shape[1]*bin_size)
            trialList.append(st)
        stList.append(trialList)
    return stList

# %%
def GPFA_trajectories(dt, data):
    Y_st_train = array_to_spiketrains(data, dt*pq.ms)
    gpfa_3dim = GPFA(bin_size=dt * pq.ms, x_dim=3)
    trajectories = gpfa_3dim.fit_transform(Y_st_train)

    t = np.linspace(0, trajectories.shape[0], (trajectories.shape[0]+1))
    norm = plt.Normalize(t.min(), t.max())
    return trajectories, norm, t

# %%
def GPFA_2d(dt, data):
    d_train = array_to_spiketrains(data, dt*pq.ms)
    gpfa = GPFA(bin_size=(dt * pq.ms), x_dim=2)
    gpfa_val_result = gpfa.fit_transform(d_train, returned_data=['latent_variable', 'VsmGP'])
    length_scales = gpfa.params_estimated['gamma']
    X_hat_GPFA = rearrange(np.stack(gpfa_val_result['latent_variable'], 0), 'trials lat time -> trials time lat')
    P_hat_GPFA = rearrange(np.stack(gpfa_val_result['VsmGP'], 0)[:, np.arange(X_hat_GPFA.shape[1]), np.arange(X_hat_GPFA.shape[1])], 'trials time lat -> trials time lat')
    return X_hat_GPFA
# %%
N_trajectories, N_norm, N_t = GPFA_trajectories(dt = 20, data = N_array_bins)
print('North 3d done!')
S_trajectories, S_norm, S_t = GPFA_trajectories(dt = 20, data = S_array_bins)
print('South 3d done!')
C_trajectories, C_norm, C_t = GPFA_trajectories(dt = 20, data = C_array_bins)
print('Cumulative 3d done!')

# %%
N_X_hat = GPFA_2d(dt = 20, data = N_array_bins)
print('North 2d done!')
S_X_hat = GPFA_2d(20, S_array_bins)
print('South 2d done!')
C_X_hat = GPFA_2d(20, C_array_bins)
print('Cumulative 2d done!')

# %%
def plot_trajectories(trajectory, norm, t, traj):
    %matplotlib inline

    if traj == 'North':
        colors = cm.Reds(norm(t))
    if traj == 'South':
        colors = cm.Blues(norm(t))
    if traj == 'Cumulative':
        colors = cm.Purples(norm(t))

    f = plt.figure(figsize=(15, 5))
    ax1 = f.add_subplot(1, 2, 1, projection='3d')

    for i, single_trial_trajectory in enumerate(trajectory): # type: ignore
        ax1.plot(single_trial_trajectory[0], single_trial_trajectory[1], single_trial_trajectory[2], alpha= 0.3, color = colors[i])
    plt.figure()
    # trial averaged trajectory
    average_trajectory = np.mean(trajectory, axis=0) # type: ignore
    ax1.plot(average_trajectory[0], average_trajectory[1], average_trajectory[2], label='Trial averaged trajectory', color = 'k')
    ax1.legend()
    ax1.set_title(f'{traj}')

def plot_2d_traj(X_hat_GPFA, trial_num, traj):
    if traj == 'North':
        colors = cm.Reds(N_norm(N_t))
    if traj == 'South':
        colors = cm.Blues(S_norm(S_t))
    if traj == 'Cumulative':
        colors = cm.Purples(C_norm(C_t))

    for k in range(trial_num):
        plt.plot(X_hat_GPFA[k,:,0], X_hat_GPFA[k,:,1], color = colors[k + 10])

    
    # plt.figure()
# %%
plot_trajectories(N_trajectories, N_norm, N_t, 'North')
plot_trajectories(S_trajectories, S_norm, S_t, 'South')
plot_trajectories(C_trajectories, C_norm, C_t, 'Cumulative')
# %%

plot_2d_traj(N_X_hat, 100, 'North')
plot_2d_traj(S_X_hat, 100, 'South')
plot_2d_traj(C_X_hat, 100, 'Cumulative')

# %%
