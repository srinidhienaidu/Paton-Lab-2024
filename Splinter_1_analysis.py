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
from mpl_toolkits.mplot3d import Axes3D
# %%

# base_dir = r'C:\Users\srini\Documents\GitHub\Paton-Lab--Summer-2024\Splinter' # Champalimaud PC dir

base_dir = r'/Users/srinidhienaidu/Downloads/GitHub/Paton-Lab--Summer-2024/Splinter' # Srinidhi Mac dir
session = 0 # Change based on which trial you want
date = sorted(os.listdir(base_dir))[session + 1]
path = os.path.join(base_dir, date)
os.chdir(path)
print(path)

log_df = pd.read_csv('LogDf.csv')
session_df = pd.read_csv('SessionDf.csv')

log_data = log_df.to_dict(orient= 'dict')
session_data = session_df.to_dict(orient= 'dict')

spike_times = np.load('spiketimes.npy', allow_pickle=True)
NUM_NEURONS = spike_times.shape[0]
NUM_STIMS = 8
# %%
print('Date: ', sorted(os.listdir(base_dir))[session + 1])
print('Number Neurons: ', NUM_NEURONS)
print('Log Keys: ', log_data.keys())
print('Session Keys: ', session_data.keys())
# %%
def set_spines_invisible(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def set_grid_invisible(ax):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
    ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)

def common_label(fig, xlabel, ylabel):
    """Put a common `xlabel` and `ylabel` on the figure `fig`.
    
    Args:
        - fig (plt.figure)
        - xlabel (str)
        - ylabel (str)
    """
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
# %%
def remove_nans(arr):
    return [x for x in arr if not math.isnan(x)]

def nan_idx(data):
    nan_idx = []
    for index in data:
        if isinstance(data[index], float):
            nan_idx.append(index)
    return nan_idx

def list_to_sorted_arr(og_list, list_to_sort_by):
    arr = np.array(og_list)
    return arr[list_to_sort_by]

for cell in range(NUM_NEURONS):
    spike_times[cell] = remove_nans(spike_times[cell])  

# %%
stim_onset_t_pre = []
stim_offset_t = []

for index in range(len(log_data['t'])):

    time = log_data['t'][index]
    name = log_data['name'][index]

    if name == 'STIM_ONSET_EVENT':
        stim_onset_t_pre.append(time)
    elif name == 'STIM_OFFSET_EVENT':
        stim_offset_t.append(time)  
    
NUM_TRIALS = len(stim_offset_t)
trial_num = np.arange(0, len(stim_onset_t_pre), 1)
# %%
choice = session_data['chosen_side']
no_choice_idx = nan_idx(choice)

stim_onset_t = [value for index, value in enumerate(stim_onset_t_pre) if index not in no_choice_idx]
og_trial_num = [value for index, value in enumerate(trial_num) if index not in no_choice_idx]

general_trial_num = np.arange(0, len(stim_onset_t), 1)

int_durations = np.zeros((NUM_TRIALS))
choice_rt = []

for idx, val in enumerate(og_trial_num):

    int_durations[idx] = (session_data['this_interval'][val])
    choice_rt.append(session_data['choice_rt'][val])


sort_by_dur_idx = np.argsort(int_durations)

sorted_int_durations = int_durations[sort_by_dur_idx]
new_sorted_trials = general_trial_num[sort_by_dur_idx]

dur_vals = [600, 1050, 1260, 1380, 1620, 1740, 1950, 2400]

# stim_offset_t = list_to_sorted_arr(stim_offset_t_pre, sort_by_dur_idx)

# %%
n_sorted_init = []
n_init_dur = []

s_sorted_init = []
s_init_dur = []

init_port = session_data['init_port']

for dur, val in enumerate(new_sorted_trials):
    if init_port[val] == 'N':
        n_sorted_init.append(val)
        n_init_dur.append(sorted_int_durations[dur])

    elif init_port[val] == 'S':
        s_sorted_init.append(val)
        s_init_dur.append(sorted_int_durations[dur])

n_sorted_init = np.array(n_sorted_init)
s_sorted_init = np.array(s_sorted_init)
# %%
def plot_colors(cmap):
    cmap = plt.get_cmap(cmap)
    points = np.linspace(0, 1, NUM_STIMS + 1)
    colors = []
    for stim in range(NUM_STIMS):
        colors.append(cmap(points[stim + 1]))
    return colors

def counts_and_cumulative(to_count):
    int_counts = Counter(to_count)
    counts = list(int_counts.values())

    cumulative_sum_list = [0]
    running_total = 0
    for value in counts:
        running_total += value
        cumulative_sum_list.append(running_total)

    return counts, cumulative_sum_list

def bin_values_by_range(values, start_values, end_values):
    bins = []
    for start, end in zip(start_values, end_values):
        mask = (values >= start) & (values < end)
        bins.append(len(values[mask]))
        bin_array = np.array(bins)
    return bin_array

def sum_over_trial_bins(stacked_bin_array, cumulative_sum_list):
    for_trial_sums = []
    raw_bins = []
    for stim_dur in range(NUM_STIMS):
        bin_start = cumulative_sum_list[stim_dur]
        bin_end = cumulative_sum_list[stim_dur + 1] 
        summed_counts = (stacked_bin_array[:][bin_start:bin_end])
        raw_bins.append(summed_counts)
        normalized_counts = summed_counts / BIN_SIZE * 1000
        mean_counts = np.mean(normalized_counts, axis = 0)
        for_trial_sums.append(mean_counts)

    sum_over_trial_bins = np.vstack(for_trial_sums)
    raw_bin_arr = np.vstack(raw_bins)
    return raw_bin_arr, sum_over_trial_bins


# %%

# %%
counts, cumulative_sum_list = counts_and_cumulative(sorted_int_durations)
n_counts, n_cumulative_sum = counts_and_cumulative(n_init_dur)
s_counts, s_cumulative_sum = counts_and_cumulative(s_init_dur)

BIN_SIZE = 100
window = [-1000, 8000]
bins = np.arange(window[0], window[1], BIN_SIZE) 
num_neurons = NUM_NEURONS
# %%
def rasters_and_psths(color, data, data_points, num_neurons, window, BIN_SIZE, new_sorted_trials, cumulative_sum_list):
    COND_NUM_TRIALS = len(new_sorted_trials)
    psth_colors = plot_colors(cmap = color)
    cmap = plt.get_cmap(color)
    norm = mcolors.Normalize(vmin=600, vmax=2400)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    full_spikes_neuron = []
    full_trial_nums = []
    full_summed_over_bins = []
    full_raw_bins = []
    full_cond_spike_times = []
    for neuron in range(num_neurons):


        bins = np.arange(window[0], window[1], BIN_SIZE)

        all_spikes_neuron = []
        all_trial_nums = []
        
        binned_counts_array = []
        for_cond_spike_times = []

        
        colored_time_points = np.array(list_to_sorted_arr(stim_offset_t, sort_by_dur_idx) + list_to_sorted_arr(data_points, sort_by_dur_idx) - list_to_sorted_arr(data, sort_by_dur_idx))
        
        for num, val in enumerate(new_sorted_trials):
            # For Raster
            aligned_spikes = np.array(spike_times[neuron]) - data[val]
            trial_index_spikes = np.intersect1d(np.where(aligned_spikes <= window[1])[0], np.where(aligned_spikes > window[0])[0])
            spike_times_in_window = aligned_spikes[trial_index_spikes]
            all_spikes_neuron.extend(spike_times_in_window)
            n_spikes = len(trial_index_spikes)
            all_trial_nums.extend([num] * n_spikes)

            # For PSTH
            binned_counts = bin_values_by_range(values= spike_times_in_window, start_values= bins, end_values= bins + BIN_SIZE)
            binned_counts_array.append(np.array(binned_counts))
            for_cond_spike_times.append(colored_time_points[num])
            # raw_spike_times.append(chunk_spike_times[idx])

        full_spikes_neuron.append(all_spikes_neuron)
        full_trial_nums.append(all_trial_nums)

        stacked_bin_array = np.vstack(binned_counts_array)
        raw_bin, norm_sum_bins = sum_over_trial_bins(stacked_bin_array, cumulative_sum_list)
        full_raw_bins.append(raw_bin)
        full_summed_over_bins.append(norm_sum_bins)

        full_cond_spike_times.append(for_cond_spike_times)
        

    return full_raw_bins, COND_NUM_TRIALS, psth_colors, full_spikes_neuron, full_trial_nums, full_summed_over_bins, full_cond_spike_times

# %%
_, N_NUM_TRIALS, n_onset_psth_colors, n_onset_spikes, n_onset_trials, n_onset_summed, n_onset_colored_points = rasters_and_psths(color= 'Reds', data= stim_onset_t, data_points= choice_rt, num_neurons= NUM_NEURONS, window= window, BIN_SIZE= BIN_SIZE, new_sorted_trials= n_sorted_init, cumulative_sum_list=n_cumulative_sum)
_, N_NUM_TRIALS, n_offset_psth_colors, n_offset_spikes, n_offset_trials, n_offset_summed, n_offset_colored_points = rasters_and_psths(color= 'Reds', data= stim_offset_t, data_points= choice_rt, num_neurons= NUM_NEURONS, window= window, BIN_SIZE= BIN_SIZE, new_sorted_trials= n_sorted_init, cumulative_sum_list= n_cumulative_sum)
_, S_NUM_TRIALS, s_onset_psth_colors, s_onset_spikes, s_onset_trials, s_onset_summed, s_onset_colored_points = rasters_and_psths(color= 'Blues', data= stim_onset_t, data_points= choice_rt, num_neurons= NUM_NEURONS, window= window, BIN_SIZE= BIN_SIZE, new_sorted_trials= s_sorted_init, cumulative_sum_list= s_cumulative_sum)
_, S_NUM_TRIALS, s_offset_psth_colors, s_offset_spikes, s_offset_trials, s_offset_summed, s_offset_colored_points = rasters_and_psths(color= 'Blues', data= stim_offset_t, data_points= choice_rt, num_neurons= NUM_NEURONS, window= window, BIN_SIZE= BIN_SIZE, new_sorted_trials= s_sorted_init, cumulative_sum_list= s_cumulative_sum)
# %%
cmap1 = plt.get_cmap('Reds')
norm1 = mcolors.Normalize(vmin=600, vmax=2400)
sm1 = plt.cm.ScalarMappable(cmap=cmap1, norm=norm1)
sm1.set_array([])

cmap2 = plt.get_cmap('Blues')
norm2 = mcolors.Normalize(vmin=0, vmax=100)  
sm2 = plt.cm.ScalarMappable(cmap=cmap2, norm=norm2)
sm2.set_array([])

for neuron in range(num_neurons):
    fig, axs = plt.subplots(4, 2, figsize=(16, 12))
    fig.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.1, hspace=0.4, wspace=0.4)

    cbar_ax1 = fig.add_axes([0.1, 0.1, 0.02, 0.8])  
    cbar1 = plt.colorbar(sm1, cax=cbar_ax1, orientation='vertical')
    cbar1.ax.invert_yaxis()
    cbar1.ax.set_yticks([])

    cbar_ax2 = fig.add_axes([0.15, 0.1, 0.02, 0.8]) 
    cbar2 = plt.colorbar(sm2, cax=cbar_ax2, orientation='vertical')
    cbar2.set_label('Trial #')
    cbar2.ax.invert_yaxis()

    # North Rasters- Onset
    axs[0, 0].scatter(n_onset_spikes[neuron], n_onset_trials[neuron], s = 0.01, color = 'k')
    axs[0, 0].scatter(n_onset_colored_points[neuron], range(N_NUM_TRIALS), s = 0.3, color = 'r')
    axs[0, 0].set_ylabel("Trial #")
    axs[0, 0].set_title(f"Aligned to Stimulus Onset")
    axs[0, 0].axvline(x=0, color="k", ls="--")
    axs[0, 0].set_xlim(window[0], window[1])  

    # North Rasters- Offset
    axs[0, 1].scatter(n_offset_spikes[neuron], n_offset_trials[neuron], s = 0.01, color = 'k')
    axs[0, 1].scatter(n_offset_colored_points[neuron], range(N_NUM_TRIALS), s = 0.3, color = 'r')
    axs[0, 1].set_title(f"Aligned to Stimulus Offset")
    axs[0, 1].axvline(x=0, color="k", ls="--")
    axs[0, 1].set_xlim(window[0], window[1])

    # South Rasters- Onset
    axs[1, 0].scatter(s_onset_spikes[neuron], s_onset_trials[neuron], s = 0.01, color = 'k')
    axs[1, 0].scatter(s_onset_colored_points[neuron], range(S_NUM_TRIALS), s = 0.3, color = 'b')
    axs[1, 0].set_ylabel("Trial #")
    axs[1, 0].axvline(x=0, color="k", ls="--")
    axs[1, 0].set_xlim(window[0], window[1])  

    # South Rasters- Offset
    axs[1, 1].scatter(s_offset_spikes[neuron], s_offset_trials[neuron], s = 0.01, color = 'k')
    axs[1, 1].scatter(s_offset_colored_points[neuron], range(S_NUM_TRIALS), s = 0.3, color = 'b')
    axs[1, 1].axvline(x=0, color="k", ls="--")
    axs[1, 1].set_xlim(window[0], window[1])

    for stim in range(NUM_STIMS):

        # North PSTHs- Onset
        axs[2, 0].plot(bins, n_onset_summed[neuron][stim], color = n_onset_psth_colors[stim], linewidth = 0.9)
        axs[2, 0].axvline(x=0, color="k", ls="--")
        axs[2, 0].set_xlim(window[0], window[1])
        axs[2, 0].set_ylabel('Firing Rate (Hz)')


        # North PSTHs- Offset
        axs[2, 1].plot(bins, n_offset_summed[neuron][stim], color = n_offset_psth_colors[stim], linewidth = 0.9)
        axs[2, 1].axvline(x=0, color="k", ls="--")
        axs[2, 1].set_xlim(window[0], window[1])

        # South PSTHs- Onset
        axs[3, 0].plot(bins, s_onset_summed[neuron][stim], color = s_onset_psth_colors[stim], linewidth = 0.9)
        axs[3, 0].axvline(x=0, color="k", ls="--")
        axs[3, 0].set_xlim(window[0], window[1])
        axs[3, 0].set_ylabel('Firing Rate (Hz)')
        # South PSTHs- Offset
        axs[3, 1].plot(bins, s_offset_summed[neuron][stim], color = s_offset_psth_colors[stim], linewidth = 0.9)
        axs[3, 1].axvline(x=0, color="k", ls="--")
        axs[3, 1].set_xlim(window[0], window[1])

    fig.suptitle(f'Neuron {neuron}', fontsize = 25)

# %%

reshaped_bins = []

for neuron in range(NUM_NEURONS):
    og_binned_counts = s_onset_summed[neuron]
    len_spike_train = s_onset_summed[neuron].shape[0] * s_onset_summed[neuron].shape[1]
    reshaped_bins.append(np.reshape(og_binned_counts, (1, len_spike_train)))

X_data = np.vstack(reshaped_bins)
mean_vec = np.mean(X_data, axis=0)
centered_data = X_data - mean_vec

cov_matrix = np.cov(centered_data)

real_evals, real_evecs = np.linalg.eig(cov_matrix)

sqrt_eval = np.sqrt(real_evals)

sorted_indices = np.argsort(sqrt_eval)[::-1]
sorted_eigenvecs = real_evecs[:, sorted_indices]
sqrt_eigvals = sqrt_eval[sorted_indices]

PC_numbers = np.arange(1,num_neurons+1)

fig, ax = plt.subplots(1, 1, figsize=(6, 4))
ax.plot(PC_numbers, sqrt_eigvals, c="k", marker="o", mfc="r", mec="k")
ax.set_ylabel(r"$\sqrt{\lambda_i}$")  
ax.set_xlabel("Principal component number, i")
plt.show()

# %%

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

colors = plt.cm.Reds(np.linspace(0, 1, NUM_STIMS))
stims = np.arange(0, (720 + 9), 90)

for i in np.arange((len(stims)) - 1):
    start_idx = s_cumulative_sum[i]
    end_idx = s_cumulative_sum[i + 1]
    dot_data = centered_data.T[start_idx:end_idx]
    eig_vec = sorted_eigenvecs[:, :3]
    # print(dot_data.shape, eig_vec.shape)
    data_3d = np.dot(dot_data, eig_vec)

    ax.scatter(data_3d[:, 0], data_3d[:, 1], data_3d[:, 2], alpha=0.7, c = [colors[i]], label=f'Reach Angle {i + 1}')

ax.set_xlabel("PCA 1")
ax.set_ylabel("PCA 2")
ax.set_zlabel("PCA 3")
set_grid_invisible(ax)

# ax.view_init(elev=20, azim=240)  

plt.show()

# %%

conditions = [600, 1050, 1260, 1380, 1620, 1740, 1950, 2400]

# def summed_bins_across_neurons(condition):
#     all = []
#     if condition == 'north':
#         sorted_trials = n_sorted_init
#         cum_sum_list = n_cumulative_sum
#     elif condition == 'south':
#         sorted_trials = s_sorted_init
#         cum_sum_list = s_cumulative_sum
#     elif condition == 'invariant':
sorted_trials = new_sorted_trials
cum_sum_list = cumulative_sum_list

# for idx, time in enumerate(conditions):
window = [0, 600]
raw_binned, _, _, _, _, _, _ = rasters_and_psths(color= 'Blues', data= stim_onset_t, data_points= choice_rt, num_neurons= NUM_NEURONS, window= window, BIN_SIZE= 10, new_sorted_trials= sorted_trials, cumulative_sum_list= cum_sum_list)
summed = np.sum(raw_binned, axis = 1)
# all.append(summed)

final_array_stacked = np.hstack(all)
    # print(condition, ' ', time, 'ms complete')

# return final_array_stacked
    
raw_binned




# %%
n_sum_across_trials = summed_bins_across_neurons('north')
s_sum_across_trials = summed_bins_across_neurons('south')
invariant_sum_across_trials = summed_bins_across_neurons('invariant')
# %%
cond_split = np.hstack((n_sum_across_trials, s_sum_across_trials))
cond_ind = np.hstack((invariant_sum_across_trials, invariant_sum_across_trials))
cond_dep  = (cond_split-cond_ind) * -1
# %%
fig, axs = plt.subplots(2, 2, figsize=(12, 12))
axs[0, 0].imshow(n_sum_across_trials, aspect='auto', cmap='Reds');
axs[0, 0].set_xlabel('Bin');
axs[0, 0].set_ylabel('Neuron');
axs[0, 0].set_title('North (Condition Split) Stimulus');

axs[0, 1].imshow(s_sum_across_trials, aspect='auto', cmap='Blues');
axs[0, 1].set_xlabel('Bin');
axs[0, 1].set_ylabel('Neuron');
axs[0, 1].set_title('South (Condition Split) Stimulus');

axs[1, 0].imshow(invariant_sum_across_trials, aspect='auto', cmap='Grays');
axs[1, 0].set_xlabel('Bin');
axs[1, 0].set_ylabel('Neuron');
axs[1, 0].set_title('Condition Independent Average');

axs[1, 1].imshow(cond_dep, aspect='auto', cmap='Greens');
axs[1, 1].set_xlabel('Bin');
axs[1, 1].set_ylabel('Neuron');
axs[1, 1].set_title('Condition Dependent Average');

# %%

