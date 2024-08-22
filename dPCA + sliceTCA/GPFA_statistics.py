# %%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from quantities import ms, s, Hz
from elephant.spike_train_generation import homogeneous_poisson_process, homogeneous_gamma_process
from elephant.statistics import time_histogram, instantaneous_rate
from elephant.statistics import mean_firing_rate
from elephant.kernels import GaussianKernel
from elephant.statistics import isi, cv

# %%
t_start = 275.5 * ms
t_start2 = 3. * s
t_start_sum = t_start + t_start2
# %%
np.random.seed(28)
spiketrain1 = homogeneous_poisson_process(rate = 10 * Hz, t_start = 0. * ms, t_stop= 10000. * ms)
spiketrain2 = homogeneous_gamma_process(a = 3, b = 10*Hz, t_start = 0. * ms, t_stop= 10000. * ms)
print('spiketrain1 type is', type(spiketrain1))
print('spiketrain2 type is', type(spiketrain2))

print(f"spiketrain2 has {len(spiketrain2)} spikes:")
print("  t_start:", spiketrain2.t_start)
print("  t_stop:", spiketrain2.t_stop)
print("  spike times:", spiketrain2.times)

# %%
plt.figure(figsize=(8, 3))
plt.eventplot([spiketrain1.magnitude, spiketrain2.magnitude], linelengths=0.75, color='black')
plt.xlabel('Time (ms)', fontsize=16)
plt.yticks([0,1], labels=["spiketrain1", "spiketrain2"], fontsize=16)
plt.title("Figure 1");
# %%
print("The mean firing rate of spiketrain1 is", mean_firing_rate(spiketrain1))
print("The mean firing rate of spiketrain2 is", mean_firing_rate(spiketrain2))
# %%
fr1 = len(spiketrain1) / (spiketrain1.t_stop - spiketrain1.t_start)
fr2 = len(spiketrain2) / (spiketrain2.t_stop - spiketrain2.t_start)
print("The mean firing rate of spiketrain1 is", fr1)
print("The mean firing rate of spiketrain2 is", fr2)
# %%
mean_firing_rate(spiketrain1, t_start=0*ms, t_stop=1000*ms)

# Mean FR can even be represented in multidimensional arrays when they contain the same
# number of spikes! 
multi_spiketrains = np.array([[1,2,3],[4,5,6],[7,8,9]])*ms
mean_firing_rate(multi_spiketrains, axis=0, t_start=0*ms, t_stop=5*ms)
# %%
histogram_count = time_histogram([spiketrain1], 500*ms)
print(type(histogram_count), f"of shape {histogram_count.shape}: {histogram_count.shape[0]} samples, {histogram_count.shape[1]} channel")
print('sampling rate:', histogram_count.sampling_rate)
print('times:', histogram_count.times)
print('counts:', histogram_count.T[0])
# %%
histogram_rate = time_histogram([spiketrain1], 500*ms, output='rate')
print('times:', histogram_rate.times)
print('rate:', histogram_rate.T[0])
# %%
inst_rate = instantaneous_rate(spiketrain1, sampling_period=50*ms)
print(type(inst_rate), f"of shape {inst_rate.shape}: {inst_rate.shape[0]} samples, {inst_rate.shape[1]} channel")
print('sampling rate:', inst_rate.sampling_rate)
print('times (first 10 samples): ', inst_rate.times[:10])
print('instantaneous rate (first 10 samples):', inst_rate.T[0, :10])
# %%
instantaneous_rate(spiketrain1, sampling_period=20*ms, kernel=GaussianKernel(200*ms))
# %%
plt.figure(dpi=150)

# plotting the original spiketrain
plt.plot(spiketrain1, [0]*len(spiketrain1), 'r', marker=2, ms=25, markeredgewidth=2, lw=0, label='poisson spike times')

# mean firing rate
plt.hlines(mean_firing_rate(spiketrain1), xmin=spiketrain1.t_start, xmax=spiketrain1.t_stop, linestyle='--', label='mean firing rate')

# time histogram
plt.bar(histogram_rate.times, histogram_rate.magnitude.flatten(), width=histogram_rate.sampling_period, align='edge', alpha=0.3, label='time histogram (rate)')

# instantaneous rate
plt.plot(inst_rate.times.rescale(ms), inst_rate.rescale(histogram_rate.dimensionality).magnitude.flatten(), label='instantaneous rate')

# axis labels and legend
plt.xlabel('time [{}]'.format(spiketrain1.times.dimensionality.latex))
plt.ylabel('firing rate [{}]'.format(histogram_rate.dimensionality.latex))
plt.xlim(spiketrain1.t_start, spiketrain1.t_stop)
plt.legend()
plt.show()
# %%
spiketrain_list = [homogeneous_poisson_process(rate=10.0*Hz, t_start=0.0*s, t_stop=100.0*s) for i in range(100)]

plt.figure(dpi=150)
plt.eventplot([st.magnitude for st in spiketrain_list], linelengths=0.75, linewidths=0.75, color='black')
plt.xlabel("Time, s")
plt.ylabel("Neuron id")
plt.xlim([0, 1]);
# %%
cv_list = [cv(isi(spiketrain)) for spiketrain in spiketrain_list]

# let's plot the histogram of CVs
plt.figure(dpi=100)
plt.hist(cv_list)
plt.xlabel('CV')
plt.ylabel('count')
plt.title("Coefficient of Variation of homogeneous Poisson process");

# %%
