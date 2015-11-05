###############################################################################
from mne.datasets import sample

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
subject = 'sample'


###############################################################################
# jobs and runtime performance
n_jobs = 8`

###############################################################################
# Filters

# can be a list for repeated filtering
filter_params = [dict(l_freq=1.0, h_freq=100, n_jobs=n_jobs, method='fft',
                      l_trans_bandwidth=0.1, h_trans_bandwidth=0.5)]

notch_filter_params = dict(freqs=(50, 100, 150,))

# margins for the PSD plot
plot_fmin = 0.0
plot_fmax = 120.0

###############################################################################
# ICA

n_components = 0.99
# comment out to select ICA components via rank (useful with SSSed data):
# n_components = 'rank'

ica_meg_combined = True  # esimtate combined MAG and GRADs
decim = 5  # decimation
n_max_ecg, n_max_eog = 3, 2  # limit components detected due to ECG / EOG
ica_reject = {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}

###############################################################################
# Report

img_scale = 1.5  # make big PNG images
