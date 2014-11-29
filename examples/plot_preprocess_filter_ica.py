"""
Preprocessing MEG and EEG data using filters and ICA
====================================================

This examples filters MEG and EEG data and subsequently computes separate
ICA solutions for MEG and EEG. An html report is created, which includes
diagnostic plots of the preprocessing steps.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from meeg_preprocessing import check_apply_filter, compute_ica
from meeg_preprocessing.utils import get_data_picks

from mne import io
from mne.datasets import sample

data_path = sample.data_path()
fname = data_path + '/MEG/sample/sample_audvis_raw.fif'
subject = 'sample'

raw = io.Raw(fname, preload=True)

include = []  # or stim channels ['STI 014']
raw.info['bads'] += ['EEG 053']  # bads + 1 more

################################################################################
# jobs and runtime performance
n_jobs = 1

################################################################################
# Filters

# can be a list for repeated filtering
filter_params = [dict(l_freq=1.0, h_freq=100, n_jobs=n_jobs, method='fft',
                      l_trans_bandwidth=0.1, h_trans_bandwidth=0.5)]

notch_filter_params = dict(freqs=(50, 100, 150,))

# margins for the PSD plot
plot_fmin = 0.0
plot_fmax = 120.0

################################################################################
# ICA

n_components = 0.99
# comment out to select ICA components via rank (useful with SSSed data):
# n_components = 'rank'

ica_meg_combined = True  # esimtate combined MAG and GRADs
decim = 5  # decimation
n_max_ecg, n_max_eog = 3, 2  # limit components detected due to ECG / EOG
ica_reject = {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}

################################################################################
# Report

img_scale = 1.5  # make big PNG images


fig, report = check_apply_filter(raw, subject=subject,
                                 filter_params=filter_params,
                                 notch_filter_params=notch_filter_params,
                                 plot_fmin=plot_fmin, plot_fmax=plot_fmax,
                                 n_jobs=n_jobs, img_scale=img_scale)


# get picks and iterate over channels
for picks, ch_type in get_data_picks(raw, meg_combined=ica_meg_combined):
    ica, _ = compute_ica(raw, picks=picks,
                         subject=subject, n_components=n_components,
                         n_max_ecg=n_max_ecg, n_max_eog=n_max_eog,
                         reject=ica_reject,
                         decim=decim, report=report, img_scale=img_scale)
    ica.save('{}-ica.fif'.format(ch_type))

report.save('preprocessing-report-{}.html'.format(subject),
            open_browser=True, overwrite=True)
