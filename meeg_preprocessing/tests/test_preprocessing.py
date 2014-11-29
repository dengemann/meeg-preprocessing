# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op

import numpy as np

import matplotlib as mpl
mpl.use('Agg')

from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal
import mne
from meeg_preprocessing.preprocessing import (check_apply_filter, compute_ica,
                                              _prepare_filter_plot)

test_raw_fname = op.join(op.dirname(__file__), 'data', 'test_raw.fif')

raw = mne.io.Raw(test_raw_fname, preload=True)

rng = np.random.RandomState(909)


def test_check_apply_filter():
    """Test filtering"""
    import matplotlib as mpl
    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['grad', 'mag', 'mag', 'eeg']
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    test_raw1 = mne.io.RawArray(test_data, info)
    test_raw2 = test_raw1.copy()
    test_raw2.pick_channels(['MEG 001'])

    expected_picks_lists = [
        [(np.array([1, 2]), 'mag'),
         (np.array([0]), 'grad'),
         (np.array([3]), 'eeg')],
        [(np.array([0]), 'grad')]
    ]

    test_raws = [
        test_raw1,
        test_raw2
    ]
    test_nrows = (3, 1)
    test_axes = (np.ndarray, list)
    iter_tests = zip(expected_picks_lists, test_raws, test_nrows, test_axes)

    for expected_picks_list, test_raw, this_test_nrows, this_test_axes in \
            iter_tests:

        picks_list, n_rows, fig, axes = _prepare_filter_plot(test_raw, None)

        assert_equal(n_rows, this_test_nrows)
        assert_true(isinstance(fig, mpl.figure.Figure))
        assert_true(isinstance(axes, this_test_axes))
        for (picks1, ch1), (picks2, ch2) in zip(expected_picks_list,
                                                picks_list):
            assert_equal(ch1, ch2)
            assert_array_equal(picks1, picks2)

    # test filtering
    raw2 = raw.pick_channels(['MEG 1113'], copy=True)
    lp_before = raw2.info['lowpass']

    filter_params = dict(l_freq=0.5, h_freq=20, n_jobs=1,
                         method='fft', l_trans_bandwidth=0.1,
                         h_trans_bandwidth=0.5)

    fig, report = check_apply_filter(raw2, 'test-subject',
                                     filter_params=filter_params)
    lp_after = raw2.info['lowpass']

    assert_true(isinstance(fig, mpl.figure.Figure))
    assert_equal(report.sections, ['FILTER'])
    assert_equal(len(report.html), 1)
    assert_true(lp_after < lp_before)

    fig2, _ = check_apply_filter(raw2, 'test-subject', figsize=(12, 12))
    assert_true(np.any(fig.get_size_inches() != fig2.get_size_inches()))


def test_preprocessing_ica():
    """Test ICA preprocessing"""
    n_max_ecg = 1
    n_max_eog = 0
    n_plots = (5 if n_max_ecg > 0 else 0) + (5 if n_max_eog > 0 else 0)
    n_plots += (1 if n_plots != 0 else 0)
    ica, report = compute_ica(raw, n_components=4, picks=[0, 1, 2, 3, 5],
                              subject='test-subject', decim=2,
                              n_max_ecg=n_max_ecg, n_max_eog=n_max_eog)
    assert_equal(len(ica.exclude), n_max_ecg)
    assert_equal(report.sections, ['MAG+GRAD ECG', 'MAG+GRAD RAW'])
    assert_equal(len(report.html), n_plots)
    picks = np.array([0, 1, 2, 3, 5])
    rank = raw.estimate_rank(picks=picks)
    ica, report = compute_ica(raw, n_components='rank', picks=picks,
                              subject='test-subject', decim=2,
                              n_max_ecg=n_max_ecg, n_max_eog=n_max_eog)
    assert_equal(ica.n_components_, rank)
if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
