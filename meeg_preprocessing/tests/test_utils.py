# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np
from nose.tools import assert_equal

import mne
from meeg_preprocessing.utils import get_data_picks


def test_get_data_picks():
    """Test creating pick_lists"""

    rng = np.random.RandomState(909)

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['grad', 'mag', 'mag', 'eeg']
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(test_data, info)

    pick_list = get_data_picks(raw)
    assert_equal(len(pick_list), 3)
    assert_equal(pick_list[0][1], 'mag')
    pick_list2 = get_data_picks(raw, meg_combined=False)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][1], 'mag')

    pick_list2 = get_data_picks(raw, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2) + 1)
    assert_equal(pick_list2[0][1], 'meg')

    test_data = rng.random_sample((4, 2000))
    ch_names = ['MEG %03d' % i for i in [1, 2, 3, 4]]
    ch_types = ['mag', 'mag', 'mag', 'mag']
    sfreq = 250.0
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)
    raw = mne.io.RawArray(test_data, info)

    pick_list = get_data_picks(raw)
    assert_equal(len(pick_list), 1)
    assert_equal(pick_list[0][1], 'mag')
    pick_list2 = get_data_picks(raw, meg_combined=True)
    assert_equal(len(pick_list), len(pick_list2))
    assert_equal(pick_list2[0][1], 'mag')

if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__name__)
