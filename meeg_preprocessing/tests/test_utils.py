# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np
from nose.tools import assert_equal

import mne

import os
import os.path as op
import subprocess
import json
import warnings

from nose.tools import (assert_true, assert_equals, assert_not_equals,
                        assert_raises)

from mne.utils import _TempDir
from mne import io
from mne import read_events, pick_types
from mne.io.constants import FIFF

from meeg_preprocessing.utils import (
    _get_git_head, get_versions, setup_provenance, set_eog_ecg_channels,
    get_data_picks)

warnings.simplefilter('always')  # enable b/c these tests throw warnings

base_dir = op.join(op.dirname(__file__), 'data')
raw_fname = op.join(base_dir, 'test_raw.fif')
event_name = op.join(base_dir, 'test-eve.fif')
evoked_nf_name = op.join(base_dir, 'test-nf-ave.fif')


def _get_data():
    raw = io.Raw(raw_fname, add_eeg_ref=False)
    events = read_events(event_name)
    picks = pick_types(
        raw.info, meg=True, eeg=True, stim=True, ecg=True, eog=True,
        include=['STI 014'], exclude='bads')
    return raw, events, picks


def test_get_git_hash():
    """Test reading git hashes"""

    assert_raises(ValueError, _get_git_head, 1e15)
    assert_raises(ValueError, _get_git_head, 'foofoo')

    def my_call(cmd):
        process = subprocess.Popen(cmd,
                                   stdout=subprocess.PIPE,
                                   shell=True)
        out = process.communicate()[0].strip()
        del process
        return out

    tmp_dir = _TempDir()
    command = 'cd {}; git init'.format(tmp_dir)
    my_call(command)

    get_head = 'cd {gitpath}; git rev-parse --verify HEAD'.format(
        gitpath=tmp_dir
    )

    git_out = my_call('cd {}; '
                      'echo 123 >> tmp.txt; '
                      'git add tmp.txt; '
                      'git commit -am "blub"'.format(tmp_dir))

    assert_true('1 file changed' in git_out)
    assert_true('1 insertion' in git_out)
    assert_true('create mode 100644 tmp.txt' in git_out)
    assert_true('tmp.txt' in os.listdir(tmp_dir))

    head1 = _get_git_head(tmp_dir)
    head2 = my_call(get_head)
    assert_equals(head1, head2)

    git_out = my_call('cd {};'
                      'echo -n 123 >> tmp.txt; '
                      'git add tmp.txt; '
                      'git commit -am "blub2"'.format(tmp_dir))

    head3 = _get_git_head(tmp_dir)
    head4 = my_call(get_head)

    assert_equals(head3, head4)
    assert_not_equals(head1, head3)
    assert_true(isinstance(int(head3, 16), long))


def test_get_version():
    """Test version checks"""
    import sys
    from meeg_preprocessing import __path__ as cpath
    from meeg_preprocessing import __version__ as cversion

    cpath = cpath[0]
    name = 'meeg_preprocessing'
    versions = get_versions(sys)
    cpype_version_git = _get_git_head(cpath)
    assert_true(cpype_version_git in versions[name])
    assert_true(cversion in versions[name])

    import meeg_preprocessing
    meeg_preprocessing.__version__ = '0.2.git'
    versions = get_versions(sys)
    assert_true('0.2.git' in versions[name])


def test_setup_provenance():
    """Test provenance tracking"""

    for config_opt in ['abs_py', 'default', 'other']:
        tmp_dir = _TempDir()
        if config_opt == 'default':
            config_fname = op.join(op.dirname(__file__), 'config.py')
            config_content = 'import this'
            config_param = None
        elif config_opt == 'abs_py':
            config_fname = op.join(tmp_dir, 'config.py')
            config_content = 'import antigravity'
            config_param = config_fname
        elif config_opt == 'other':
            config_fname = op.join(tmp_dir, 'config.txt')
            config_content = 'my_config :: 42'
            config_param = config_fname
        with open(config_fname, 'w') as fid_config:
            fid_config.write(config_content)

        report, run_id, results_dir, logger = setup_provenance(
            script=__file__, results_dir=tmp_dir, config=config_param)

        logging_dir = op.join(results_dir, run_id)
        assert_true(op.isdir(logging_dir))
        assert_true(op.isfile(op.join(logging_dir, 'run_time.json')))
        assert_true(op.isfile(op.join(logging_dir, 'run_output.log')))
        assert_true(op.isfile(op.join(logging_dir, 'script.py')))

        config_basename = op.split(config_fname)[-1]
        with open(op.join(results_dir, run_id, config_basename)) as config_fid:
            config_code = config_fid.read()
        assert_equal(config_code, config_content)

        with open(__file__) as fid:
            this_file_code = fid.read()

        with open(op.join(results_dir, run_id, 'script.py')) as fid:
            other_file_code = fid.read()

        assert_equals(this_file_code, other_file_code)
        with open(op.join(results_dir, run_id, 'run_time.json')) as fid:
            modules = json.load(fid)
            assert_true('meeg_preprocessing' in modules)

        assert_equals(report.title, op.splitext(op.split(__file__)[1])[0])
        assert_equals(report.data_path, logging_dir)
        if config_opt == 'default':
            os.remove(config_fname)


def test_set_eog_ecg_channels():
    """Test set eeg channels"""
    raw, _, _ = _get_data()
    set_eog_ecg_channels(raw, 'EEG 001', 'EEG 002')
    eog_idx, ecg_idx = [raw.ch_names.index(k) for k in ['EEG 001', 'EEG 002']]
    assert_equals(raw.info['chs'][eog_idx]['kind'], FIFF.FIFFV_EOG_CH)
    assert_equals(raw.info['chs'][ecg_idx]['kind'], FIFF.FIFFV_ECG_CH)


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
