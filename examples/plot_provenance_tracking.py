# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

# Note. This will create directories.

import os.path as op
from meeg_preprocessing import check_apply_filter
from meeg_preprocessing.utils import setup_provenance

from mne import io

from config import (
    subject,
    fname,
    filter_params,
    notch_filter_params,
    plot_fmin,
    plot_fmax,
    n_jobs,
)

report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir='results')

raw = io.Raw(fname, preload=True)

fig, report = check_apply_filter(
    raw, subject=subject, filter_params=filter_params,
    notch_filter_params=notch_filter_params,
    plot_fmin=plot_fmin, plot_fmax=plot_fmax,
    n_jobs=n_jobs)

report.save(op.join(results_dir, run_id, 'report.html'))
