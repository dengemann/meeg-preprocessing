meeg-preprocessing
==================

Open source preprocessing tools for MEG/EEG based on [MNE-Python](https://github.com/dengemann/mne-python)

What's this?
============

I wrote these tools to preprocess MEG data recorded in Jülich (4D Magnes 3600WH) and at Neurospin (Electa Neuromag) applying filtering and Independent Component Analysis (ICA). The tools might be overfitted to the particular data I processed. This is why you find them here and not in MNE-Python. Yet the idea is to share these tools so they can be further improved and extended by everyone who feels inclined. For the moment the code is designed to filter the data and remove ECG and EOG related components based on concomitant EOG and ECG recordings. Additional use cases might be added in the future. This repository might dissolve at some point and become part of MNE-Python.


Installation
============

Just clone this repository and install using setup.py:

```python setup.py develop --user```

You need an installation of [MNE-Python](https://github.com/dengemann/mne-python) and [Scikit-Learn](https://github.com/scikit-learn/scikit-learn) to use these tools.


Contributing
============

I appreciate contributions. Just use the issue tracker before opening a PR.
