# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from mne import pick_types


def get_data_picks(inst, meg_combined=False):
    """Get data channel indices as separate list of tuples

    Parameters
    ----------
    inst : instance of mne.measuerment_info.Info
        The info
    meg_combined : bool
        Whether to return combined picks for grad and mag.

    Returns
    -------
    picks_list : list of tuples
        The list of tuples of picks and the type string.
    """
    info = inst.info
    picks_list = []
    has_mag, has_grad, has_eeg = [k in inst for k in ('mag', 'grad', 'eeg')]
    if has_mag and (meg_combined is not True or not has_grad):
        picks_list.append(
            (pick_types(info, meg='mag', eeg=False, stim=False), 'mag')
        )
    if has_grad and (meg_combined is not True or not has_mag):
        picks_list.append(
            (pick_types(info, meg='grad', eeg=False, stim=False), 'grad')
        )
    if has_mag and has_grad and meg_combined is True:
        picks_list.append(
            (pick_types(info, meg=True, eeg=False, stim=False), 'meg')
        )
    if has_eeg:
        picks_list.append(
            (pick_types(info, meg=False, eeg=True, stim=False), 'eeg')
        )
    return picks_list
