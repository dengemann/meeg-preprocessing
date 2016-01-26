# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from copy import deepcopy

import numpy as np

from mne.report import Report
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs
from mne import pick_types
from mne.utils import logger
from mne.defaults import _handle_default

from .viz import _prepare_filter_plot, _render_components_table
from .utils import get_data_picks


def check_apply_filter(raw, subject, filter_params=None,
                       notch_filter_params=None, plot_fmin=None,
                       plot_fmax=None, n_jobs=1, figsize=None, show=True,
                       report=None, img_scale=1.0):
    """Apply filtering and save diagnostic plots

    Parameters
    ----------
    raw : instance of Raw
        Raw measurements to be decomposed.
    subject : str
        The name of the subject.
    filter_params : dict | list of dict | None
        The parametrs passed to raw.filter. If list, raw.filter will be
        invoked len(filter_params) times. Defaults to None. If None, expands
        to:

        dict(l_freq=0.5, h_freq=200, n_jobs=n_jobs,
             method='fft', l_trans_bandwidth=0.1, h_trans_bandwidth=0.5)
    notch_filter_params : dict | list of dict | None
        The parametrs passed to raw.notch_filter. Defaults to None.
        If None, expands to:
    n_jobs : int
        The number of CPUs to use in parallel.
    figsize : tuple of int
        The figsize in inches. See matplotlib documentation.
    show : bool
        Show figure if True
    scale_img : float
        The scaling factor for the report. Defaults to 1.0.
    report : instance of Report | None
        The report object. If None, a new report will be generated.
    """
    _default_filter_params = dict(l_freq=0.5, h_freq=200, n_jobs=n_jobs,
                                  method='fft',
                                  l_trans_bandwidth=0.1, h_trans_bandwidth=0.5)
    if filter_params is None:
        filter_params = _default_filter_params
    if not isinstance(filter_params, (list, tuple)):
        filter_params = [filter_params]
    if notch_filter_params is None:
        notch_filter_params = dict(freqs=(50, 100, 150, 200, 250,),
                                   method='fft')
    if report is None:
        report = Report(subject)

    notch_filter_params.update(n_jobs=n_jobs)
    picks_list, n_rows, fig, axes = _prepare_filter_plot(raw, figsize)

    iter_plot = zip(axes, picks_list)
    fmin, fmax = plot_fmin or 0, plot_fmax or raw.info['lowpass'] + 20

    ###########################################################################
    # plot before filter
    for ax, (picks, ch_type) in iter_plot:

        raw.plot_psd(fmin=fmin, fmax=fmax, ax=ax,
                     picks=picks, color='black', show=show)
        first_line = ax.get_lines()[0]
        first_line.set_label('{} - raw'.format(ch_type))
        ax.set_ylabel('Power (dB)')
        ax.grid(True)
        ax.set_title(ch_type)

    ###########################################################################
    # filter

    # Note. It turns out to be safer to first run the notch filter.
    # Ohterwise crazy notch resonance with some filter settings.
    raw.notch_filter(**notch_filter_params)

    for filter_params_ in filter_params:
        final_filter_params_ = deepcopy(_default_filter_params)
        final_filter_params_.update(filter_params_)
        final_filter_params_.update({'n_jobs': n_jobs})
        raw.filter(**final_filter_params_)

    ###########################################################################
    # plot after filter
    for ax, (picks, ch_type) in iter_plot:

        raw.plot_psd(fmin=fmin, fmax=fmax, ax=ax,
                     picks=picks, color='red', show=show)
        second_line = ax.get_lines()[1]
        second_line.set_label('{} - filtered'.format(ch_type))
        ax.legend(loc='best')

    fig.suptitle('Multitaper PSD')
    report.add_figs_to_section(fig, 'filter PSD spectra {}'.format(subject),
                               'FILTER', scale=img_scale)
    return fig, report


def _put_artifact_range(info, evoked, kind):
    """Helper to set artifact stats"""
    ch_scales = _handle_default('scalings')
    for this_picks, ch_type in get_data_picks(evoked, meg_combined=False):
        amp_range = (evoked.data[this_picks].max() -
                     evoked.data[this_picks].min()) * ch_scales[ch_type]
        info.update({'%s_amp_range_%s' % (kind, ch_type): amp_range})


def compute_ica(raw, subject, n_components=0.99, picks=None, decim=None,
                reject=None, ecg_tmin=-0.5, ecg_tmax=0.5, eog_tmin=-0.5,
                eog_tmax=0.5, n_max_ecg=3, n_max_eog=1,
                n_max_ecg_epochs=200, show=True, img_scale=1.0,
                random_state=None, report=None, artifact_stats=None):
    """Run ICA in raw data

    Parameters
    ----------,
    raw : instance of Raw
        Raw measurements to be decomposed.
    subject : str
        The name of the subject.
    picks : array-like of int, shape(n_channels, ) | None
        Channels to be included. This selection remains throughout the
        initialized ICA solution. If None only good data channels are used.
        Defaults to None.
    n_components : int | float | None | 'rank'
        The number of components used for ICA decomposition. If int, it must be
        smaller then max_pca_components. If None, all PCA components will be
        used. If float between 0 and 1 components can will be selected by the
        cumulative percentage of explained variance.
        If 'rank', the number of components equals the rank estimate.
        Defaults to 0.99.
    decim : int | None
        Increment for selecting each nth time slice. If None, all samples
        within ``start`` and ``stop`` are used. Defalts to None.
    reject : dict | None
        Rejection parameters based on peak to peak amplitude.
        Valid keys are 'grad' | 'mag' | 'eeg' | 'eog' | 'ecg'.
        If reject is None then no rejection is done. You should
        use such parameters to reject big measurement artifacts
        and not EOG for example. It only applies if `inst` is of type Raw.
        Defaults to {'mag': 5e-12}
    ecg_tmin : float
        Start time before ECG event. Defaults to -0.5.
    ecg_tmax : float
        End time after ECG event. Defaults to 0.5.
    eog_tmin : float
        Start time before rog event. Defaults to -0.5.
    eog_tmax : float
        End time after rog event. Defaults to 0.5.
    n_max_ecg : int | None
        The maximum number of ECG components to exclude. Defaults to 3.
    n_max_eog : int | None
        The maximum number of EOG components to exclude. Defaults to 1.
    n_max_ecg_epochs : int
        The maximum number of ECG epochs to use for phase-consistency
        estimation. Defaults to 200.
    show : bool
        Show figure if True
    scale_img : float
        The scaling factor for the report. Defaults to 1.0.
    random_state : None | int | instance of np.random.RandomState
        np.random.RandomState to initialize the FastICA estimation.
        As the estimation is non-deterministic it can be useful to
        fix the seed to have reproducible results. Defaults to None.
    report : instance of Report | None
        The report object. If None, a new report will be generated.
    artifact_stats : None | dict
        A dict that contains info on amplitude ranges of artifacts and
        numbers of events, etc. by channel type.

    Returns
    -------
    ica : instance of ICA
        The ICA solution.
    report : dict
        A dict with an html report ('html') and artifact statistics ('stats').
    """
    if report is None:
        report = Report(subject=subject, title='ICA preprocessing')
    if n_components == 'rank':
        n_components = raw.estimate_rank(picks=picks)
    ica = ICA(n_components=n_components, max_pca_components=None,
              random_state=random_state, max_iter=256)
    ica.fit(raw, picks=picks, decim=decim, reject=reject)

    comment = []
    for ch in ('mag', 'grad', 'eeg'):
        if ch in ica:
            comment += [ch.upper()]
    if len(comment) > 0:
        comment = '+'.join(comment) + ' '
    else:
        comment = ''

    topo_ch_type = 'mag'
    if 'GRAD' in comment and 'MAG' not in comment:
        topo_ch_type = 'grad'
    elif 'EEG' in comment:
        topo_ch_type = 'eeg'

    ###########################################################################
    # 2) identify bad components by analyzing latent sources.

    title = '%s related to %s artifacts (red) ({})'.format(subject)

    # generate ECG epochs use detection via phase statistics
    reject_ = {'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6}
    if reject is not None:
        reject_.update(reject)
    for ch_type in ['mag', 'grad', 'eeg']:
        if ch_type not in ica:
            reject_.pop(ch_type)

    picks_ = np.array([raw.ch_names.index(k) for k in ica.ch_names])
    if 'eeg' in ica:
        if 'ecg' in raw:
            picks_ = np.append(picks_,
                               pick_types(raw.info, meg=False, ecg=True)[0])
        else:
            logger.info('There is no ECG channel, trying to guess ECG from '
                        'magnetormeters')

    if artifact_stats is None:
        artifact_stats = dict()

    ecg_epochs = create_ecg_epochs(raw, tmin=ecg_tmin, tmax=ecg_tmax,
                                   keep_ecg=True, picks=picks_, reject=reject_)

    n_ecg_epochs_found = len(ecg_epochs.events)
    artifact_stats['ecg_n_events'] = n_ecg_epochs_found
    n_max_ecg_epochs = min(n_max_ecg_epochs, n_ecg_epochs_found)
    artifact_stats['ecg_n_used'] = n_max_ecg_epochs

    sel_ecg_epochs = np.arange(n_ecg_epochs_found)
    rng = np.random.RandomState(42)
    rng.shuffle(sel_ecg_epochs)
    ecg_ave = ecg_epochs.average()

    report.add_figs_to_section(ecg_ave.plot(), 'ECG-full', 'artifacts')
    ecg_epochs = ecg_epochs[sel_ecg_epochs[:n_max_ecg_epochs]]
    ecg_ave = ecg_epochs.average()
    report.add_figs_to_section(ecg_ave.plot(), 'ECG-used', 'artifacts')

    _put_artifact_range(artifact_stats, ecg_ave, kind='ecg')

    ecg_inds, scores = ica.find_bads_ecg(ecg_epochs, method='ctps')
    if len(ecg_inds) > 0:
        ecg_evoked = ecg_epochs.average()
        del ecg_epochs

        fig = ica.plot_scores(scores, exclude=ecg_inds, labels='ecg',
                              title='', show=show)

        report.add_figs_to_section(fig, 'scores ({})'.format(subject),
                                   section=comment + 'ECG',
                                   scale=img_scale)

        current_exclude = [e for e in ica.exclude]  # issue #2608 MNE
        fig = ica.plot_sources(raw, ecg_inds, exclude=ecg_inds,
                               title=title % ('components', 'ecg'), show=show)

        report.add_figs_to_section(fig, 'sources ({})'.format(subject),
                                   section=comment + 'ECG',
                                   scale=img_scale)
        ica.exclude = current_exclude

        fig = ica.plot_components(ecg_inds, ch_type=topo_ch_type,
                                  title='', colorbar=True, show=show)
        report.add_figs_to_section(fig, title % ('sources', 'ecg'),
                                   section=comment + 'ECG', scale=img_scale)
        ica.exclude = current_exclude

        ecg_inds = ecg_inds[:n_max_ecg]
        ica.exclude += ecg_inds
        fig = ica.plot_sources(ecg_evoked, exclude=ecg_inds, show=show)
        report.add_figs_to_section(fig, 'evoked sources ({})'.format(subject),
                                   section=comment + 'ECG',
                                   scale=img_scale)

        fig = ica.plot_overlay(ecg_evoked, exclude=ecg_inds, show=show)
        report.add_figs_to_section(fig,
                                   'rejection overlay ({})'.format(subject),
                                   section=comment + 'ECG',
                                   scale=img_scale)

    # detect EOG by correlation
    picks_eog = np.concatenate(
        [picks_, pick_types(raw.info, meg=False, eeg=False, ecg=False,
                            eog=True)])

    eog_epochs = create_eog_epochs(raw, tmin=eog_tmin, tmax=eog_tmax,
                                   picks=picks_eog, reject=reject_)
    artifact_stats['eog_n_events'] = len(eog_epochs.events)
    artifact_stats['eog_n_used'] = artifact_stats['eog_n_events']
    eog_ave = eog_epochs.average()
    report.add_figs_to_section(eog_ave.plot(), 'EOG-used', 'artifacts')
    _put_artifact_range(artifact_stats, eog_ave, kind='eog')

    eog_inds = None
    if len(eog_epochs.events) > 0:
        eog_inds, scores = ica.find_bads_eog(eog_epochs)

    if eog_inds is not None and len(eog_epochs.events) > 0:
        fig = ica.plot_scores(scores, exclude=eog_inds, labels='eog',
                              show=show, title='')
        report.add_figs_to_section(fig, 'scores ({})'.format(subject),
                                   section=comment + 'EOG',
                                   scale=img_scale)

        current_exclude = [e for e in ica.exclude]  # issue #2608 MNE
        fig = ica.plot_sources(raw, eog_inds, exclude=ecg_inds,
                               title=title % ('sources', 'eog'), show=show)
        report.add_figs_to_section(fig, 'sources', section=comment + 'EOG',
                                   scale=img_scale)
        ica.exclude = current_exclude

        fig = ica.plot_components(eog_inds, ch_type=topo_ch_type,
                                  title='', colorbar=True, show=show)
        report.add_figs_to_section(fig, title % ('components', 'eog'),
                                   section=comment + 'EOG', scale=img_scale)
        ica.exclude = current_exclude

        eog_inds = eog_inds[:n_max_eog]
        ica.exclude += eog_inds

        eog_evoked = eog_epochs.average()
        fig = ica.plot_sources(eog_evoked, exclude=eog_inds, show=show)
        report.add_figs_to_section(
            fig, 'evoked sources ({})'.format(subject),
            section=comment + 'EOG', scale=img_scale)

        fig = ica.plot_overlay(eog_evoked, exclude=eog_inds, show=show)
        report.add_figs_to_section(
            fig, 'rejection overlay({})'.format(subject),
            section=comment + 'EOG', scale=img_scale)
    else:
        del eog_epochs

    # check the amplitudes do not change
    if len(ica.exclude) > 0:
        fig = ica.plot_overlay(raw, show=show)  # EOG artifacts remain
        html = _render_components_table(ica)
        report.add_htmls_to_section(
            html, captions='excluded components',
            section='ICA rejection summary (%s)' % ch_type)
        report.add_figs_to_section(
            fig, 'rejection overlay({})'.format(subject),
            section=comment + 'RAW', scale=img_scale)
    return ica, dict(html=report, stats=artifact_stats)
