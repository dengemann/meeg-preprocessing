# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

from .utils import get_data_picks


def _prepare_filter_plot(raw, figsize):
    """Aux function"""
    import matplotlib.pyplot as plt

    picks_list = get_data_picks(raw)
    n_rows = len(picks_list)
    fig, axes = plt.subplots(1, n_rows, sharey=True, sharex=True,
                             figsize=(6 * n_rows, 6) if figsize is None
                             else figsize)
    if n_rows == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    return picks_list, n_rows, fig, axes


def plot_psd_ica_overlay(raw, ica, fmin=None, fmax=None, n_jobs=1,
                         figsize=None, show=True, copy=True):
    """Plot raw power spectrum before and after ICA

    Note. Additional arguments can be passed to raw.plot_psd
    Using **kwargs.

    Parameters
    ----------
    raw : instance of Raw
        Raw measurements to be decomposed.
    ica : instance of ICA
        The ICA solution.
    fmin : float
        Start frequency to consider. Is passed to `raw.plot_psd`. Defaults
        to None.
    fmax : float
        End frequency to consider. Is passed to `raw.plot_psd`. Defaults to
        None.
    subject : str
        The name of the subject.
    n_jobs : int
        The number of CPUs to use in parallel. Is passed to `raw.plot_psd`.
    figsize : tuple of int
        The figsize in inches. See matplotlib documentation.
    show : bool
        Show figure if True

    Returns
    -------
    fig : matplotlib.figure.Figure.
        The figure object.
    """
    picks_list, n_rows, fig, axes = _prepare_filter_plot(raw, figsize)

    iter_plot = zip(axes, picks_list)
    fmin, fmax = fmin or 0, fmax or raw.info['lowpass'] + 20

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
    # ICA
    if copy is True:
        raw = raw.copy()
    ica.apply(raw)

    ###########################################################################
    # plot after ICA
    for ax, (picks, ch_type) in iter_plot:

        raw.plot_psd(fmin=fmin, fmax=fmax, ax=ax,
                     picks=picks, color='red', show=show)
        second_line = ax.get_lines()[1]
        second_line.set_label('{} - ICA applied'.format(ch_type))
        ax.legend(loc='best')

    fig.suptitle('Multitaper PSD')
    return fig
