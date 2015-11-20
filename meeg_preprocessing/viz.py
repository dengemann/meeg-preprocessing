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


def _render_components_table(ica):
    css = '''<style type="text/css">
        table.scores {
        border: 1px solid black;
        margin-top: 20px;
        margin-bottom: 20px;
        margin-left: auto;
        margin-right: auto;}
        table.scores th {
        width: 150px;
        text-align:left;
        padding-top: 10px;
        padding-left: 10px;
        font-size: 20px;}
        table.scores td {
        text-align:left;
        padding-top: 10px;
        padding-left: 10px;
        font-size: 16px;}</style>'''

    header = ('<tr><th>label</th><th>component indices</th></tr>')
    table_content = ''
    row = '<tr><td>{label}</td><td>{components}</td></td></tr>'
    for label, components in ica.labels_.items():
        if '/' in label:
            split = label.split('/')
            label = split[0] + '-' + split[-1]
        table_content += row.format(
            label=label,
            components='%s' % components)

    table = '''{0}<h4>ICA solution</h4><div class="thumbnail">
            <table class="scores">{1}{2}</table></div>'''.format(
        css, header, table_content)
    return table
