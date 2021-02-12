import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_corrmat_with_histogram(save_file, data, bins=40,
                                font={}, subplot_adjust={}, size_inches=None):
    # data: (batch, dim)
    assert len(data.shape) == 2, "'data' must be a 2D array of shape (batch, dim)!"
    corr_mat = np.corrcoef(data, rowvar=False)

    matplotlib.rc('font', **font)
    fig, axes = plt.subplots(1, 2)
    plt.subplots_adjust(**subplot_adjust)
    if size_inches is not None:
        plt.gcf().set_size_inches(size_inches)

    # Plot correlation matrix
    # ---------------------------------- #
    cf = axes[0].matshow(np.abs(corr_mat), cmap=plt.get_cmap("binary"), vmin=0.0, vmax=1.0)
    axes[0].set_xticks([])
    axes[0].set_xticklabels([])

    divider = make_axes_locatable(axes[0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    # ---------------------------------- #

    # Plot histogram
    # ---------------------------------- #
    hists = axes[1].hist(np.abs(corr_mat).flatten(), bins=bins)
    for i in range(bins):
        if hists[0][i] > 0:
            plt.text(hists[1][i], 1.05 * hists[0][i], '%d' % int(hists[0][i]))
    axes[1].set_xticks(np.arange(0.0, 1.0, 0.05))
    # ---------------------------------- #

    if save_file.endswith('.pdf'):
        with PdfPages(save_file) as pdf_file:
            plt.savefig(pdf_file, format='pdf')
            plt.close()
    else:
        plt.savefig(save_file, dpi=300)

    plt.close(fig)


def plot_corrmat(save_file, data, font={}, subplot_adjust={}, size_inches=None):
    # data: (batch, dim)
    assert len(data.shape) == 2, "'data' must be a 2D array of shape (batch, dim)!"
    corr_mat = np.corrcoef(data, rowvar=False)

    matplotlib.rc('font', **font)
    fig, ax = plt.subplots()
    plt.subplots_adjust(**subplot_adjust)
    if size_inches is not None:
        plt.gcf().set_size_inches(size_inches)

    # Plot correlation matrix
    # ---------------------------------- #
    cf = ax.matshow(np.abs(corr_mat), cmap=plt.get_cmap("binary"), vmin=0.0, vmax=1.0)
    ax.set_xticks([])
    ax.set_xticklabels([])

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)
    # ---------------------------------- #

    if save_file.endswith('.pdf'):
        with PdfPages(save_file) as pdf_file:
            plt.savefig(pdf_file, format='pdf')
            plt.close()
    else:
        plt.savefig(save_file, dpi=300)

    plt.close(fig)


def plot_comp_dist(save_file_pattern, data, x_lim=None, x_ticks=None, bins=40,
                   font={}, subplot_adjust={}, size_inches=None):

    if len(data.shape) == 1:
        data = np.expand_dims(data, axis=-1)
    assert len(data.shape) == 2, "'data' must be a 1D or 2D array!"

    matplotlib.rc('font', **font)

    fig, ax = plt.subplots(1, 1)
    plt.subplots_adjust(**subplot_adjust)
    if size_inches is not None:
        plt.gcf().set_size_inches(size_inches)

    for i in range(data.shape[1]):
        print("\rPlot {} images".format(i+1), end='')
        ax.clear()

        ax.hist(data[:, i], bins=bins)
        if x_lim is not None:
            assert hasattr(x_lim, '__len__') and len(x_lim) == 2, "'x_lim' must be an array of length 2!"
            ax.set_xlim(x_lim)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)

        if save_file_pattern.endswith('.pdf'):
            with PdfPages(save_file_pattern.format(i)) as pdf_file:
                plt.savefig(pdf_file, format='pdf')
        else:
            plt.savefig(save_file_pattern.format(i), dpi=300)

    print()
    plt.close()