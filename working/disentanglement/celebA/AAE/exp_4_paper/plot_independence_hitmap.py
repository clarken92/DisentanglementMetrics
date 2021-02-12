from os.path import join, exists, abspath

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)


def plot_info_bar(run_id, save_dir,
                  independence_metrics_dir,
                  z_dim, num_bins, bin_limits, data_proportion):

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "independence_hitmap.pdf")

    result_file = join(independence_metrics_dir, 'results[bins={},bin_limits={},data={},k=-1].npz'.
                       format(num_bins, bin_limits, data_proportion))

    results = np.load(result_file, "r")

    # Plotting
    # =========================================== #
    H_z1z2_mean = results['H_z1z2_mean']
    MI_z1z2_mean = results['MI_z1z2_mean']

    # values = np.reshape(MI_z1z2_mean, [z_dim, z_dim])
    values = np.ones([z_dim, z_dim], dtype=np.float32)
    count = 0
    for i in range(0, z_dim):
        for j in range(i+1, z_dim):
            values[i, j] = MI_z1z2_mean[count]
            values[j, i] = MI_z1z2_mean[count]
            count += 1

    values = values / (2 * np.log(num_bins))
    print("values (max/min/mean): {:.3f}/{:.3f}/{:.3f}".format(np.max(values), np.min(values), np.mean(values)))

    fig, ax = plt.subplots()
    cf = ax.matshow(values, vmin=0, vmax=0.7)
    # cf = ax.matshow(values, vmin=0, vmax=0.1)
    ax.set_xticks(range(0, z_dim, 5))
    ax.set_yticks(range(0, z_dim, 5))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(cf, cax=cax)

    matplotlib.rcParams.update({'font.size': 14})
    plt.subplots_adjust(**{'left': 0.03, 'right': 0.96, 'bottom': 0.02, 'top': 0.95})
    plt.gcf().set_size_inches((6.8, 6))

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def main():
    run_id = "0_Gz50"
    save_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", "auxiliary", "plot_independence_hitmap",
                            "AAE_{}".format(run_id)))

    independence_metrics_dir = abspath(join(RESULTS_DIR, "celebA", "AAE",
        "auxiliary", "independence_metrics", "AAE_{}".format(run_id)))

    z_dim = 65
    num_bins = 50
    bin_limits = "(-4.0, 4.0)"
    data_proportion = 0.1

    plot_info_bar(run_id, save_dir, independence_metrics_dir,
                  z_dim, num_bins, bin_limits, data_proportion)


if __name__ == "__main__":
    main()
