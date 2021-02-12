from os.path import join, exists, abspath

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def plot_info_bar(run_id, save_dir,
                  informativeness_metrics_dir,
                  num_bins, bin_limits, data_proportion):

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "info_bar.pdf")

    result_file = join(informativeness_metrics_dir, 'results[bins={},bin_limits={},data={}].npz'.
                       format(num_bins, bin_limits, data_proportion))

    results = np.load(result_file, "r")

    # Plotting
    # =========================================== #
    sorted_MI = results["sorted_MI_z_x"]
    norm_sorted_MI = sorted_MI / (1.0 * np.log(num_bins))

    plt.bar(range(len(norm_sorted_MI)), height=norm_sorted_MI, width=0.8)
    plt.ylim(bottom=0, top=1)
    plt.xlim(left=-1, right=len(norm_sorted_MI))
    plt.xticks(range(0, len(norm_sorted_MI), 5))

    plt.tight_layout()

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def main():
    run_id = "0_Gz50"
    # run_id = "1_Gz100"
    # run_id = "2_Gz200"
    save_dir = abspath(join(RESULTS_DIR, "celebA", "AAE", "auxiliary", "plot_info_bar",
                            "AAE_{}".format(run_id)))

    informativeness_metrics_dir = abspath(join(RESULTS_DIR, "celebA", "AAE",
        "auxiliary", "informativeness_metrics_v3", "AAE_{}".format(run_id)))

    num_bins = 100
    bin_limits = "(-4.0, 4.0)"
    data_proportion = 1.0

    plot_info_bar(run_id, save_dir, informativeness_metrics_dir,
                  num_bins, bin_limits, data_proportion)


if __name__ == "__main__":
    main()
