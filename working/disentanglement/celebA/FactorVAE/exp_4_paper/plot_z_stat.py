from os.path import join, exists, abspath

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist
from utils.visualization import plot_comp_dist

from global_settings import RESULTS_DIR


def plot_info_bar(run_id, save_dir,
                  informativeness_metrics_dir,
                  num_bins, bin_limits, data_proportion):

    z_data_file = join(informativeness_metrics_dir, "z_data[data={}].npz".format(data_proportion))

    with np.load(z_data_file, "r") as f:
        all_z_mean = f['all_z_mean']
        all_z_stddev = f['all_z_stddev']

    # Plotting
    # =========================================== #
    save_dir = make_dir_if_not_exist(save_dir)
    plot_comp_dist(join(save_dir, 'z_mean_{}.pdf'), all_z_mean, x_lim=(-5, 5),
                   subplot_adjust={'left': 0.1, 'right': 0.98, 'bottom': 0.05, 'top': 0.95})
    plot_comp_dist(join(save_dir, 'z_stddev_{}.pdf'), all_z_stddev, x_lim=(0, 3),
                   subplot_adjust={'left': 0.1, 'right': 0.98, 'bottom': 0.05, 'top': 0.95})
    # =========================================== #


def main():
    # run_id = "1_tc50_multiSave"
    run_id = "1_tc50_zdim100"
    # run_id = "5_tc50_zdim200"
    # run_id = "6_VAE_beta50"
    # run_id = "8_VAE"
    save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "plot_z_stat",
                            "FactorVAE_{}".format(run_id)))

    informativeness_metrics_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE",
        "auxiliary", "informativeness_metrics_v3", "FactorVAE_{}".format(run_id)))

    num_bins = 100
    bin_limits = "(-4.0, 4.0)"
    data_proportion = 1.0

    plot_info_bar(run_id, save_dir, informativeness_metrics_dir,
                  num_bins, bin_limits, data_proportion)


if __name__ == "__main__":
    main()
