from os.path import join, abspath

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def plot_num_bins(save_dir, all_result_files, labels, all_num_bins):
    RMIGs = []
    JEMMIs = []
    JEMMIs_unnorms = []
    Sums = []
    Hzy_tops = []

    for i in range(len(labels)):
        for j in range(len(all_num_bins)):
            rmig_results = np.load(all_result_files[i][j], "r")
            print(rmig_results.files)

            RMIGs.append(np.mean(rmig_results['RMIG']))
            JEMMIs.append(np.mean(rmig_results['JEMMI']))

            RMIGs_unnorm = rmig_results['RMIG'] * rmig_results['H_y']
            JEMMIs_unnorm = rmig_results['JEMMI'] * (rmig_results['H_y'] + np.log(all_num_bins[j]))

            JEMMIs_unnorms.append(np.mean(JEMMIs_unnorm))

            Sums.append(np.mean(JEMMIs_unnorm + RMIGs_unnorm))
            Hzy_tops.append(np.mean(rmig_results['H_z_y_sorted'][0]))

    RMIGs = np.reshape(np.asarray(RMIGs, dtype=np.float32),
                       [len(labels), len(all_num_bins)])
    JEMMIs = np.reshape(np.asarray(JEMMIs, dtype=np.float32),
                        [len(labels), len(all_num_bins)])
    JEMMIs_unnorms = np.reshape(np.asarray(JEMMIs_unnorms, dtype=np.float32),
                        [len(labels), len(all_num_bins)])

    Sums = np.reshape(np.asarray(Sums, dtype=np.float32),
                       [len(labels), len(all_num_bins)])
    Hzy_tops = np.reshape(np.asarray(Hzy_tops, dtype=np.float32),
                      [len(labels), len(all_num_bins)])
    print(Hzy_tops)

    '''
    # Plotting num_bins dependency
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_bins, RMIGs[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#bins")
    plt.ylabel("RMIG")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "num_bins_RMIG.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

    # Plotting num_bins dependency
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_bins, JEMMIs[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#bins")
    plt.ylabel("JEMMI")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "num_bins_JEMMI.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

    # Plotting num_bins dependency
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_bins, Sums[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#bins")
    plt.ylabel("JEMMI + RMIG")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "num_bins_JEMMI_RMIG.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #
    #'''

    # Plotting num_bins dependency
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_bins, JEMMIs_unnorms[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#bins")
    plt.ylabel("JEMMIs_unnorms")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "num_bins_JEMMI_RMIG.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

    # Plotting num_bins dependency
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_bins, Hzy_tops[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#bins")
    plt.ylabel("H(z,y)")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "num_bins_Hzy.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def main():
    enc_dec_model = "1Konny"
    run_ids = [
        "0_FactorVAE_tc10",
        "2_FactorVAE_tc20",
        "3_FactorVAE_tc50",
        "9_VAE_beta10",
        "10_VAE_beta20",
        "11_VAE_beta30",
        "6_VAE_beta50",
    ]
    all_num_bins = [20, 50, 80, 100, 150, 200, 300, 500]
    labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "rmig_mig_comparison"))

    # RMIG result files
    # =========================================== #
    all_result_files = []

    bin_limits = "(-4.0, 4.0)"
    data_proportion = 1.0

    for run_id in run_ids:
        result_files = []
        for num_bins in all_num_bins:
            result_files.append(
                abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                    "auxiliary", "interpretability_metrics_v2",
                    "{}_{}".format(enc_dec_model, run_id),
                    "results[bins={},bin_limits={},data={}].npz".format(
                        num_bins, bin_limits, data_proportion)
                )))

        all_result_files.append(result_files)
    # =========================================== #

    plot_num_bins(save_dir, all_result_files, labels, all_num_bins)


if __name__ == "__main__":
    main()
