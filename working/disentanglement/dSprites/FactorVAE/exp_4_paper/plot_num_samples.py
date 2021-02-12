from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def plot_JEMMIG_num_samples(save_dir, all_JEMMIG_result_files, labels, all_num_samples):
    JEMMIGs = []

    for i in range(len(labels)):
        for j in range(len(all_num_samples)):
            JEMMIG_results = np.load(all_JEMMIG_result_files[i][j], "r")
            JEMMIGs.append(np.mean(JEMMIG_results['JEMMIG_yk']))

    JEMMIGs = np.reshape(np.asarray(JEMMIGs, dtype=np.float32),
                        [len(labels), len(all_num_samples)])

    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_samples, JEMMIGs[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#samples")
    plt.ylabel("JEMMIG")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_num_samples.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def plot_SEPIN_num_samples(save_dir, all_SEPIN_result_files, labels, all_num_samples):
    WSEPINs = []

    for i in range(len(labels)):
        for j in range(len(all_num_samples)):
            SEPIN_results = np.load(all_SEPIN_result_files[i][j], "r")
            WSEPINs.append(np.mean(SEPIN_results['WSEPIN']))

    WSEPINs = np.reshape(np.asarray(WSEPINs, dtype=np.float32),
                         [len(labels), len(all_num_samples)])

    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    for i in range(len(labels)):
        plt.plot(all_num_samples, WSEPINs[i], '-', label=labels[i], marker='o')

    plt.legend()
    plt.xlabel("#samples")
    plt.ylabel("WSEPIN")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "WSEPIN_num_samples.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def main():
    enc_dec_model = "1Konny"
    run_ids = [
        # "0_FactorVAE_tc10",
        # "2_FactorVAE_tc20",
        # "3_FactorVAE_tc50",
        "9_VAE_beta10",
        "49b_VAE_beta10_z20",
        # "10_VAE_beta20",
        # "11_VAE_beta30",
        # "6_VAE_beta50",
    ]
    all_num_samples = [1000, 2000, 5000, 10000, 20000, 50000]
    labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "plot_num_samples"))

    # JEMMIG sampling result files
    # =========================================== #
    all_result_files = []

    for run_id in run_ids:
        result_files = []
        for num_samples in all_num_samples:
            result_files.append(
                abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                             "auxiliary", "JEMMIG", "{}_{}".format(enc_dec_model, run_id),
                             "results[num_samples={}].npz".format(num_samples))))

        all_result_files.append(result_files)
    # =========================================== #

    # SEPIN sampling result files
    # =========================================== #
    all_result_files = []

    for run_id in run_ids:
        result_files = []
        for num_samples in all_num_samples:
            result_files.append(
                abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                             "auxiliary", "SEPIN", "{}_{}".format(enc_dec_model, run_id),
                             "results[num_samples={}].npz".format(num_samples))))

        all_result_files.append(result_files)
    # =========================================== #

    # plot_JEMMIG_num_samples(save_dir, all_result_files, labels, all_num_samples)
    plot_SEPIN_num_samples(save_dir, all_result_files, labels, all_num_samples)


if __name__ == "__main__":
    main()
