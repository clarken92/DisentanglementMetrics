from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def plot_comparison_chen(save_dir, rmig_result_files, mig_chen_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(rmig_result_files) == len(mig_chen_result_files), \
        "len(rmig_result_files)={} while len(mig_chen_result_files)={}".format(
            len(rmig_result_files), len(mig_chen_result_files))

    RMIGs = []
    JEMMIs = []
    MIGs_chen = []

    for i in range(len(rmig_result_files)):
        rmig_results = np.load(rmig_result_files[i], "r")
        mig_chen_results = np.load(mig_chen_result_files[i], "r")

        RMIGs.append(np.mean(rmig_results['RMIG']))
        JEMMIs.append(np.mean(rmig_results['JEMMI']))
        MIGs_chen.append(np.mean(mig_chen_results['MIG']))

    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    JEMMIs = np.asarray(JEMMIs, dtype=np.float32)
    MIGs_chen = np.asarray(MIGs_chen, dtype=np.float32)

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    plt.scatter(RMIGs, MIGs_chen, s=100, c="b", marker="o", alpha=0.3)
    for i in range(len(RMIGs)):
        if i == 4:
            plt.text(RMIGs[i] * 1.01, MIGs_chen[i] * 1.02, labels[i], fontsize=12, ha='center')
        elif i == 2:
            plt.text(RMIGs[i], MIGs_chen[i] * 0.97, labels[i], fontsize=12, ha='left')
        elif i == 7:
            plt.text(RMIGs[i], MIGs_chen[i] * 1.0, labels[i], fontsize=12, ha='right')
        elif i == 9:
            plt.text(RMIGs[i], MIGs_chen[i] * 0.97, labels[i], fontsize=12, ha='right')
        elif i == 6:
            plt.text(RMIGs[i], MIGs_chen[i] * 0.95, labels[i], fontsize=12, ha='left')
        elif i == 5:
            plt.text(RMIGs[i], MIGs_chen[i] * 0.95, labels[i], fontsize=12, ha='left')
        elif i == 3:
            plt.text(RMIGs[i], MIGs_chen[i] * 1, labels[i], fontsize=12, ha='right')
        elif i == 8:
            plt.text(RMIGs[i], MIGs_chen[i] * 1, labels[i], fontsize=12, ha='right')
        elif i == 15:
            plt.text(RMIGs[i], MIGs_chen[i] * 1, labels[i], fontsize=12, ha='left')
        elif i == 1:
            plt.text(RMIGs[i], MIGs_chen[i] * 1, labels[i], fontsize=12, ha='right')
        elif i == 10:
            plt.text(RMIGs[i], MIGs_chen[i] * 1.05, labels[i], fontsize=12, ha='center')
        elif i == 16:
            plt.text(RMIGs[i], MIGs_chen[i], labels[i], fontsize=12, ha='left')
        elif i == 17:
            plt.text(RMIGs[i], MIGs_chen[i] * 0.92, labels[i], fontsize=12, ha='left')
        elif i == 11:  # beta50
            plt.text(RMIGs[i], MIGs_chen[i] * 0.90, labels[i], fontsize=12, ha='center')
        elif i == 12:  # beta 1
            plt.text(RMIGs[i], MIGs_chen[i] * 1.05, labels[i], fontsize=12, ha='center')
        else:
            plt.text(RMIGs[i], MIGs_chen[i], labels[i], fontsize=12, ha='center')

    plt.plot([0, 0.6], [0, 0.6], 'r-')
    # plt.xlim(left=0, right=1)
    # plt.ylim(bottom=0, top=1)
    plt.xlabel("RMIG")
    plt.ylabel("MIG (Chen et. al.)")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_MIG_chen.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

    '''
    # Plotting JEMMI-MIG relationship
    # =========================================== #
    plt.scatter(JEMMIs, MIGs_chen, s=50, c="b", marker="o", alpha=0.3)
    # plt.xlim(left=0, right=1)
    # plt.ylim(bottom=0, top=1)
    plt.xlabel("JEMMIs")
    plt.ylabel("MIG (Chen)")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMI_MIG.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #
    '''


def plot_comparison_chen_tc_beta(save_dir, rmig_result_files, mig_chen_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(rmig_result_files) == len(mig_chen_result_files), \
        "len(rmig_result_files)={} while len(mig_chen_result_files)={}".format(
            len(rmig_result_files), len(mig_chen_result_files))

    RMIGs = []
    MIGs_chen = []

    for i in range(len(rmig_result_files)):
        my_results = np.load(rmig_result_files[i], "r")
        chen_results = np.load(mig_chen_result_files[i], "r")

        RMIGs.append(np.mean(my_results['RMIG']))
        MIGs_chen.append(np.mean(chen_results['MIG']))

    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    MIGs_chen = np.asarray(MIGs_chen, dtype=np.float32)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        else:
            colors.append("orange")

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    plt.plot([0, 0.6], [0, 0.6], 'r-')
    plt.scatter(RMIGs, MIGs_chen, s=100, color=colors, marker="o", alpha=0.3)

    plt.xlabel('RMIG')
    plt.ylabel('MIG (Chen et. al.)')
    subplot_adjust = {'left': 0.14, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
    plt.subplots_adjust(**subplot_adjust)
    # plt.gcf().set_size_inches(4, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_MIG_chen2.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def plot_comparison_locatello(save_dir, rmig_result_files, mig_locatello_result_files, labels):
    # Plot comparison between rmig and mig_locatello
    assert len(rmig_result_files) == len(mig_locatello_result_files), \
        "len(rmig_result_files)={} while len(mig_locatello_result_files)={}".format(
            len(rmig_result_files), len(mig_locatello_result_files))

    RMIGs = []
    MIGs_locatello = []

    for i in range(len(rmig_result_files)):
        rmig_results = np.load(rmig_result_files[i], "r")
        mig_locatello_results = np.load(mig_locatello_result_files[i], "r")

        RMIGs.append(np.mean(rmig_results['RMIG']))
        MIGs_locatello.append(np.mean(mig_locatello_results['MIG']))

    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    MIGs_locatello = np.asarray(MIGs_locatello, dtype=np.float32)

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    plt.scatter(RMIGs, MIGs_locatello, s=100, c="b", marker="o", alpha=0.3)

    for i in range(len(RMIGs)):
        # plt.text(RMIGs[i], MIGs_locatello[i], labels[i], fontsize=12, ha='center')
        if i == 16:  # beta 4
            plt.text(RMIGs[i], MIGs_locatello[i] * 0.95, labels[i], fontsize=12, ha='center')
        elif i == 18:  # beta 1
            plt.text(RMIGs[i], MIGs_locatello[i] * 0.98, labels[i], fontsize=12, ha='right')
        elif i == 17:  # beta 1
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.04, labels[i], fontsize=12, ha='center')
        elif i == 14:  # beta 1
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.01, labels[i], fontsize=12, ha='center')
        elif i == 5:
            plt.text(RMIGs[i], MIGs_locatello[i] * 0.97, labels[i], fontsize=12, ha='left')
        elif i == 9:
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.00, labels[i], fontsize=12, ha='right')
        elif i == 6:
            plt.text(RMIGs[i], MIGs_locatello[i] * 0.98, labels[i], fontsize=12, ha='left')
        elif i == 7:
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.02, labels[i], fontsize=12, ha='center')
        elif i == 4:
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.00, labels[i], fontsize=12, ha='left')
        elif i == 2:
            plt.text(RMIGs[i], MIGs_locatello[i] * 1.02, labels[i], fontsize=12, ha='center')
        else:
            plt.text(RMIGs[i], MIGs_locatello[i], labels[i], fontsize=12, ha='center')
        #'''

    plt.plot([0, 0.6], [0, 0.6], 'r-')
    # plt.xlim(left=0, right=1)
    # plt.ylim(bottom=0, top=1)
    plt.xlabel("RMIG")
    plt.ylabel("MIG (Locatello et. al.)")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_MIG_locatello.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_comparison_locatello_tc_beta(save_dir, rmig_result_files, mig_locatello_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(rmig_result_files) == len(mig_locatello_result_files), \
        "len(rmig_result_files)={} while len(mig_chen_result_files)={}".format(
            len(rmig_result_files), len(mig_locatello_result_files))

    RMIGs = []
    MIGs_locatello = []

    for i in range(len(rmig_result_files)):
        my_results = np.load(rmig_result_files[i], "r")
        chen_results = np.load(mig_locatello_result_files[i], "r")

        RMIGs.append(np.mean(my_results['RMIG']))
        MIGs_locatello.append(np.mean(chen_results['MIG']))

    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    MIGs_locatello = np.asarray(MIGs_locatello, dtype=np.float32)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        else:
            colors.append("orange")

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    plt.plot([0, 0.6], [0, 0.6], 'r-')

    plt.scatter(RMIGs, MIGs_locatello, s=100, color=colors, marker="o", alpha=0.3)
    for k in range(len(labels)):
        if k == 8 or k == 9 or k == 12 or k == 13 or k == 17 or k == 25 or k == 28 or k == 32:
            plt.text(RMIGs[k], MIGs_locatello[k], labels[k], fontsize=12, ha='center')

    plt.xlabel('RMIG')
    plt.ylabel('MIG (Locatello et. al.)')
    subplot_adjust = {'left': 0.14, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
    plt.subplots_adjust(**subplot_adjust)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_MIG_locatello2.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def main():
    enc_dec_model = "1Konny"

    #'''
    run_ids = [
        "0_FactorVAE_tc10",
        "0a_FactorVAE_tc10",
        "0b_FactorVAE_tc10",
        "0c_FactorVAE_tc10",

        "2_FactorVAE_tc20",
        "2a_FactorVAE_tc20",
        "2b_FactorVAE_tc20",
        "2c_FactorVAE_tc20",

        "3_FactorVAE_tc50",
        "3a_FactorVAE_tc50",
        "3b_FactorVAE_tc50",
        "3c_FactorVAE_tc50",

        "4_FactorVAE_tc100",
        "4a_FactorVAE_tc100",
        "4b_FactorVAE_tc100",
        "4c_FactorVAE_tc100",

        "7_VAE_beta1",
        "7a_VAE_beta1",
        "7b_VAE_beta1",
        "7c_VAE_beta1",

        "8_VAE_beta4",
        "8a_VAE_beta4",
        "8b_VAE_beta4",
        "8c_VAE_beta4",

        "9_VAE_beta10",
        "9a_VAE_beta10",
        "9b_VAE_beta10",
        "9c_VAE_beta10",

        "10_VAE_beta20",
        "10a_VAE_beta20",
        "10b_VAE_beta20",
        "10c_VAE_beta20",
        
        "6_VAE_beta50",
        "6a_VAE_beta50",
        "6b_VAE_beta50",
        "6c_VAE_beta50",
    ]
    #'''

    '''
    run_ids = ["0_FactorVAE_tc10",
               "1_FactorVAE_tc4",
               "2_FactorVAE_tc20",
               "2a_FactorVAE_tc20",
               "2b_FactorVAE_tc20",
               "2c_FactorVAE_tc20",
               "2z_FactorVAE_tc30",
               "3_FactorVAE_tc50",
               "3a_FactorVAE_tc50",
               "3b_FactorVAE_tc50",
               "4_FactorVAE_tc100",
               "6_VAE_beta50",
               "7_VAE_beta1",
               "8_VAE_beta4",
               "9_VAE_beta10",
               "10_VAE_beta20",
               "11_VAE_beta30",
    ]
    '''

    # labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]
    labels = ["{}_".format(i) + run_id[run_id.rfind('_') + 1:] for i, run_id in enumerate(run_ids)]

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "rmig_mig_comparison"))

    # RMIG result files
    # =========================================== #
    rmig_result_files = []

    num_bins = 100  # 100
    bin_limits = "(-4.0, 4.0)"
    data_proportion = 1.0

    for run_id in run_ids:
        rmig_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                "auxiliary", "interpretability_metrics_v2",
                "{}_{}".format(enc_dec_model, run_id),
                "results[bins={},bin_limits={},data={}].npz".format(
                    num_bins, bin_limits, data_proportion)
            )))
    # =========================================== #

    # MIG (Chen) result files
    # =========================================== #
    mig_chen_result_files = []

    num_samples = 10000
    for run_id in run_ids:
        mig_chen_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                "auxiliary", "mig_chen",
                "{}_{}".format(enc_dec_model, run_id),
                "results[num_samples={}].npz".format(num_samples)
            )))
    # =========================================== #

    # MIG (Locatello) result files
    # =========================================== #
    mig_locatello_result_files = []

    num_bins = 100
    data_proportion = 1.0
    for run_id in run_ids:
        mig_locatello_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                         "auxiliary", "mig_locatello",
                         "{}_{}".format(enc_dec_model, run_id),
                         "results[bins={},data={}].npz".format(num_bins, data_proportion)
                         )))
    # =========================================== #

    # plot_comparison_chen(save_dir, rmig_result_files, mig_chen_result_files, labels)
    # plot_comparison_locatello(save_dir, rmig_result_files, mig_locatello_result_files, labels)
    plot_comparison_chen_tc_beta(save_dir, rmig_result_files, mig_chen_result_files, labels)
    plot_comparison_locatello_tc_beta(save_dir, rmig_result_files, mig_locatello_result_files, labels)


if __name__ == "__main__":
    main()
