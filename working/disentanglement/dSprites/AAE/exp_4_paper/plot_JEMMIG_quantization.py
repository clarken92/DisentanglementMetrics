from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def plot_JEMMIG_quan_tc_beta(save_dir, JEMMIG_quan_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}
    JEMMIGs_by_beta = {}
    JEMMIGs_by_Gz = {}

    for i in range(len(JEMMIG_quan_result_files)):
        JEMMIG_results = np.load(JEMMIG_quan_result_files[i], "r")
        MI_z_y_sorted = JEMMIG_results['MI_z_y_sorted']
        H_z_y_sorted = JEMMIG_results['H_z_y_sorted']
        JEMMIG = np.mean(H_z_y_sorted[0, :] - MI_z_y_sorted[0, :] + MI_z_y_sorted[1, :], axis=0)

        JEMMIGs_all.append(JEMMIG)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                jemmig_list = JEMMIGs_by_Gz.get(Gz)
                if jemmig_list is None:
                    JEMMIGs_by_Gz[Gz] = [JEMMIG]
                else:
                    jemmig_list.append(JEMMIG)

            else:
                beta = int(labels[i][idx + len('beta'):])
                jemmig_list = JEMMIGs_by_beta.get(beta)
                if jemmig_list is None:
                    JEMMIGs_by_beta[beta] = [JEMMIG]
                else:
                    jemmig_list.append(JEMMIG)
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [JEMMIG]
            else:
                jemmig_list.append(JEMMIG)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(JEMMIGs_by_Gz)]
    JEMMIGs_mean_by_Gz = [(Gz, np.mean(jemmig_list)) for Gz, jemmig_list in iteritems(JEMMIGs_by_Gz)]
    JEMMIGs_std_by_Gz = [(Gz, np.std(jemmig_list)) for Gz, jemmig_list in iteritems(JEMMIGs_by_Gz)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in JEMMIGs_mean_by_Gz],
            yerr=[a[1] for a in JEMMIGs_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("JEMMIG (quantization)")
    plt.ylim(bottom=3)
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(9, 3.2)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_RMIG_norm_tc_beta(save_dir, JEMMIG_result_files, labels):
    RMIGs_all = []
    RMIGs_by_tc = {}
    RMIGs_by_beta = {}
    RMIGs_by_Gz = {}

    for i in range(len(JEMMIG_result_files)):
        RMIG_results = np.load(JEMMIG_result_files[i], "r")
        RMIGs_all.append(np.mean(RMIG_results['RMIG_norm_yk']))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                rmig_list = RMIGs_by_Gz.get(Gz)
                if rmig_list is None:
                    RMIGs_by_Gz[Gz] = [np.mean(RMIG_results['RMIG_norm_yk'])]
                else:
                    rmig_list.append(np.mean(RMIG_results['RMIG_norm_yk']))

            else:
                beta = int(labels[i][idx + len('beta'):])
                rmig_list = RMIGs_by_beta.get(beta)
                if rmig_list is None:
                    RMIGs_by_beta[beta] = [np.mean(RMIG_results['RMIG_norm_yk'])]
                else:
                    rmig_list.append(np.mean(RMIG_results['RMIG_norm_yk']))
        else:
            tc = int(labels[i][idx + len('tc'):])
            rmig_list = RMIGs_by_tc.get(tc)
            if rmig_list is None:
                RMIGs_by_tc[tc] = [np.mean(RMIG_results['RMIG_norm_yk'])]
            else:
                rmig_list.append(np.mean(RMIG_results['RMIG_norm_yk']))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(RMIGs_by_tc)]
    RMIGs_mean_by_tc = [(tc, np.mean(rmig_list)) for tc, rmig_list in iteritems(RMIGs_by_tc)]
    RMIGs_std_by_tc = [(tc, np.std(rmig_list)) for tc, rmig_list in iteritems(RMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(RMIGs_by_beta)]
    RMIGs_mean_by_beta = [(beta, np.mean(rmig_list)) for beta, rmig_list in iteritems(RMIGs_by_beta)]
    RMIGs_std_by_beta = [(beta, np.std(rmig_list)) for beta, rmig_list in iteritems(RMIGs_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(RMIGs_by_Gz)]
    RMIGs_mean_by_Gz = [(Gz, np.mean(rmig_list)) for Gz, rmig_list in iteritems(RMIGs_by_Gz)]
    RMIGs_std_by_Gz = [(Gz, np.std(rmig_list)) for Gz, rmig_list in iteritems(RMIGs_by_Gz)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in RMIGs_mean_by_tc],
            yerr=[a[1] for a in RMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in RMIGs_mean_by_beta],
            yerr=[a[1] for a in RMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in RMIGs_mean_by_Gz],
            yerr=[a[1] for a in RMIGs_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("RMIG (normalized)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(9, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_norm_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_RMIG_tc_beta(save_dir, JEMMIG_result_files, labels):
    RMIGs_all = []
    RMIGs_by_tc = {}
    RMIGs_by_beta = {}
    RMIGs_by_Gz = {}

    for i in range(len(JEMMIG_result_files)):
        RMIG_results = np.load(JEMMIG_result_files[i], "r")
        RMIGs_all.append(np.mean(RMIG_results['RMIG_yk']))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                rmig_list = RMIGs_by_Gz.get(Gz)
                if rmig_list is None:
                    RMIGs_by_Gz[Gz] = [np.mean(RMIG_results['RMIG_yk'])]
                else:
                    rmig_list.append(np.mean(RMIG_results['RMIG_yk']))
            else:
                beta = int(labels[i][idx + len('beta'):])
                rmig_list = RMIGs_by_beta.get(beta)
                if rmig_list is None:
                    RMIGs_by_beta[beta] = [np.mean(RMIG_results['RMIG_yk'])]
                else:
                    rmig_list.append(np.mean(RMIG_results['RMIG_yk']))
        else:
            tc = int(labels[i][idx + len('tc'):])
            rmig_list = RMIGs_by_tc.get(tc)
            if rmig_list is None:
                RMIGs_by_tc[tc] = [np.mean(RMIG_results['RMIG_yk'])]
            else:
                rmig_list.append(np.mean(RMIG_results['RMIG_yk']))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(RMIGs_by_tc)]
    RMIGs_mean_by_tc = [(tc, np.mean(rmig_list)) for tc, rmig_list in iteritems(RMIGs_by_tc)]
    RMIGs_std_by_tc = [(tc, np.std(rmig_list)) for tc, rmig_list in iteritems(RMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(RMIGs_by_beta)]
    RMIGs_mean_by_beta = [(beta, np.mean(rmig_list)) for beta, rmig_list in iteritems(RMIGs_by_beta)]
    RMIGs_std_by_beta = [(beta, np.std(rmig_list)) for beta, rmig_list in iteritems(RMIGs_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(RMIGs_by_Gz)]
    RMIGs_mean_by_Gz = [(Gz, np.mean(rmig_list)) for Gz, rmig_list in iteritems(RMIGs_by_Gz)]
    RMIGs_std_by_Gz = [(Gz, np.std(rmig_list)) for Gz, rmig_list in iteritems(RMIGs_by_Gz)]

    print("RMIGs_by_tc: {}".format(RMIGs_by_tc))
    print("RMIGs_by_beta: {}".format(RMIGs_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in RMIGs_mean_by_tc],
            yerr=[a[1] for a in RMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in RMIGs_mean_by_beta],
            yerr=[a[1] for a in RMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in RMIGs_mean_by_Gz],
            yerr=[a[1] for a in RMIGs_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("RMIG")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(9, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


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

        "3a_Gz10_zdim10",
        "3a_Gz10_zdim10",
        "3b_Gz10_zdim10",
        "3c_Gz10_zdim10",

        "2_Gz20_zdim10",
        "2a_Gz20_zdim10",
        "2b_Gz20_zdim10",
        "2c_Gz20_zdim10",

        "1_Gz50_zdim10",
        "1a_Gz50_zdim10",
        "1b_Gz50_zdim10",
        "1c_Gz50_zdim10",
    ]
    labels = []
    for i, run_id in enumerate(run_ids):
        if 'VAE' in run_id:
            labels.append("{}_".format(i) + run_id[run_id.rfind('_') + 1:])
        else:
            labels.append("{}_".format(i) + run_id[run_id.find('_') + 1: run_id.find('_') + 5])

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary", "JEMMIG_quantization_plot"))

    # =========================================== #
    JEMMIG_quan_result_files = []

    num_bins = 200
    bin_limits = (-4.0, 4.0)
    data_proportion = 1.0

    for run_id in run_ids:
        if 'VAE' in run_id:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                         "auxiliary", "interpretability_metrics_v2",
                         "{}_{}".format(enc_dec_model, run_id),
                         "results[bins={},bin_limits={},data={}].npz".format(
                             num_bins, bin_limits, data_proportion)))
        else:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "AAE",
                         "auxiliary", "interpretability_metrics_v2",
                         "{}_{}".format(enc_dec_model, run_id),
                         "results[bins={},bin_limits={},data={}].npz".format(
                             num_bins, bin_limits, data_proportion)))

        JEMMIG_quan_result_files.append(result_file)
        # =========================================== #

    plot_JEMMIG_quan_tc_beta(save_dir, JEMMIG_quan_result_files, labels)


if __name__ == "__main__":
    main()
