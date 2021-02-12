from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist
from global_settings import RESULTS_DIR


def compare_matrices(save_dir, JEMMIG_result_files, metrics_Ridgeway_result_files, labels):
    np.set_printoptions(precision=4, threshold=np.nan, linewidth=1000, suppress=True)
    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        metrics_Ridgeway_results = np.load(metrics_Ridgeway_result_files[i], "r")

        MI_true = np.maximum(JEMMIG_results['MI_zi_yk'], 0.0)
        MI = np.maximum(metrics_Ridgeway_results['MI_zi_yk'], 0.0)

        print("\n=========================")
        print("{}".format(labels[i]))
        print("\nMI_true:\n\n{}".format(MI_true))
        print("\nMI:\n\n{}".format(MI))


def plot_JEMMIG_metricsRidgeway_comparison(save_dir, JEMMIG_result_files,
                                           metrics_Ridgeway_result_files,
                                           labels, factors=None):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(metrics_Ridgeway_result_files), \
        "len(rmig_result_files)={} while len(metrics_Ridgeway_result_files)={}".format(
            len(JEMMIG_result_files), len(metrics_Ridgeway_result_files))

    JEMMIGs = []
    RMIGs = []
    RMIGs_norm = []
    Moduls = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        metrics_Ridgeway_results = np.load(metrics_Ridgeway_result_files[i], "r")

        if factors is None:
            ks = range(len(JEMMIG_results['JEMMIG_yk']))
        else:
            ks = factors

        JEMMIGs.append(np.mean(JEMMIG_results['JEMMIG_yk'][ks]))
        RMIGs.append(np.mean(JEMMIG_results['RMIG_yk'][ks]))
        RMIGs_norm.append(np.mean(JEMMIG_results['RMIG_norm_yk'][ks]))

        Moduls.append(metrics_Ridgeway_results['modularity'])

    JEMMIGs = np.asarray(JEMMIGs, dtype=np.float32)
    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    RMIGs_norm = np.asarray(RMIGs_norm, dtype=np.float32)

    Moduls = np.asarray(Moduls, dtype=np.float32)

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

    X_objs = [JEMMIGs, RMIGs, RMIGs_norm]
    X_names = ['JEMMIG', 'RMIG', 'RMIG (normalized)']
    X_file_names = ['JEMMIG', 'RMIG', 'RMIG_norm']

    Y_objs = [Moduls]
    Y_names = ['Modularity']
    Y_file_names = ['Modul']

    for i in range(len(X_objs)):
        for j in range(len(Y_objs)):

            plt.scatter(X_objs[i], Y_objs[j], s=100, color=colors, marker="o", alpha=0.3)
            for k in range(len(X_objs[i])):
                if X_names[i] == 'JEMMIG' and Y_names[j] == "Modularity":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG' and Y_names[j] == "Modularity":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG (normalized)' and Y_names[j] == "Modularity":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                else:
                    plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')

            plt.xlabel(X_names[i])
            plt.ylabel(Y_names[j])
            subplot_adjust = {'left': 0.17, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
            plt.subplots_adjust(**subplot_adjust)
            # plt.gcf().set_size_inches(4, 3)

            save_dir = make_dir_if_not_exist(save_dir)
            if factors is None:
                save_file = join(save_dir, "{}_{}.pdf".format(X_file_names[i], Y_file_names[j]))
            else:
                save_file = join(save_dir, "{}_{}_{}.pdf".format(X_file_names[i], Y_file_names[j], factors))

            with PdfPages(save_file) as pdf_file:
                plt.savefig(pdf_file, format='pdf')

            plt.show()
            plt.close()


def plot_Modularity_tc_beta(save_dir, metrics_Ridgeway_result_files, labels):
    Moduls_all = []
    Moduls_by_tc = {}
    Moduls_by_beta = {}

    for i in range(len(metrics_Ridgeway_result_files)):
        results = np.load(metrics_Ridgeway_result_files[i], "r")
        modul = results['modularity']
        Moduls_all.append(modul)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            modul_list = Moduls_by_beta.get(beta)
            if modul_list is None:
                Moduls_by_beta[beta] = [modul]
            else:
                modul_list.append(modul)
        else:
            tc = int(labels[i][idx + len('tc'):])
            modul_list = Moduls_by_tc.get(tc)
            if modul_list is None:
                Moduls_by_tc[tc] = [modul]
            else:
                modul_list.append(modul)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(Moduls_by_tc)]
    Moduls_mean_by_tc = [(tc, np.mean(modul_list)) for tc, modul_list in iteritems(Moduls_by_tc)]
    Moduls_std_by_tc = [(tc, np.std(modul_list)) for tc, modul_list in iteritems(Moduls_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(Moduls_by_beta)]
    Moduls_mean_by_beta = [(beta, np.mean(modul_list)) for beta, modul_list in iteritems(Moduls_by_beta)]
    Moduls_std_by_beta = [(beta, np.std(modul_list)) for beta, modul_list in iteritems(Moduls_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in Moduls_mean_by_tc],
            yerr=[a[1] for a in Moduls_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in Moduls_mean_by_beta],
            yerr=[a[1] for a in Moduls_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("Modularity")
    plt.ylim(bottom=0.8, top=1.0)
    # plt.tight_layout()

    subplot_adjust = {'left': 0.13, 'right': 0.99, 'bottom': 0.18, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "Modul_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_Modularity_trueMI_tc_beta(save_dir, JEMMIG_result_files, labels):
    Moduls_all = []
    Moduls_by_tc = {}
    Moduls_by_beta = {}

    for i in range(len(JEMMIG_result_files)):
        results = np.load(JEMMIG_result_files[i], "r")

        MI_zi_yk = np.maximum(results['MI_zi_yk'], 0)
        num_latents = MI_zi_yk.shape[0]
        num_factors = MI_zi_yk.shape[1]

        yk_idx_max = np.argmax(MI_zi_yk, axis=1)
        MI_zi_yk_top_over_k = MI_zi_yk[np.arange(0, num_latents), yk_idx_max]

        T = np.zeros([num_latents, num_factors], dtype=np.float32)
        for j in range(num_latents):
            T[j, yk_idx_max[j]] = MI_zi_yk[j, yk_idx_max[j]]

        modularity_scores = 1 - (np.sum((MI_zi_yk - T) ** 2, axis=1) /
                                 np.maximum(MI_zi_yk_top_over_k ** 2 * (num_factors - 1), 1e-8))
        # modularity_scores = np.sum((MI_zi_yk - T) ** 2, axis=1)

        modularity = np.mean(modularity_scores, axis=0)
        modul = modularity

        print("\nlabel[{}]: {}".format(i, labels[i]))
        print("modul: {}".format(modul))

        Moduls_all.append(modul)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            modul_list = Moduls_by_beta.get(beta)
            if modul_list is None:
                Moduls_by_beta[beta] = [modul]
            else:
                modul_list.append(modul)
        else:
            tc = int(labels[i][idx + len('tc'):])
            modul_list = Moduls_by_tc.get(tc)
            if modul_list is None:
                Moduls_by_tc[tc] = [modul]
            else:
                modul_list.append(modul)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(Moduls_by_tc)]
    Moduls_mean_by_tc = [(tc, np.mean(modul_list)) for tc, modul_list in iteritems(Moduls_by_tc)]
    Moduls_std_by_tc = [(tc, np.std(modul_list)) for tc, modul_list in iteritems(Moduls_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(Moduls_by_beta)]
    Moduls_mean_by_beta = [(beta, np.mean(modul_list)) for beta, modul_list in iteritems(Moduls_by_beta)]
    Moduls_std_by_beta = [(beta, np.std(modul_list)) for beta, modul_list in iteritems(Moduls_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in Moduls_mean_by_tc],
            yerr=[a[1] for a in Moduls_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in Moduls_mean_by_beta],
            yerr=[a[1] for a in Moduls_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    # plt.legend()
    plt.xlabel("model")
    plt.ylabel("Modularity")
    plt.ylim(bottom=0.8, top=1.0)
    # plt.tight_layout()

    subplot_adjust = {'left': 0.13, 'right': 0.99, 'bottom': 0.18, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "Modul_trueMI_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_trueMI_MI_correlation_tc_beta(save_dir, JEMMIG_result_files,
                                       metrics_Ridgeway_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(metrics_Ridgeway_result_files), \
        "len(rmig_result_files)={} while len(metrics_Ridgeway_result_files)={}".format(
            len(JEMMIG_result_files), len(metrics_Ridgeway_result_files))

    trueMIs = []
    MIs = []

    np.set_printoptions(suppress=True, precision=4, linewidth=1000, threshold=np.nan)
    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        metrics_Ridgeway_results = np.load(metrics_Ridgeway_result_files[i], "r")

        print("\n{}".format(labels[i]))
        print("True MI_zi_yk:\n{}".format(np.maximum(JEMMIG_results['MI_zi_yk'], 0)))
        print("MI_zi_yk:\n{}".format(metrics_Ridgeway_results['MI_zi_yk']))

        true_MI_zi_yk_top = np.max(JEMMIG_results['MI_zi_yk'], axis=1)
        MI_zi_yk_top = np.max(metrics_Ridgeway_results['MI_zi_yk'], axis=1)

        trueMIs.append(np.mean(true_MI_zi_yk_top, axis=0))
        MIs.append(np.mean(MI_zi_yk_top, axis=0))

    trueMIs = np.asarray(trueMIs, dtype=np.float32)
    MIs = np.asarray(MIs, dtype=np.float32)

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

    plt.scatter(trueMIs, MIs, s=100, color=colors, marker="o", alpha=0.3)
    plt.xlabel("MI(zi, yk*) (using q(zi|x))")
    plt.ylabel("MI(zi, yk*) (no q(zi|x)")
    subplot_adjust = {'left': 0.17, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
    plt.subplots_adjust(**subplot_adjust)
    # plt.gcf().set_size_inches(4, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "trueMI_MI.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def main():
    enc_dec_model = "1Konny"
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
    # labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]
    labels = ["{}_".format(i) + run_id[run_id.rfind('_') + 1:] for i, run_id in enumerate(run_ids)]

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "metrics_Ridgeway_plot"))

    # Metrics Eastwood result files
    # =========================================== #
    metrics_Ridgeway_result_files = []

    num_bins = 100

    for run_id in run_ids:
        metrics_Ridgeway_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                         "auxiliary", "metrics_Ridgeway", "{}_{}".format(enc_dec_model, run_id),
                         "results[bins={}].npz".format(num_bins))))
    # =========================================== #

    # JEMMIG result files
    # =========================================== #
    JEMMIG_result_files = []

    num_samples = 10000
    for run_id in run_ids:
        JEMMIG_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                         "auxiliary", "JEMMIG", "{}_{}".format(enc_dec_model, run_id),
                         "results[num_samples={}].npz".format(num_samples))))
    # =========================================== #

    # plot_trueMI_MI_correlation_tc_beta(save_dir, JEMMIG_result_files,
    #                                    metrics_Ridgeway_result_files, labels)
    # plot_JEMMIG_metricsRidgeway_comparison(save_dir, JEMMIG_result_files,
    #                                        metrics_Ridgeway_result_files, labels)
    # plot_Modularity_tc_beta(save_dir, metrics_Ridgeway_result_files, labels)
    # plot_Modularity_trueMI_tc_beta(save_dir, JEMMIG_result_files, labels)

    #'''
    ids = [4, 24]
    compare_matrices(save_dir,
                     [JEMMIG_result_files[i] for i in ids],
                     [metrics_Ridgeway_result_files[i] for i in ids],
                     [labels[i] for i in ids])
    #'''


if __name__ == "__main__":
    main()
