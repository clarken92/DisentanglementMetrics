from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist
from global_settings import RESULTS_DIR


def compare_matrices(save_dir, JEMMIG_result_files, metrics_Eastwood_result_files,
                     labels, factors=None):
    np.set_printoptions(precision=4, threshold=np.nan, linewidth=1000, suppress=True)
    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        metrics_Eastwood_results = np.load(metrics_Eastwood_result_files[i], "r")

        if factors is None:
            factors = range(JEMMIG_results['JEMMIG_yk'].shape[1])

        MI = np.maximum(JEMMIG_results['MI_zi_yk'][:, factors], 0.0)
        MI_ids_sorted = np.argsort(MI, axis=0)[::-1]
        MI_sorted = np.take_along_axis(MI, MI_ids_sorted, axis=0)

        R = np.abs(metrics_Eastwood_results['importance_matrix'])
        P_disent = R / np.maximum(np.sum(R, axis=1, keepdims=True), 1e-8)
        one_m_H_disent = np.expand_dims(metrics_Eastwood_results['disentanglement_scores'], axis=-1)
        coeff_disent = np.sum(R, axis=1, keepdims=True) / np.sum(R, keepdims=True)

        R_ids_sorted = np.argsort(R, axis=0)[::-1]
        R_sorted = np.take_along_axis(R, R_ids_sorted, axis=0)

        print("\n=========================")
        print("{}".format(labels[i]))
        print("\nMI:\n\n{}".format(MI))
        print("\nR:\n\n{}".format(R))
        print("\nP_disent:\n{}".format(P_disent))
        print("\n1-H_disent:\n{}".format(one_m_H_disent))
        print("\ncoeff_disent:\n{}".format(coeff_disent))

        print("\n\nMI_sorted:\n{}".format(MI_sorted))
        print("\nR_sorted:\n{}".format(R_sorted))
        print("\nMI_ids_sorted:\n{}".format(MI_ids_sorted))
        print("\nR_ids_sorted:\n{}".format(R_ids_sorted))



def plot_JEMMIG_metricsEastwood_comparison(save_dir, JEMMIG_result_files,
                                           metrics_Eastwood_result_files,
                                           labels, factors=None):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(metrics_Eastwood_result_files), \
        "len(JEMMIG_result_files)={} while len(metrics_Eastwood_result_files)={}".format(
            len(JEMMIG_result_files), len(metrics_Eastwood_result_files))

    JEMMIGs = []
    RMIGs = []
    RMIGs_norm = []
    Disents = []
    Comps = []
    Errors = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        metrics_Eastwood_results = np.load(metrics_Eastwood_result_files[i], "r")

        if factors is None:
            factors = range(len(JEMMIG_results['JEMMIG_yk']))

        JEMMIGs.append(np.mean(JEMMIG_results['JEMMIG_yk'][factors]))
        RMIGs.append(np.mean(JEMMIG_results['RMIG_yk'][factors]))
        RMIGs_norm.append(np.mean(JEMMIG_results['RMIG_norm_yk'][factors]))

        # Disents.append(metrics_Eastwood_results['disentanglement'])
        disentanglement_scores = metrics_Eastwood_results['disentanglement_scores']
        R = np.abs(metrics_Eastwood_results['importance_matrix'])
        c_rel_importance = np.sum(R, axis=1) / np.sum(R)
        disentanglement = np.sum(disentanglement_scores * c_rel_importance)
        Disents.append(disentanglement)

        Comps.append(metrics_Eastwood_results['completeness'])
        Errors.append(metrics_Eastwood_results['train_avg_error'])

    JEMMIGs = np.asarray(JEMMIGs, dtype=np.float32)
    RMIGs = np.asarray(RMIGs, dtype=np.float32)
    RMIGs_norm = np.asarray(RMIGs_norm, dtype=np.float32)

    Disents = np.asarray(Disents, dtype=np.float32)
    Comps = np.asarray(Comps, dtype=np.float32)
    Errors = np.asarray(Errors, dtype=np.float32)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        elif "beta" in l:
            colors.append("orange")
        else:
            colors.append("green")

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 16}
    matplotlib.rc('font', **font)

    X_objs = [JEMMIGs, RMIGs, RMIGs_norm]
    X_names = ['JEMMIG', 'RMIG', 'RMIG (normalized)']
    X_file_names = ['JEMMIG', 'RMIG', 'RMIG_norm']

    Y_objs = [Disents, Comps, Errors]
    Y_names = ['Disentanglement', 'Completeness', 'Error']
    Y_file_names = ['Disent', 'Comp', 'Error']

    for i in range(len(X_objs)):
        for j in range(len(Y_objs)):

            plt.scatter(X_objs[i], Y_objs[j], s=100, color=colors, marker="o", alpha=0.3)
            for k in range(len(X_objs[i])):
                if X_names[i] == 'JEMMIG' and Y_names[j] == "Disentanglement":
                    # if k!= 1 and k != 26 and k != 24 and k != 29 and k!= 31 and k!=35 and k!= 33:
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'JEMMIG' and Y_names[j] == "Completeness":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG' and Y_names[j] == "Disentanglement":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG' and Y_names[j] == "Completeness":
                    if k == 0 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                    elif k == 7:
                        plt.text(X_objs[i][k], Y_objs[j][k] * 0.95, labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG (normalized)' and Y_names[j] == "Disentanglement":
                    if k == 0 or k == 7 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                elif X_names[i] == 'RMIG (normalized)' and Y_names[j] == "Completeness":
                    if k == 0 or k == 8 or k == 10 or k == 12 or k == 14 or \
                       k == 17 or k == 18 or k == 21 or k == 28 or k == 32:
                        plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')
                    elif k == 7:
                        plt.text(X_objs[i][k], Y_objs[j][k] * 0.95, labels[k], fontsize=12, ha='center')
                elif Y_names[j] == "Error":
                    pass
                else:
                    plt.text(X_objs[i][k], Y_objs[j][k], labels[k], fontsize=12, ha='center')

            plt.xlabel(X_names[i])
            plt.ylabel(Y_names[j])
            if Y_names[j] == "Error":
                subplot_adjust = {'left': 0.15, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
            else:
                subplot_adjust = {'left': 0.13, 'right': 0.98, 'bottom': 0.14, 'top': 0.99}
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


def plot_Disentanglement_tc_beta(save_dir, metrics_Eastwood_result_files, labels):
    Disents_all = []
    Disents_by_tc = {}
    Disents_by_beta = {}
    Disents_by_Gz = {}

    for i in range(len(metrics_Eastwood_result_files)):
        results = np.load(metrics_Eastwood_result_files[i], "r")
        disent = results['disentanglement']
        Disents_all.append(disent)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                disent_list = Disents_by_Gz.get(Gz)
                if disent_list is None:
                    Disents_by_Gz[Gz] = [disent]
                else:
                    disent_list.append(disent)

            else:
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                beta = int(labels[i][idx + len('beta'):])
                disent_list = Disents_by_beta.get(beta)
                if disent_list is None:
                    Disents_by_beta[beta] = [disent]
                else:
                    disent_list.append(disent)
        else:
            tc = int(labels[i][idx + len('tc'):])
            disent_list = Disents_by_tc.get(tc)
            if disent_list is None:
                Disents_by_tc[tc] = [disent]
            else:
                disent_list.append(disent)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(Disents_by_tc)]
    Disents_mean_by_tc = [(tc, np.mean(disent_list)) for tc, disent_list in iteritems(Disents_by_tc)]
    Disents_std_by_tc = [(tc, np.std(disent_list)) for tc, disent_list in iteritems(Disents_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(Disents_by_beta)]
    Disents_mean_by_beta = [(beta, np.mean(disent_list)) for beta, disent_list in iteritems(Disents_by_beta)]
    Disents_std_by_beta = [(beta, np.std(disent_list)) for beta, disent_list in iteritems(Disents_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(Disents_by_Gz)]
    Disents_mean_by_Gz = [(Gz, np.mean(disent_list)) for Gz, disent_list in iteritems(Disents_by_Gz)]
    Disents_std_by_Gz = [(Gz, np.std(disent_list)) for Gz, disent_list in iteritems(Disents_by_Gz)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in Disents_mean_by_tc],
            yerr=[a[1] for a in Disents_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in Disents_mean_by_beta],
            yerr=[a[1] for a in Disents_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in Disents_mean_by_Gz],
            yerr=[a[1] for a in Disents_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("Disentanglement")
    # plt.ylim(bottom=0.5, top=1.0)
    plt.ylim(bottom=0.0, top=1.0)

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(9, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "Disent_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_Completeness_tc_beta(save_dir, metrics_Eastwood_result_files, labels):
    Comps_all = []
    Comps_by_tc = {}
    Comps_by_beta = {}
    Comps_by_Gz = {}

    for i in range(len(metrics_Eastwood_result_files)):
        results = np.load(metrics_Eastwood_result_files[i], "r")
        comp = results['completeness']
        Comps_all.append(comp)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                comp_list = Comps_by_Gz.get(Gz)
                if comp_list is None:
                    Comps_by_Gz[Gz] = [comp]
                else:
                    comp_list.append(comp)
            else:
                beta = int(labels[i][idx + len('beta'):])
                comp_list = Comps_by_beta.get(beta)
                if comp_list is None:
                    Comps_by_beta[beta] = [comp]
                else:
                    comp_list.append(comp)
        else:
            tc = int(labels[i][idx + len('tc'):])
            comp_list = Comps_by_tc.get(tc)
            if comp_list is None:
                Comps_by_tc[tc] = [comp]
            else:
                comp_list.append(comp)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(Comps_by_tc)]
    Comps_mean_by_tc = [(tc, np.mean(comp_list)) for tc, comp_list in iteritems(Comps_by_tc)]
    Comps_std_by_tc = [(tc, np.std(comp_list)) for tc, comp_list in iteritems(Comps_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(Comps_by_beta)]
    Comps_mean_by_beta = [(beta, np.mean(comp_list)) for beta, comp_list in iteritems(Comps_by_beta)]
    Comps_std_by_beta = [(beta, np.std(comp_list)) for beta, comp_list in iteritems(Comps_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(Comps_by_Gz)]
    Comps_mean_by_Gz = [(Gz, np.mean(comp_list)) for Gz, comp_list in iteritems(Comps_by_Gz)]
    Comps_std_by_Gz = [(Gz, np.std(comp_list)) for Gz, comp_list in iteritems(Comps_by_Gz)]

    print("Comps_by_Gz: {}".format(Comps_by_Gz))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in Comps_mean_by_tc],
            yerr=[a[1] for a in Comps_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in Comps_mean_by_beta],
            yerr=[a[1] for a in Comps_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in Comps_mean_by_Gz],
            yerr=[a[1] for a in Comps_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("Completeness")
    # plt.ylim(bottom=0.4, top=1.0)

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(9, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "Comp_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_Error_tc_beta(save_dir, metrics_Eastwood_result_files, labels):
    Errors_all = []
    Errors_by_tc = {}
    Errors_by_beta = {}

    for i in range(len(metrics_Eastwood_result_files)):
        results = np.load(metrics_Eastwood_result_files[i], "r")
        error = results['train_avg_error']
        Errors_all.append(error)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            error_list = Errors_by_beta.get(beta)
            if error_list is None:
                Errors_by_beta[beta] = [error]
            else:
                error_list.append(error)
        else:
            tc = int(labels[i][idx + len('tc'):])
            error_list = Errors_by_tc.get(tc)
            if error_list is None:
                Errors_by_tc[tc] = [error]
            else:
                error_list.append(error)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(Errors_by_tc)]
    Errors_mean_by_tc = [(tc, np.mean(error_list)) for tc, error_list in iteritems(Errors_by_tc)]
    Errors_std_by_tc = [(tc, np.std(error_list)) for tc, error_list in iteritems(Errors_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(Errors_by_beta)]
    Errors_mean_by_beta = [(beta, np.mean(error_list)) for beta, error_list in iteritems(Errors_by_beta)]
    Errors_std_by_beta = [(beta, np.std(error_list)) for beta, error_list in iteritems(Errors_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in Errors_mean_by_tc],
            yerr=[a[1] for a in Errors_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in Errors_mean_by_beta],
            yerr=[a[1] for a in Errors_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("Error")
    plt.ylim(bottom=0.3, top=0.7)

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "Error_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


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
    # labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]
    labels = []
    for i, run_id in enumerate(run_ids):
        if 'VAE' in run_id:
            labels.append("{}_".format(i) + run_id[run_id.rfind('_') + 1:])
        else:
            labels.append("{}_".format(i) + run_id[run_id.find('_') + 1: run_id.find('_') + 5])

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary", "metrics_Eastwood_plot"))

    # Metrics Eastwood result files
    # =========================================== #
    metrics_Eastwood_result_files = []

    continuous_only = True
    LASSO_alpha = 0.002
    LASSO_iters = 10000

    for run_id in run_ids:
        if 'VAE' in run_id:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary",
                                   "metrics_Eastwood", "{}_{}".format(enc_dec_model, run_id),
                                   "results[LASSO,{},alpha={},iters={}].npz".format(
                                   "cont" if continuous_only else "all", LASSO_alpha, LASSO_iters)
                                  ))
        else:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary",
                                       "metrics_Eastwood", "{}_{}".format(enc_dec_model, run_id),
                                       "results[LASSO,{},alpha={},iters={}].npz".format(
                                           "cont" if continuous_only else "all", LASSO_alpha, LASSO_iters)
                                       ))
        metrics_Eastwood_result_files.append(result_file)
    # =========================================== #

    # JEMMIG result files
    # =========================================== #
    JEMMIG_result_files = []

    num_samples = 10000
    for run_id in run_ids:
        if 'VAE' in run_id:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                                  "auxiliary", "JEMMIG", "{}_{}".format(enc_dec_model, run_id),
                                  "results[num_samples={}].npz".format(num_samples)))
        else:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "AAE",
                                  "auxiliary", "JEMMIG", "{}_{}".format(enc_dec_model, run_id),
                                  "results[num_samples={}].npz".format(num_samples)))
        JEMMIG_result_files.append(result_file)
    # =========================================== #

    '''
    plot_JEMMIG_metricsEastwood_comparison(
        save_dir, JEMMIG_result_files,
        metrics_Eastwood_result_files, labels, factors=[1, 2, 3, 4])
    # '''

    '''
    ids = [4]
    compare_matrices(save_dir,
                     [JEMMIG_result_files[i] for i in ids],
                     [metrics_Eastwood_result_files[i] for i in ids],
                     [labels[i] for i in ids],
                     factors=[1, 2, 3, 4])
    #'''

    # plot_Disentanglement_tc_beta(save_dir, metrics_Eastwood_result_files, labels)
    plot_Completeness_tc_beta(save_dir, metrics_Eastwood_result_files, labels)
    # plot_Error_tc_beta(save_dir, metrics_Eastwood_result_files, labels)


if __name__ == "__main__":
    main()
