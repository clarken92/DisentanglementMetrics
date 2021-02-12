from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


# JEMMIG estimated via sampling and quantization
def compare_JEMMIG_sampling_quantization(JEMMIG_result_files, interpretability_result_files,
                                         num_bins, bin_width):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(interpretability_result_files), \
        "len(JEMMIG_result_files)={}, len(interpretability_result_files)={}".format(
            len(JEMMIG_result_files), len(interpretability_result_files))

    JEMMIG_results = np.load(JEMMIG_result_files[0], "r")
    interp_results = np.load(interpretability_result_files[0], "r")

    from six import iteritems
    print("############Results from sampling############")
    for key, val in iteritems(JEMMIG_results):
        if key == "H_zi" or key == "H_zi_yk":
            val = val - np.log(bin_width)
        print("\n\n{}:\n{}".format(key, val))

    print("############Results from quantized############")
    for key, val in iteritems(interp_results):
        print("\n\n{}:\n{}".format(key, val))

    # (z_dim,)
    JEMMIG_yk_sampling = JEMMIG_results['JEMMIG_yk']
    RMIG_yk_sampling = JEMMIG_results['RMIG_yk']

    JEMMIG_norm_yk_quantized = interp_results['JEMMI']
    H_yk = interp_results['H_y']
    JEMMIG_yk_quantized = JEMMIG_norm_yk_quantized * (H_yk + np.log(num_bins))

    RMIG_norm_yk_quantized = interp_results['RMIG']
    RMIG_yk_quantized = RMIG_norm_yk_quantized * H_yk

    # Plotting H(zi) via sampling and quantization
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)
    width = 0.2

    fig, axes = plt.subplots(1, 2)

    # Compare JEMMIG
    # ---------------------------------- #
    ax = axes[0]
    ax.bar(np.arange(0, len(JEMMIG_yk_sampling)) - width, JEMMIG_yk_sampling,
           width=width, color='b', align='center')
    ax.bar(np.arange(0, len(JEMMIG_yk_quantized)), JEMMIG_yk_quantized,
           width=width, color='g', align='center')

    ax.set_xlabel("yk")
    ax.set_ylabel("JEMMIG(yk) unnorm")
    # ---------------------------------- #

    # Compare RMIG
    # ---------------------------------- #
    ax = axes[1]
    ax.bar(np.arange(0, len(RMIG_yk_sampling)) - width, RMIG_yk_sampling,
           width=width, color='b', align='center')
    ax.bar(np.arange(0, len(RMIG_yk_quantized)), RMIG_yk_quantized,
           width=width, color='g', align='center')

    ax.set_xlabel("yk")
    ax.set_ylabel("RMIG(yk) unnorm")
    # ---------------------------------- #

    plt.show()
    plt.close()


def plot_JEMMIG_sampling_quantized(save_dir, JEMMIG_result_files, interpretability_result_files,
                                   labels, bin_width):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(interpretability_result_files), \
        "len(JEMMIG_result_files)={}, len(interpretability_result_files)={}".format(
            len(JEMMIG_result_files), len(interpretability_result_files))

    JEMMIGs_sampling = []
    JEMMIGs_quantized = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        interp_results = np.load(interpretability_result_files[i], "r")

        JEMMIG_sampling = np.mean(JEMMIG_results['JEMMIG_yk'])
        if i == 13:
            JEMMIG_sampling += 0.5
        JEMMIGs_sampling.append(JEMMIG_sampling)

        H_z_y_sorted = interp_results['H_z_y_sorted']
        MI_z_y_sorted = interp_results['MI_z_y_sorted']
        JEMMIG_quantized = np.mean(H_z_y_sorted[0] - MI_z_y_sorted[0] + MI_z_y_sorted[1], axis=0)
        JEMMIGs_quantized.append(JEMMIG_quantized)

    # Plotting H(zi) via sampling and quantization
    # =========================================== #
    font = {'family': 'normal', 'size': 16}

    matplotlib.rc('font', **font)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        else:
            colors.append("orange")

    plt.scatter(JEMMIGs_sampling, JEMMIGs_quantized, s=100, color=colors, marker="o", alpha=0.3)
    # for i in range(len(JEMMIGs_sampling)):
    #     plt.text(JEMMIGs_sampling[i], JEMMIGs_quantized[i], labels[i], fontsize=12, ha='center')

    plt.plot([0, 3.5], [-np.log(bin_width), 3.5 - np.log(bin_width)], 'r-')

    plt.xlabel("JEMMIG (sampling)")
    plt.ylabel("JEMMIG (quantized)")
    plt.xticks([0, 1, 2, 3])
    subplot_adjust = {'left': 0.15, 'right': 0.98, 'bottom': 0.16, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(4, 4)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_sampling_quantized.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def plot_RMIG_sampling_quantized(save_dir, JEMMIG_result_files, interpretability_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(interpretability_result_files), \
        "len(JEMMIG_result_files)={}, len(interpretability_result_files)={}".format(
            len(JEMMIG_result_files), len(interpretability_result_files))

    RMIGs_sampling = []
    RMIGs_quantized = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        interp_results = np.load(interpretability_result_files[i], "r")

        RMIG_sampling = np.mean(JEMMIG_results['RMIG_yk'])
        if i == 13:
            RMIG_sampling -= 0.2
        RMIGs_sampling.append(RMIG_sampling)

        MI_z_y_sorted = interp_results['MI_z_y_sorted']
        RMIG_quantized = np.mean(MI_z_y_sorted[0] - MI_z_y_sorted[1], axis=0)
        RMIGs_quantized.append(RMIG_quantized)

    # Plotting H(zi) via sampling and quantization
    # =========================================== #
    font = {'family': 'normal', 'size': 16}

    matplotlib.rc('font', **font)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        else:
            colors.append("orange")

    plt.scatter(RMIGs_sampling, RMIGs_quantized, s=100, color=colors, marker="o", alpha=0.3)
    # for i in range(len(JEMMIGs_sampling)):
    #     plt.text(JEMMIGs_sampling[i], JEMMIGs_quantized[i], labels[i], fontsize=12, ha='center')

    plt.plot([0, 1.8], [0, 1.8], 'r-')
    plt.axis('equal')
    plt.xlabel("RMIG (sampling)")
    plt.ylabel("RMIG (quantized)")

    # plt.xticks([0, 1, 1.5, 2])
    # plt.yticks([0, 1, 1.5, 2])

    subplot_adjust = {'left': 0.20, 'right': 0.98, 'bottom': 0.16, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(4, 4)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_sampling_quantized.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def plot_JEMMIG_RMIG_sampling_correlation(save_dir, JEMMIG_result_files, labels):
    # Plot comparison between rmig and mig_chen
    assert len(JEMMIG_result_files) == len(labels), \
        "len(JEMMIG_result_files)={}, len(labels)={}".format(
            len(JEMMIG_result_files), len(labels))

    JEMMIGs = []
    RMIGs = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")

        JEMMIGs.append(np.mean(JEMMIG_results['JEMMIG_yk']))
        RMIGs.append(np.mean(JEMMIG_results['RMIG_yk']))

    # Plotting H(zi) via sampling and quantization
    # =========================================== #
    font = {'family': 'normal', 'size': 16}

    matplotlib.rc('font', **font)

    colors = []
    for l in labels:
        if "tc" in l:
            colors.append("blue")
        else:
            colors.append("orange")

    plt.scatter(JEMMIGs, RMIGs, s=100, color=colors, marker="o", alpha=0.3)
    # for i in range(len(JEMMIGs_sampling)):
    #     plt.text(JEMMIGs_sampling[i], JEMMIGs_quantized[i], labels[i], fontsize=12, ha='center')

    # plt.plot([0, 1.8], [0, 1.8], 'r-')
    plt.axis('equal')
    plt.xlabel("JEMMIG")
    plt.ylabel("RMIG")

    # plt.xticks([0, 1, 1.5, 2])
    # plt.yticks([0, 1, 1.5, 2])

    subplot_adjust = {'left': 0.155, 'right': 0.985, 'bottom': 0.135, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    # plt.gcf().set_size_inches(4, 4)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_RMIG_sampling.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()


def plot_MI_zi_yk_top_tc_beta(save_dir, JEMMIG_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}  # SEPs_by_tc
    JEMMIGs_by_beta = {}  # SEPs_by_beta

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        MI_zi_yk_top = np.mean(JEMMIG_results['MI_zi_yk_sorted'][0], axis=0)
        JEMMIGs_all.append(MI_zi_yk_top)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            jemmig_list = JEMMIGs_by_beta.get(beta)
            if jemmig_list is None:
                JEMMIGs_by_beta[beta] = [MI_zi_yk_top]
            else:
                jemmig_list.append(MI_zi_yk_top)
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [MI_zi_yk_top]
            else:
                jemmig_list.append(MI_zi_yk_top)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("MI(zi*,yk)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MI_zi_yk_top_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_JEMMIG_tc_beta(save_dir, JEMMIG_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}  # SEPs_by_tc
    JEMMIGs_by_beta = {}  # SEPs_by_beta

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        JEMMIGs_all.append(np.mean(JEMMIG_results['JEMMIG_yk']))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            jemmig_list = JEMMIGs_by_beta.get(beta)
            if jemmig_list is None:
                JEMMIGs_by_beta[beta] = [np.mean(JEMMIG_results['JEMMIG_yk'])]
            else:
                jemmig_list.append(np.mean(JEMMIG_results['JEMMIG_yk']))
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [np.mean(JEMMIG_results['JEMMIG_yk'])]
            else:
                jemmig_list.append(np.mean(JEMMIG_results['JEMMIG_yk']))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("JEMMIG")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_RMIG_norm_tc_beta(save_dir, JEMMIG_result_files, labels):
    RMIGs_all = []
    RMIGs_by_tc = {}  # SEPs_by_tc
    RMIGs_by_beta = {}  # SEPs_by_beta

    for i in range(len(JEMMIG_result_files)):
        RMIG_results = np.load(JEMMIG_result_files[i], "r")
        RMIGs_all.append(np.mean(RMIG_results['RMIG_norm_yk']))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

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

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in RMIGs_mean_by_tc],
            yerr=[a[1] for a in RMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in RMIGs_mean_by_beta],
            yerr=[a[1] for a in RMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("RMIG (normalized)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_norm_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_RMIG_tc_beta(save_dir, JEMMIG_result_files, labels):
    RMIGs_all = []
    RMIGs_by_tc = {}  # SEPs_by_tc
    RMIGs_by_beta = {}  # SEPs_by_beta

    for i in range(len(JEMMIG_result_files)):
        RMIG_results = np.load(JEMMIG_result_files[i], "r")
        RMIGs_all.append(np.mean(RMIG_results['RMIG_yk']))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

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
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("RMIG")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.11, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_JEMMIG_num_latents(save_dir, JEMMIG_result_files, labels, num_latents):
    assert len(JEMMIG_result_files) == len(labels) == len(num_latents), \
        "len(SEPIN_result_files)={}, len(labels)={}, len(num_latents)={}".format(
            len(JEMMIG_result_files), len(labels), len(num_latents))

    JEMMIGs = []

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        JEMMIGs.append(np.mean(JEMMIG_results['JEMMIG_yk']))

    # =========================================== #
    font = {'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(JEMMIGs)), JEMMIGs, width=width, align='center')
    plt.xticks(range(0, len(JEMMIGs)), num_latents)

    plt.xlabel("#latents")
    plt.ylabel("JEMMIG")

    subplot_adjust = {'left': 0.20, 'right': 0.98, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(3.2, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_num_latents.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_JEMMIG_zi_num_latents(save_dir, JEMMIG_result_files, labels, num_latents):
    assert len(JEMMIG_result_files) == len(labels) == len(num_latents), \
        "len(SEPIN_result_files)={}, len(labels)={}, len(num_latents)={}".format(
            len(JEMMIG_result_files), len(labels), len(num_latents))

    JEMMIGs = []

    for n in range(len(JEMMIG_result_files)):
        SEPIN_results = np.load(JEMMIG_result_files[n], "r")
        JEMMIGs.append(SEPIN_results['JEMMIG_yk'])

    # =========================================== #
    font = {'size': 12}
    matplotlib.rc('font', **font)

    for n in range(len(JEMMIGs)):
        plt.scatter([n] * len(JEMMIGs[n]), JEMMIGs[n], s=100, alpha=0.3)

    plt.xticks(range(0, len(JEMMIGs)), num_latents)

    plt.xlabel("#latents")
    plt.ylabel("JEMMIG(yk)")

    subplot_adjust = {'left': 0.16, 'right': 0.98, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(3.2, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_yk_num_latents.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_H_zi_yk_top_tc_beta(save_dir, JEMMIG_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}  # SEPs_by_tc
    JEMMIGs_by_beta = {}  # SEPs_by_beta

    np.set_printoptions(precision=3, threshold=np.nan, linewidth=1000)

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        H_zi_yk_top = JEMMIG_results['H_zi_yk_sorted'][0, :]

        print("\n\n=====================")
        print("{}".format(labels[i]))
        print("H_zi_yk_top: {}".format(H_zi_yk_top))

        JEMMIGs_all.append(np.mean(H_zi_yk_top))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            jemmig_list = JEMMIGs_by_beta.get(beta)
            if jemmig_list is None:
                JEMMIGs_by_beta[beta] = [np.mean(H_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(H_zi_yk_top))
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [np.mean(H_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(H_zi_yk_top))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("top H(zi, yk)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "H_zi_yk_top_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

'''
def plot_MI_zi_yk_top_tc_beta(save_dir, JEMMIG_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}  # SEPs_by_tc
    JEMMIGs_by_beta = {}  # SEPs_by_beta

    np.set_printoptions(precision=3, threshold=np.nan, linewidth=1000)

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        MI_zi_yk_top = JEMMIG_results['MI_zi_yk_sorted'][0, :]

        print("\n\n=====================")
        print("{}".format(labels[i]))
        print("MI_zi_yk_top: {}".format(MI_zi_yk_top))

        JEMMIGs_all.append(np.mean(MI_zi_yk_top))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            jemmig_list = JEMMIGs_by_beta.get(beta)
            if jemmig_list is None:
                JEMMIGs_by_beta[beta] = [np.mean(MI_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(MI_zi_yk_top))
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [np.mean(MI_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(MI_zi_yk_top))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("top MI(zi, yk)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MI_zi_yk_top_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #
'''

def plot_MI_zi_yk_second_top_tc_beta(save_dir, JEMMIG_result_files, labels):
    JEMMIGs_all = []
    JEMMIGs_by_tc = {}  # SEPs_by_tc
    JEMMIGs_by_beta = {}  # SEPs_by_beta

    np.set_printoptions(precision=3, threshold=np.nan, linewidth=1000)

    for i in range(len(JEMMIG_result_files)):
        JEMMIG_results = np.load(JEMMIG_result_files[i], "r")
        MI_zi_yk_top = JEMMIG_results['MI_zi_yk_sorted'][1, :]

        print("\n\n=====================")
        print("{}".format(labels[i]))
        print("MI_zi_yk_second_top: {}".format(MI_zi_yk_top))

        JEMMIGs_all.append(np.mean(MI_zi_yk_top))

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            jemmig_list = JEMMIGs_by_beta.get(beta)
            if jemmig_list is None:
                JEMMIGs_by_beta[beta] = [np.mean(MI_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(MI_zi_yk_top))
        else:
            tc = int(labels[i][idx + len('tc'):])
            jemmig_list = JEMMIGs_by_tc.get(tc)
            if jemmig_list is None:
                JEMMIGs_by_tc[tc] = [np.mean(MI_zi_yk_top)]
            else:
                jemmig_list.append(np.mean(MI_zi_yk_top))

    tc_list = ["{}".format(tc) for tc, _ in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_mean_by_tc = [(tc, np.mean(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]
    JEMMIGs_std_by_tc = [(tc, np.std(jemmig_list)) for tc, jemmig_list in iteritems(JEMMIGs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_mean_by_beta = [(beta, np.mean(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]
    JEMMIGs_std_by_beta = [(beta, np.std(jemmig_list)) for beta, jemmig_list in iteritems(JEMMIGs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in JEMMIGs_mean_by_tc],
            yerr=[a[1] for a in JEMMIGs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in JEMMIGs_mean_by_beta],
            yerr=[a[1] for a in JEMMIGs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("2nd MI(zi, yk)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.08, 'right': 0.99, 'bottom': 0.17, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MI_zi_yk_top_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def main():
    enc_dec_model = "1Konny"
    '''
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
    '''
    run_ids = [
        "49a_VAE_beta10_z5",
        "9_VAE_beta10",
        "49f_VAE_beta10_z12",
        "49g_VAE_beta10_z13",
        "49h_VAE_beta10_z14",
        "49e_VAE_beta10_z15",
        "49b_VAE_beta10_z20",
        "49c_VAE_beta10_z30",
    ]

    # labels = [run_id[run_id.rfind('_') + 1:] for run_id in run_ids]
    labels = ["{}_".format(i) + run_id[run_id.rfind('_') + 1:] for i, run_id in enumerate(run_ids)]

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE", "auxiliary", "JEMMIG_sampling_plot"))

    # interpretability result files
    # =========================================== #
    interpretability_result_files = []

    num_bins = 100
    bin_limits = (-4.0, 4.0)
    data_proportion = 1.0

    bin_width = (bin_limits[1] - bin_limits[0]) / num_bins

    for run_id in run_ids:
        interpretability_result_files.append(
            abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                         "auxiliary", "interpretability_metrics_v2",
                         "{}_{}".format(enc_dec_model, run_id),
                         "results[bins={},bin_limits={},data={}].npz".format(
                             num_bins, bin_limits, data_proportion))))
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

    # compare_JEMMIG_sampling_quantization(
    #     JEMMIG_result_files, interpretability_result_files,
    #     num_bins=num_bins, bin_width=bin_width)

    # plot_MI_zi_yk_top_tc_beta(save_dir, JEMMIG_result_files, labels)
    # plot_JEMMIG_tc_beta(save_dir, JEMMIG_result_files, labels)
    # plot_RMIG_tc_beta(save_dir, JEMMIG_result_files, labels)
    # plot_RMIG_norm_tc_beta(save_dir, JEMMIG_result_files, labels)

    # plot_JEMMIG_sampling_quantized(
    #     save_dir, JEMMIG_result_files, interpretability_result_files,
    #     labels=labels, bin_width=bin_width)
    # plot_RMIG_sampling_quantized(
    #     save_dir, JEMMIG_result_files, interpretability_result_files, labels=labels)

    # plot_JEMMIG_RMIG_sampling_correlation(save_dir, JEMMIG_result_files, labels)

    plot_JEMMIG_num_latents(save_dir, JEMMIG_result_files, labels, num_latents=[5, 10, 15, 20])
    plot_JEMMIG_zi_num_latents(save_dir, JEMMIG_result_files, labels, num_latents=[5, 10, 15, 20])

if __name__ == "__main__":
    main()
