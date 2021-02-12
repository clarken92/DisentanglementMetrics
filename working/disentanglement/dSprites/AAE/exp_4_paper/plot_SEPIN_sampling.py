from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

# from my_utils.python_utils.functions import softmax_np
from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


# Compare the H_zi computed by SEP (with sampling) and what computed by MISJED (with quantization)
def compare_H_zi(SEP_result_files, INFO_result_files, bin_width):
    # Plot comparison between rmig and mig_chen
    assert len(SEP_result_files) == len(INFO_result_files), \
        "len(SEP_result_files)={}, len(MISJED_result_files)={}".format(
            len(SEP_result_files), len(INFO_result_files))

    for i in range(len(SEP_result_files)):
        SEP_results = np.load(SEP_result_files[i], "r")
        INFO_results = np.load(INFO_result_files[i], "r")

        # (z_dim,)
        H_zi_sampling = SEP_results['H_zi']
        # (z_dim,)
        H_zi_cond_x_sampling = SEP_results['H_zi_cond_x']
        # (z_dim,)
        MI_zi_x_sampling = SEP_results['MI_zi_x']

        # (z_dim,)
        H_zi_quantized = INFO_results['H_z']
        # (z_dim,)
        H_zi_cond_x_quantized = INFO_results['H_z_cond_x']
        # (z_dim,)
        MI_zi_x_quantized = INFO_results['MI_z_x']

        print("H_zi_sampling: {}".format(H_zi_sampling))
        print("MI_zi_x_sampling: {}".format(MI_zi_x_sampling))
        print("H_zi_quantized: {}".format(H_zi_quantized))
        print("MI_zi_x_quantized: {}".format(MI_zi_x_quantized))

        # Plotting H(zi) via sampling and quantization
        # =========================================== #
        font = {'family': 'normal',
                'size': 16}

        matplotlib.rc('font', **font)
        width = 0.2

        fig, axes = plt.subplots(2, 3)

        # Compare H_zi
        # ---------------------------------- #
        ax = axes[0][0]
        ax.bar(np.arange(0, len(H_zi_sampling)) - width, H_zi_sampling, width=width, color='b', align='center')
        ax.bar(np.arange(0, len(H_zi_quantized)), H_zi_quantized, width=width, color='g', align='center')

        ax.set_xlabel("zi")
        ax.set_ylabel("H(zi)")
        # ---------------------------------- #

        # Compare H_zi_cond_x
        # ---------------------------------- #
        ax = axes[0][1]
        ax.bar(np.arange(0, len(H_zi_cond_x_sampling)) - width, H_zi_cond_x_sampling,
               width=width, color='b', align='center')
        ax.bar(np.arange(0, len(H_zi_cond_x_quantized)), H_zi_cond_x_quantized,
               width=width, color='g', align='center')

        ax.set_xlabel("zi")
        ax.set_ylabel("H(zi|x)")
        # ---------------------------------- #

        # Compare MI_zi_x
        # ---------------------------------- #
        ax = axes[0][2]
        ax.bar(np.arange(0, len(MI_zi_x_sampling)) - width, MI_zi_x_sampling, width=width, color='b', align='center')
        ax.bar(np.arange(0, len(MI_zi_x_quantized)), MI_zi_x_quantized, width=width, color='g', align='center')

        ax.set_xlabel("zi")
        ax.set_ylabel("MI(zi, x)")
        # ---------------------------------- #

        # Chuan CMNR
        # Compare H_zi
        # ---------------------------------- #
        ax = axes[1][0]
        ax.bar(np.arange(0, len(H_zi_sampling)) - width, H_zi_sampling - np.log(bin_width), width=width, color='b', align='center')
        ax.bar(np.arange(0, len(H_zi_quantized)), H_zi_quantized, width=width, color='g', align='center')

        ax.set_xlabel("zi")
        ax.set_ylabel("H(zi) - log(bin_width)")
        # ---------------------------------- #

        # Compare H_zi_cond_x
        # ---------------------------------- #
        ax = axes[1][1]
        ax.bar(np.arange(0, len(H_zi_cond_x_sampling)) - width, H_zi_cond_x_sampling - np.log(bin_width),
               width=width, color='b', align='center')
        ax.bar(np.arange(0, len(H_zi_cond_x_quantized)), H_zi_cond_x_quantized,
               width=width, color='g', align='center')

        ax.set_xlabel("zi")
        ax.set_ylabel("H(zi|x) - log(bin_width)")
        # ---------------------------------- #

        plt.show()
        plt.close()
        # =========================================== #


'''
def plot_SEP(save_dir, SEP_result_files, labels):
    SEPs = []

    for i in range(len(SEP_result_files)):
        SEP_results = np.load(SEP_result_files[i], "r")
        SEPs.append(np.mean(SEP_results['SEP']))

    SEPs = np.asarray(SEPs, dtype=np.float32)

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    plt.bar(labels, SEPs, alpha=0.3)

    plt.xlabel("model")
    plt.ylabel("SEP")
    plt.tight_layout()

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "SEP.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #
'''

def plot_SEP_with_latents(save_dir, SEP_result_files, num_plots, labels):
    assert num_plots <= len(SEP_result_files), "num_plots={} but len(SEP_result_files)={}".format(
        num_plots, len(SEP_result_files))

    SEPs_zi = []
    for i in range(num_plots):
        SEP_results = np.load(SEP_result_files[i], "r")
        SEPs_zi.append(SEP_results['SEP_zi'])

    SEPs_zi = np.asarray(SEPs_zi, dtype=np.float32)

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal',
            'size': 16}

    matplotlib.rc('font', **font)

    fig, axes = plt.subplots(1, num_plots, sharey=True)
    width = 0.5
    for i in range(num_plots):
        axes[i].bar(np.arange(SEPs_zi.shape[1]), SEPs_zi[i], width=width, color='b', align='center')
        axes[i].set_xlabel(labels[i])

    axes[0].set_ylabel("SEP(zi)")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.1, 'right': 0.985, 'bottom': 0.22, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)

    fig.set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "SEP_zi.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_NewSEP_tc_beta(save_dir, SEP_result_files, labels):
    WSEPs_all = []
    WSEPs_by_tc = {}         # SEPs_by_tc
    WSEPs_by_beta = {}       # SEPs_by_beta

    np.set_printoptions(precision=4, threshold=np.nan, linewidth=1000)
    for i in range(len(SEP_result_files)):
        print("\n\n=======================")
        print("{}".format(labels[i]))
        SEP_results = np.load(SEP_result_files[i], "r")

        MI_zi_x = np.maximum(SEP_results['MI_zi_x'], 0)
        MI_zi_zj = np.maximum(SEP_results['MI_zi_zj'], 0)
        for k in range(len(MI_zi_zj)):
            MI_zi_zj[k, k] = 0
        sep_coeff = MI_zi_zj / np.sum(MI_zi_zj, axis=-1, keepdims=True)
        # sep_coeff = np.exp(np.log(MI_zi_zj) - np.log(np.sum(MI_zi_zj, axis=-1, keepdims=True)))
        print("MI_zi_x: {}".format(MI_zi_x))
        print("MI_zi_zj:\n{}".format(MI_zi_zj))
        print("sep_coeff:\n{}".format(sep_coeff))
        print("sum(sep_coeff): {}".format(np.sum(sep_coeff, axis=-1)))
        print("np.sum(MI_zi_zj * sep_coeff):\n{}".format(np.sum(MI_zi_zj * sep_coeff, axis=-1)))

        WSEP_zi = MI_zi_x - np.sum(MI_zi_zj * sep_coeff, axis=-1)
        info_coeff = MI_zi_x / np.sum(MI_zi_x, axis=0, keepdims=True)
        print("WSEP_zi: {}".format(WSEP_zi))
        print("info_coeff: {}".format(info_coeff))
        print("sum(info_coeff): {}".format(np.sum(info_coeff, axis=-1)))

        WSEP = np.sum(WSEP_zi * info_coeff)

        WSEPs_all.append(WSEP)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            wsep_list = WSEPs_by_beta.get(beta)
            if wsep_list is None:
                WSEPs_by_beta[beta] = [WSEP]
            else:
                wsep_list.append(WSEP)
        else:
            tc = int(labels[i][idx + len('tc'):])
            wsep_list = WSEPs_by_tc.get(tc)
            if wsep_list is None:
                WSEPs_by_tc[tc] = [WSEP]
            else:
                wsep_list.append(WSEP)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(WSEPs_by_tc)]
    WSEPs_mean_by_tc = [(tc, np.mean(wsep_list)) for tc, wsep_list in iteritems(WSEPs_by_tc)]
    WSEPs_std_by_tc = [(tc, np.std(wsep_list)) for tc, wsep_list in iteritems(WSEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(WSEPs_by_beta)]
    WSEPs_mean_by_beta = [(beta, np.mean(wsep_list)) for beta, wsep_list in iteritems(WSEPs_by_beta)]
    WSEPs_std_by_beta = [(beta, np.std(wsep_list)) for beta, wsep_list in iteritems(WSEPs_by_beta)]

    print("WSEPs_all: {}".format(WSEPs_all))

    print("WSEPs_by_tc: {}".format(WSEPs_by_tc))
    print("WSEPs_mean_by_tc: {}".format(WSEPs_mean_by_tc))
    print("WSEPs_std_by_tc: {}".format(WSEPs_std_by_tc))

    print("WSEPs_by_beta: {}".format(WSEPs_by_beta))
    print("WSEPs_mean_by_beta: {}".format(WSEPs_mean_by_beta))
    print("WSEPs_std_by_beta: {}".format(WSEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in WSEPs_mean_by_tc],
            yerr=[a[1] for a in WSEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in WSEPs_mean_by_beta],
            yerr=[a[1] for a in WSEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("NewSEP")
    # plt.tight_layout()

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "NewSEP_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #

'''
def inspect_SEPIN(SEPIN_result_file):
    np.set_printoptions(suppress=True, precision=4, threshold=np.nan, linewidth=1000)

    SEPIN_results = np.load(SEPIN_result_file, "r")

    MI_zi_x = SEPIN_results['MI_zi_x']
    MI_z_x = SEPIN_results['MI_z_x']
    MI_z_not_i_x = SEPIN_results['MI_z_not_i_x']
    coeff_zi = MI_zi_x / np.sum(MI_zi_x)
    SEP_zi = SEPIN_results['SEP_zi']

    SEPIN_zi = SEPIN_results['SEPIN_zi']
    WSEPIN = SEPIN_results['WSEPIN']
    SEPIN = np.mean(SEPIN_zi, axis=0)

    INDIN_zi = SEPIN_results['INDIN_zi']
    WINDIN = SEPIN_results['WINDIN']
    INDIN = np.mean(INDIN_zi, axis=0)

    print("MI_zi_x: {}".format(MI_zi_x))
    print("MI_z_x: {}".format(MI_z_x))
    print("MI_z_not_i_x: {}".format(MI_z_not_i_x))
    print("SEP_zi: {}".format(SEP_zi))
    print("\nSEPIN_zi: {}".format(SEPIN_zi))
    print("\nINDIN_zi: {}".format(INDIN_zi))
    print("\ncoeff_zi: {}".format(coeff_zi))
    print("\nWSEPIN: {}".format(WSEPIN))
    print("\nSEPIN: {}".format(SEPIN))
    print("\nWINDIN: {}".format(WINDIN))
    print("\nINDIN: {}".format(INDIN))
'''


def inspect_SEPIN(SEPIN_result_files, labels):
    np.set_printoptions(threshold=np.nan, suppress=True, precision=4, linewidth=1000)
    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")

        H_zi = SEPIN_results['H_zi']
        H_zi_cond_x = SEPIN_results['H_zi_cond_x']
        MI_zi_x = SEPIN_results['MI_zi_x']

        H_z_not_i = SEPIN_results['H_z_not_i']
        H_z_not_i_cond_x = SEPIN_results['H_z_not_i_cond_x']
        MI_z_not_i_x = SEPIN_results['MI_z_not_i_x']

        H_z = SEPIN_results['H_z']
        H_z_cond_x = SEPIN_results['H_z_cond_x']
        MI_z_x = SEPIN_results['MI_z_x']

        SEP_zi = SEPIN_results['SEP_zi']
        SEPIN_zi = SEPIN_results['SEPIN_zi']
        INDIN_zi = SEPIN_results['INDIN_zi']

        WSEPIN = SEPIN_results['WSEPIN']
        WINDIN = SEPIN_results['WINDIN']

        # =========================================== #
        print("\n======================================")
        print("\n{}".format(labels[i]))
        print("\nH_zi: {}".format(H_zi))
        print("\nH_zi_cond_x: {}".format(H_zi_cond_x))
        print("\nMI_zi_x: {}".format(MI_zi_x))
        print("\nH_z_not_i: {}".format(H_z_not_i))
        print("\nH_z_not_i_cond_x: {}".format(H_z_not_i_cond_x))
        print("\nMI_z_not_i_x: {}".format(MI_z_not_i_x))
        print("\nH_z: {}".format(H_z))
        print("\nH_z_cond_x: {}".format(H_z_cond_x))
        print("\nMI_z_x: {}".format(MI_z_x))
        print("\nMI_zi_zni (SEP_zi): {}".format(SEP_zi))
        print("\nMI_z_x - MI_zni_x (SEPIN_zi): {}".format(SEPIN_zi))
        print("\nMI_zi_x - MI_zi_zni (INDIN_zi): {}".format(INDIN_zi))
        print("\nWSEPIN: {}".format(WSEPIN))
        print("\nWINDIN: {}".format(WINDIN))
        # =========================================== #


def plot_MeanInfo_tc_beta(save_dir, SEPIN_result_files, labels):
    INFOs_all = []
    INFOs_by_tc = {}
    INFOs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        INFO = np.mean(SEPIN_results['MI_zi_x'], axis=0)

        INFOs_all.append(INFO)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            info_list = INFOs_by_beta.get(beta)
            if info_list is None:
                INFOs_by_beta[beta] = [INFO]
            else:
                info_list.append(INFO)
        else:
            tc = int(labels[i][idx + len('tc'):])
            info_list = INFOs_by_tc.get(tc)
            if info_list is None:
                INFOs_by_tc[tc] = [INFO]
            else:
                info_list.append(INFO)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(INFOs_by_tc)]
    INFOs_mean_by_tc = [(tc, np.mean(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]
    INFOs_std_by_tc = [(tc, np.std(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(INFOs_by_beta)]
    INFOs_mean_by_beta = [(beta, np.mean(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]
    INFOs_std_by_beta = [(beta, np.std(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]

    print("INFOs_all: {}".format(INFOs_all))

    print("INFOs_by_tc: {}".format(INFOs_by_tc))
    print("INFOs_mean_by_tc: {}".format(INFOs_mean_by_tc))
    print("INFOs_std_by_tc: {}".format(INFOs_std_by_tc))

    print("INFOs_by_beta: {}".format(INFOs_by_beta))
    print("INFOs_mean_by_beta: {}".format(INFOs_mean_by_beta))
    print("INFOs_std_by_beta: {}".format(INFOs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in INFOs_mean_by_tc],
            yerr=[a[1] for a in INFOs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in INFOs_mean_by_beta],
            yerr=[a[1] for a in INFOs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("mean I(zi, x)")

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MeanInfo_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_MaxInfo_tc_beta(save_dir, SEPIN_result_files, labels):
    INFOs_all = []
    INFOs_by_tc = {}
    INFOs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        INFO = np.max(SEPIN_results['MI_zi_x'], axis=0)

        INFOs_all.append(INFO)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            info_list = INFOs_by_beta.get(beta)
            if info_list is None:
                INFOs_by_beta[beta] = [INFO]
            else:
                info_list.append(INFO)
        else:
            tc = int(labels[i][idx + len('tc'):])
            info_list = INFOs_by_tc.get(tc)
            if info_list is None:
                INFOs_by_tc[tc] = [INFO]
            else:
                info_list.append(INFO)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(INFOs_by_tc)]
    INFOs_mean_by_tc = [(tc, np.mean(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]
    INFOs_std_by_tc = [(tc, np.std(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(INFOs_by_beta)]
    INFOs_mean_by_beta = [(beta, np.mean(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]
    INFOs_std_by_beta = [(beta, np.std(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]

    print("INFOs_all: {}".format(INFOs_all))

    print("INFOs_by_tc: {}".format(INFOs_by_tc))
    print("INFOs_mean_by_tc: {}".format(INFOs_mean_by_tc))
    print("INFOs_std_by_tc: {}".format(INFOs_std_by_tc))

    print("INFOs_by_beta: {}".format(INFOs_by_beta))
    print("INFOs_mean_by_beta: {}".format(INFOs_mean_by_beta))
    print("INFOs_std_by_beta: {}".format(INFOs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in INFOs_mean_by_tc],
            yerr=[a[1] for a in INFOs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in INFOs_mean_by_beta],
            yerr=[a[1] for a in INFOs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("max I(zi, x)")

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MaxInfo_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_AllInfo_tc_beta(save_dir, SEPIN_result_files, labels):
    INFOs_all = []
    INFOs_by_tc = {}
    INFOs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        INFO = np.mean(SEPIN_results['MI_z_x'], axis=0)

        INFOs_all.append(INFO)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            info_list = INFOs_by_beta.get(beta)
            if info_list is None:
                INFOs_by_beta[beta] = [INFO]
            else:
                info_list.append(INFO)
        else:
            tc = int(labels[i][idx + len('tc'):])
            info_list = INFOs_by_tc.get(tc)
            if info_list is None:
                INFOs_by_tc[tc] = [INFO]
            else:
                info_list.append(INFO)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(INFOs_by_tc)]
    INFOs_mean_by_tc = [(tc, np.mean(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]
    INFOs_std_by_tc = [(tc, np.std(info_list)) for tc, info_list in iteritems(INFOs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(INFOs_by_beta)]
    INFOs_mean_by_beta = [(beta, np.mean(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]
    INFOs_std_by_beta = [(beta, np.std(info_list)) for beta, info_list in iteritems(INFOs_by_beta)]

    print("INFOs_all: {}".format(INFOs_all))

    print("INFOs_by_tc: {}".format(INFOs_by_tc))
    print("INFOs_mean_by_tc: {}".format(INFOs_mean_by_tc))
    print("INFOs_std_by_tc: {}".format(INFOs_std_by_tc))

    print("INFOs_by_beta: {}".format(INFOs_by_beta))
    print("INFOs_mean_by_beta: {}".format(INFOs_mean_by_beta))
    print("INFOs_std_by_beta: {}".format(INFOs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in INFOs_mean_by_tc],
            yerr=[a[1] for a in INFOs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in INFOs_mean_by_beta],
            yerr=[a[1] for a in INFOs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("I(z,x)")

    subplot_adjust = {'left': 0.12, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "AllInfo_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_MinSep_tc_beta(save_dir, SEPIN_result_files, labels):
    SEPs_all = []
    SEPs_by_tc = {}
    SEPs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        SEP = np.min(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)

        SEPs_all.append(SEP)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            sep_list = SEPs_by_beta.get(beta)
            if sep_list is None:
                SEPs_by_beta[beta] = [SEP]
            else:
                sep_list.append(SEP)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPs_by_tc.get(tc)
            if sep_list is None:
                SEPs_by_tc[tc] = [SEP]
            else:
                sep_list.append(SEP)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPs_by_tc)]
    SEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]
    SEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPs_by_beta)]
    SEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]
    SEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]

    print("SEPs_all: {}".format(SEPs_all))

    print("SEPs_by_tc: {}".format(SEPs_by_tc))
    print("SEPs_mean_by_tc: {}".format(SEPs_mean_by_tc))
    print("SEPs_std_by_tc: {}".format(SEPs_std_by_tc))

    print("SEPs_by_beta: {}".format(SEPs_by_beta))
    print("SEPs_mean_by_beta: {}".format(SEPs_mean_by_beta))
    print("SEPs_std_by_beta: {}".format(SEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPs_mean_by_tc],
            yerr=[a[1] for a in SEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPs_mean_by_beta],
            yerr=[a[1] for a in SEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("min I(zi,z_ni)")

    subplot_adjust = {'left': 0.13, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MinSep_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_MaxSep_tc_beta(save_dir, SEPIN_result_files, labels):
    SEPs_all = []
    SEPs_by_tc = {}
    SEPs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        SEP = np.max(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)

        SEPs_all.append(SEP)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            sep_list = SEPs_by_beta.get(beta)
            if sep_list is None:
                SEPs_by_beta[beta] = [SEP]
            else:
                sep_list.append(SEP)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPs_by_tc.get(tc)
            if sep_list is None:
                SEPs_by_tc[tc] = [SEP]
            else:
                sep_list.append(SEP)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPs_by_tc)]
    SEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]
    SEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPs_by_beta)]
    SEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]
    SEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]

    print("SEPs_all: {}".format(SEPs_all))

    print("SEPs_by_tc: {}".format(SEPs_by_tc))
    print("SEPs_mean_by_tc: {}".format(SEPs_mean_by_tc))
    print("SEPs_std_by_tc: {}".format(SEPs_std_by_tc))

    print("SEPs_by_beta: {}".format(SEPs_by_beta))
    print("SEPs_mean_by_beta: {}".format(SEPs_mean_by_beta))
    print("SEPs_std_by_beta: {}".format(SEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPs_mean_by_tc],
            yerr=[a[1] for a in SEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPs_mean_by_beta],
            yerr=[a[1] for a in SEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("max I(zi,z_ni)")

    subplot_adjust = {'left': 0.10, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MaxSep_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_MeanSep_tc_beta(save_dir, SEPIN_result_files, labels):
    SEPs_all = []
    SEPs_by_tc = {}
    SEPs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        SEP = np.mean(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)

        SEPs_all.append(SEP)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            sep_list = SEPs_by_beta.get(beta)
            if sep_list is None:
                SEPs_by_beta[beta] = [SEP]
            else:
                sep_list.append(SEP)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPs_by_tc.get(tc)
            if sep_list is None:
                SEPs_by_tc[tc] = [SEP]
            else:
                sep_list.append(SEP)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPs_by_tc)]
    SEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]
    SEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPs_by_beta)]
    SEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]
    SEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]

    print("SEPs_all: {}".format(SEPs_all))

    print("SEPs_by_tc: {}".format(SEPs_by_tc))
    print("SEPs_mean_by_tc: {}".format(SEPs_mean_by_tc))
    print("SEPs_std_by_tc: {}".format(SEPs_std_by_tc))

    print("SEPs_by_beta: {}".format(SEPs_by_beta))
    print("SEPs_mean_by_beta: {}".format(SEPs_mean_by_beta))
    print("SEPs_std_by_beta: {}".format(SEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPs_mean_by_tc],
            yerr=[a[1] for a in SEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPs_mean_by_beta],
            yerr=[a[1] for a in SEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("mean I(zi,z_ni)")

    subplot_adjust = {'left': 0.11, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MeanSep_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_MaxMeanMinSep_tc_beta(save_dir, SEPIN_result_files, labels):
    MeanSEPs_all = []
    MeanSEPs_by_tc = {}
    MeanSEPs_by_beta = {}

    MaxSEPs_all = []
    MaxSEPs_by_tc = {}
    MaxSEPs_by_beta = {}

    MinSEPs_all = []
    MinSEPs_by_tc = {}
    MinSEPs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        SEP_mean = np.mean(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)
        SEP_max = np.max(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)
        SEP_min = np.min(np.maximum(SEPIN_results['SEP_zi'], 0), axis=0)

        MeanSEPs_all.append(SEP_mean)
        MaxSEPs_all.append(SEP_max)
        MinSEPs_all.append(SEP_min)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            mean_sep_list = MeanSEPs_by_beta.get(beta)
            max_sep_list = MaxSEPs_by_beta.get(beta)
            min_sep_list = MinSEPs_by_beta.get(beta)

            if mean_sep_list is None:
                MeanSEPs_by_beta[beta] = [SEP_mean]
                MaxSEPs_by_beta[beta] = [SEP_max]
                MinSEPs_by_beta[beta] = [SEP_min]
            else:
                mean_sep_list.append(SEP_mean)
                max_sep_list.append(SEP_max)
                min_sep_list.append(SEP_min)

        else:
            tc = int(labels[i][idx + len('tc'):])
            mean_sep_list = MeanSEPs_by_tc.get(tc)
            max_sep_list = MaxSEPs_by_tc.get(tc)
            min_sep_list = MinSEPs_by_tc.get(tc)

            if mean_sep_list is None:
                MeanSEPs_by_tc[tc] = [SEP_mean]
                MaxSEPs_by_tc[tc] = [SEP_max]
                MinSEPs_by_tc[tc] = [SEP_min]

            else:
                mean_sep_list.append(SEP_mean)
                max_sep_list.append(SEP_max)
                min_sep_list.append(SEP_min)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(MeanSEPs_by_tc)]
    MeanSEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(MeanSEPs_by_tc)]
    MeanSEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(MeanSEPs_by_tc)]

    MaxSEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(MaxSEPs_by_tc)]
    MaxSEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(MaxSEPs_by_tc)]

    MinSEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(MinSEPs_by_tc)]
    MinSEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(MinSEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(MeanSEPs_by_beta)]
    MeanSEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(MeanSEPs_by_beta)]
    MeanSEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(MeanSEPs_by_beta)]

    MaxSEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(MaxSEPs_by_beta)]
    MaxSEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(MaxSEPs_by_beta)]

    MinSEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(MinSEPs_by_beta)]
    MinSEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(MinSEPs_by_beta)]

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.25
    tc_color = 'blue'
    beta_color = 'orange'
    max_pattern = 'x'
    min_pattern = '/'

    plt.bar(np.arange(0, len(tc_list)), [a[1] for a in MaxSEPs_mean_by_tc],
            yerr=[a[1] for a in MaxSEPs_std_by_tc], width=width, align='center',
            color=tc_color, hatch=max_pattern, edgecolor='black')
    plt.bar(np.arange(0, len(tc_list)) + width, [a[1] for a in MeanSEPs_mean_by_tc],
            yerr=[a[1] for a in MeanSEPs_std_by_tc], width=width, align='center',
            color=tc_color, label="TC", edgecolor='black')
    plt.bar(np.arange(0, len(tc_list)) + 2 * width, [a[1] for a in MinSEPs_mean_by_tc],
            yerr=[a[1] for a in MinSEPs_std_by_tc], width=width, align='center',
            color=tc_color, hatch=min_pattern, edgecolor='black')

    plt.bar(np.arange(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in MaxSEPs_mean_by_beta],
            yerr=[a[1] for a in MaxSEPs_std_by_beta], width=width, align='center',
            color=beta_color, hatch=max_pattern, edgecolor='black')
    plt.bar(np.arange(len(tc_list), len(beta_list) + len(tc_list)) + width, [a[1] for a in MeanSEPs_mean_by_beta],
            yerr=[a[1] for a in MeanSEPs_std_by_beta], width=width, align='center',
            color=beta_color, label="Beta", edgecolor='black')
    plt.bar(np.arange(len(tc_list), len(beta_list) + len(tc_list)) + 2 * width, [a[1] for a in MinSEPs_mean_by_beta],
            yerr=[a[1] for a in MinSEPs_std_by_beta], width=width, align='center',
            color=beta_color, hatch=min_pattern, edgecolor='black')

    plt.xticks(np.arange(0, len(tc_list) + len(beta_list)) + 0.5 * width, tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("max/mean/min I(zi,z_ni)")

    subplot_adjust = {'left': 0.11, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MaxMeanMinSep_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_cond_MI_zi_zni(save_dir, SEPIN_result_files, labels):
    SEPs_all = []
    SEPs_by_tc = {}
    SEPs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        H_zi_cond_x = SEPIN_results['H_zi_cond_x']
        H_z_not_i_cond_x = SEPIN_results['H_z_not_i_cond_x']
        H_z_cond_x = SEPIN_results['H_z_cond_x']
        MI_zi_zni_cond_x = np.mean(H_zi_cond_x + H_z_not_i_cond_x - H_z_cond_x, axis=0)

        SEPs_all.append(MI_zi_zni_cond_x)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            sep_list = SEPs_by_beta.get(beta)
            if sep_list is None:
                SEPs_by_beta[beta] = [MI_zi_zni_cond_x]
            else:
                sep_list.append(MI_zi_zni_cond_x)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPs_by_tc.get(tc)
            if sep_list is None:
                SEPs_by_tc[tc] = [MI_zi_zni_cond_x]
            else:
                sep_list.append(MI_zi_zni_cond_x)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPs_by_tc)]
    SEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]
    SEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPs_by_beta)]
    SEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]
    SEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]

    print("SEPs_all: {}".format(SEPs_all))

    print("SEPs_by_tc: {}".format(SEPs_by_tc))
    print("SEPs_mean_by_tc: {}".format(SEPs_mean_by_tc))
    print("SEPs_std_by_tc: {}".format(SEPs_std_by_tc))

    print("SEPs_by_beta: {}".format(SEPs_by_beta))
    print("SEPs_mean_by_beta: {}".format(SEPs_mean_by_beta))
    print("SEPs_std_by_beta: {}".format(SEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPs_mean_by_tc],
            yerr=[a[1] for a in SEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPs_mean_by_beta],
            yerr=[a[1] for a in SEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("mean MI(zi,z_ni|x)")

    subplot_adjust = {'left': 0.11, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "MI_zi_zni_cond_x_tc_beta.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_SEP_tc_beta(save_dir, SEPIN_result_files, labels, weighted=True):
    SEPs_all = []
    SEPs_by_tc = {}
    SEPs_by_beta = {}
    SEPs_by_Gz = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")
        if weighted:
            SEP = SEPIN_results['WSEP']
        else:
            SEP = np.mean(SEPIN_results['SEP_zi'], axis=0)

        SEPs_all.append(SEP)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                sep_list = SEPs_by_Gz.get(Gz)
                if sep_list is None:
                    SEPs_by_Gz[Gz] = [SEP]
                else:
                    sep_list.append(SEP)
            else:
                beta = int(labels[i][idx + len('beta'):])
                sep_list = SEPs_by_beta.get(beta)
                if sep_list is None:
                    SEPs_by_beta[beta] = [SEP]
                else:
                    sep_list.append(SEP)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPs_by_tc.get(tc)
            if sep_list is None:
                SEPs_by_tc[tc] = [SEP]
            else:
                sep_list.append(SEP)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPs_by_tc)]
    SEPs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]
    SEPs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPs_by_beta)]
    SEPs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]
    SEPs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPs_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(SEPs_by_Gz)]
    SEPs_mean_by_Gz = [(Gz, np.mean(sep_list)) for Gz, sep_list in iteritems(SEPs_by_Gz)]
    SEPs_std_by_Gz = [(Gz, np.std(sep_list)) for Gz, sep_list in iteritems(SEPs_by_Gz)]

    print("SEPs_all: {}".format(SEPs_all))

    print("SEPs_by_tc: {}".format(SEPs_by_tc))
    print("SEPs_mean_by_tc: {}".format(SEPs_mean_by_tc))
    print("SEPs_std_by_tc: {}".format(SEPs_std_by_tc))

    print("SEPs_by_beta: {}".format(SEPs_by_beta))
    print("SEPs_mean_by_beta: {}".format(SEPs_mean_by_beta))
    print("SEPs_std_by_beta: {}".format(SEPs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPs_mean_by_tc],
            yerr=[a[1] for a in SEPs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPs_mean_by_beta],
            yerr=[a[1] for a in SEPs_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [a[1] for a in SEPs_mean_by_Gz],
            yerr=[a[1] for a in SEPs_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    if weighted:
        plt.ylabel("SEP (weighted)")
    else:
        plt.ylabel("SEP")

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    if weighted:
        save_file = join(save_dir, "WSEP_tc_beta_Gz.pdf")
    else:
        save_file = join(save_dir, "SEP_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_SEPIN_tc_beta(save_dir, SEPIN_result_files, labels, weighted=True):
    SEPINs_all = []
    SEPINs_by_tc = {}
    SEPINs_by_beta = {}
    SEPINs_by_Gz = {}

    for i in range(len(SEPIN_result_files)):
        SEPIN_results = np.load(SEPIN_result_files[i], "r")

        if weighted:
            SEPIN = SEPIN_results['WSEPIN']
        else:
            SEPIN = np.mean(SEPIN_results['SEPIN_zi'])

        SEPINs_all.append(SEPIN)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            if idx < 0:
                idx = labels[i].find('Gz')
                assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

                Gz = int(labels[i][idx + len('Gz'): idx + len('Gz') + 2])
                sep_list = SEPINs_by_Gz.get(Gz)
                if sep_list is None:
                    SEPINs_by_Gz[Gz] = [SEPIN]
                else:
                    sep_list.append(SEPIN)
            else:
                beta = int(labels[i][idx + len('beta'):])
                sep_list = SEPINs_by_beta.get(beta)
                if sep_list is None:
                    SEPINs_by_beta[beta] = [SEPIN]
                else:
                    sep_list.append(SEPIN)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPINs_by_tc.get(tc)
            if sep_list is None:
                SEPINs_by_tc[tc] = [SEPIN]
            else:
                sep_list.append(SEPIN)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPINs_by_tc)]
    SEPINs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPINs_by_tc)]
    SEPINs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPINs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPINs_by_beta)]
    SEPINs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPINs_by_beta)]
    SEPINs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPINs_by_beta)]

    Gz_list = ["{}".format(Gz) for Gz, _ in iteritems(SEPINs_by_Gz)]
    SEPINs_mean_by_Gz = [(Gz, np.mean(sep_list)) for Gz, sep_list in iteritems(SEPINs_by_Gz)]
    SEPINs_std_by_Gz = [(Gz, np.std(sep_list)) for Gz, sep_list in iteritems(SEPINs_by_Gz)]

    print("weighted: {}".format(weighted))
    print("SEPINs_all: {}".format(SEPINs_all))

    print("SEPINs_by_tc: {}".format(SEPINs_by_tc))
    print("SEPINs_mean_by_tc: {}".format(SEPINs_mean_by_tc))
    print("SEPINs_std_by_tc: {}".format(SEPINs_std_by_tc))

    print("SEPINs_by_beta: {}".format(SEPINs_by_beta))
    print("SEPINs_mean_by_beta: {}".format(SEPINs_mean_by_beta))
    print("SEPINs_std_by_beta: {}".format(SEPINs_std_by_beta))

    print("SEPINs_by_Gz: {}".format(SEPINs_by_Gz))
    print("SEPINs_mean_by_Gz: {}".format(SEPINs_mean_by_Gz))
    print("SEPINs_std_by_Gz: {}".format(SEPINs_std_by_Gz))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPINs_mean_by_tc],
            yerr=[a[1] for a in SEPINs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPINs_mean_by_beta],
            yerr=[a[1] for a in SEPINs_std_by_beta], width=width, align='center', label="Beta")
    plt.bar(range(len(beta_list) + len(tc_list), len(beta_list) + len(tc_list) + len(Gz_list)),
            [max(a[1], 0.001) for a in SEPINs_mean_by_Gz],
            yerr=[a[1] for a in SEPINs_std_by_Gz], width=width, align='center', label="Gz")
    plt.xticks(range(0, len(tc_list) + len(beta_list) + len(Gz_list)), tc_list + beta_list + Gz_list)

    plt.legend()
    plt.xlabel("model")
    if weighted:
        plt.ylabel("WSEPIN")
    else:
        plt.ylabel("SEPIN")

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    if weighted:
        save_file = join(save_dir, "WSEPIN_tc_beta_Gz.pdf")
    else:
        save_file = join(save_dir, "SEPIN_tc_beta_Gz.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    # =========================================== #


def plot_SEPINatK_tc_beta(save_dir, SEPIN_result_files, labels, K):
    SEPINatKs_all = []
    SEPINatKs_by_tc = {}
    SEPINatKs_by_beta = {}

    for i in range(len(SEPIN_result_files)):
        SEP_results = np.load(SEPIN_result_files[i], "r")

        SEPIN_zi = SEP_results['SEPIN_zi']
        assert len(SEPIN_zi.shape) == 1
        SEPIN_sorted_zi = np.sort(SEPIN_zi, axis=0)[::-1]
        SEPINatK = np.mean(SEPIN_sorted_zi[0: K])

        SEPINatKs_all.append(SEPINatK)

        idx = labels[i].find('tc')
        if idx < 0:
            idx = labels[i].find('beta')
            assert idx >= 0, "labels[{}]='{}'".format(i, labels[i])

            beta = int(labels[i][idx + len('beta'):])
            sep_list = SEPINatKs_by_beta.get(beta)
            if sep_list is None:
                SEPINatKs_by_beta[beta] = [SEPINatK]
            else:
                sep_list.append(SEPINatK)
        else:
            tc = int(labels[i][idx + len('tc'):])
            sep_list = SEPINatKs_by_tc.get(tc)
            if sep_list is None:
                SEPINatKs_by_tc[tc] = [SEPINatK]
            else:
                sep_list.append(SEPINatK)

    tc_list = ["{}".format(tc) for tc, _ in iteritems(SEPINatKs_by_tc)]
    SEPINatKs_mean_by_tc = [(tc, np.mean(sep_list)) for tc, sep_list in iteritems(SEPINatKs_by_tc)]
    SEPINatKs_std_by_tc = [(tc, np.std(sep_list)) for tc, sep_list in iteritems(SEPINatKs_by_tc)]

    beta_list = ["{}".format(beta) for beta, _ in iteritems(SEPINatKs_by_beta)]
    SEPINatKs_mean_by_beta = [(beta, np.mean(sep_list)) for beta, sep_list in iteritems(SEPINatKs_by_beta)]
    SEPINatKs_std_by_beta = [(beta, np.std(sep_list)) for beta, sep_list in iteritems(SEPINatKs_by_beta)]

    print("SEPINatKs_all: {}".format(SEPINatKs_all))

    print("SEPINatKs_by_tc: {}".format(SEPINatKs_by_tc))
    print("SEPINatKs_mean_by_tc: {}".format(SEPINatKs_mean_by_tc))
    print("SEPINatKs_std_by_tc: {}".format(SEPINatKs_std_by_tc))

    print("SEPINatKs_by_beta: {}".format(SEPINatKs_by_beta))
    print("SEPINatKs_mean_by_beta: {}".format(SEPINatKs_mean_by_beta))
    print("SEPINatKs_std_by_beta: {}".format(SEPINatKs_std_by_beta))

    # Plotting RMIG-MIG relationship
    # =========================================== #
    font = {'family': 'normal', 'size': 12}

    matplotlib.rc('font', **font)

    width = 0.5
    plt.bar(range(0, len(tc_list)), [a[1] for a in SEPINatKs_mean_by_tc],
            yerr=[a[1] for a in SEPINatKs_std_by_tc], width=width, align='center', label="TC")
    plt.bar(range(len(tc_list), len(beta_list) + len(tc_list)), [a[1] for a in SEPINatKs_mean_by_beta],
            yerr=[a[1] for a in SEPINatKs_std_by_beta], width=width, align='center', label="Beta")
    plt.xticks(range(0, len(tc_list) + len(beta_list)), tc_list + beta_list)

    plt.legend()
    plt.xlabel("model")
    plt.ylabel("SEPIN@{}".format(K))
    # plt.tight_layout()

    subplot_adjust = {'left': 0.115, 'right': 0.985, 'bottom': 0.18, 'top': 0.97}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(6, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "SEPINat{}_tc_beta.pdf".format(K))

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

    save_dir = abspath(join(RESULTS_DIR, "dSprites", "AAE", "auxiliary", "SEPIN_plot"))


    # SEPIN result files
    # =========================================== #
    SEPIN_result_files = []

    num_samples = 10000
    for run_id in run_ids:
        if 'VAE' in run_id:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "FactorVAE",
                                       "auxiliary", "SEPIN", "{}_{}".format(enc_dec_model, run_id),
                                       "results[num_samples={}].npz".format(num_samples)))
        else:
            result_file = abspath(join(RESULTS_DIR, "dSprites", "AAE",
                                       "auxiliary", "SEPIN", "{}_{}".format(enc_dec_model, run_id),
                                       "results[num_samples={}].npz".format(num_samples)))
        SEPIN_result_files.append(result_file)
    # =========================================== #

    plot_SEPIN_tc_beta(save_dir, SEPIN_result_files, labels, weighted=True)
    plot_SEP_tc_beta(save_dir, SEPIN_result_files, labels, weighted=True)

if __name__ == "__main__":
    main()
