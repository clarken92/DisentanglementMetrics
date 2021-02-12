from six import iteritems
from os.path import join, abspath

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages

from my_utils.python_utils.general import make_dir_if_not_exist

from global_settings import RESULTS_DIR


def compare_JEMMIG_norm_models(save_dir, result_files, attr_names, num_bins):
    JEMMIGs_norm_by_model = []
    ids_sorted_by_factor = []

    assert len(result_files) == 3, "These files must be the result files of FactorVAE, BetaVAE and AAEs!"

    for i in range(len(result_files)):
        results = np.load(result_files[i], "r")
        print("JEMMIG_results.keys: {}".format(list(results.keys())))

        MI_z_y = results['MI_z_y']
        H_z_y = results['H_z_y']
        H_y = results['H_y_4_diff_z'][:, 0]

        ids_sorted_by_MI = np.argsort(MI_z_y, axis=0)[::-1]
        MI_z_y_sorted = np.take_along_axis(MI_z_y, ids_sorted_by_MI, axis=0)
        H_z_y_sorted = np.take_along_axis(H_z_y, ids_sorted_by_MI, axis=0)

        H_diff = H_z_y_sorted[0, :] - MI_z_y_sorted[0, :]
        JEMMIG = H_diff + MI_z_y_sorted[1, :]
        JEMMIG_norm = JEMMIG / (np.log(num_bins) + H_y)

        JEMMIGs_norm_by_model.append(JEMMIG_norm)

        if i == 0:  # Sort for FactorVAE
            ids_sorted_by_factor = np.argsort(JEMMIG_norm)

    #"""
    attr_names = [attr_names[i][:10] for i in ids_sorted_by_factor]
    for i in range(len(JEMMIGs_norm_by_model)):
        JEMMIGs_norm_by_model[i] = JEMMIGs_norm_by_model[i][ids_sorted_by_factor]

    font = {'size': 12}
    matplotlib.rc('font', **font)

    width = 0.6
    plt.bar(2 * np.arange(len(attr_names)), JEMMIGs_norm_by_model[0], width=width,
            align='center', label='FactorVAE')
    plt.bar(2 * np.arange(len(attr_names)) + width, JEMMIGs_norm_by_model[1], width=width,
            align='center', label='BetaVAE')
    plt.bar(2 * np.arange(len(attr_names)) + 2 * width, JEMMIGs_norm_by_model[2], width=width,
            align='center', label='AAE')

    plt.xticks(2 * np.arange(len(attr_names)) + 1.5 * width, attr_names, rotation=45, ha='right')
    plt.ylabel("JEMMIG (normalized)")
    plt.ylim(bottom=0.4)
    plt.legend()

    subplot_adjust = {'left': 0.04, 'right': 0.996, 'bottom': 0.34, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(16, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_norm_models.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    #"""


def compare_JEMMIG_unnorm_models(save_dir, result_files, attr_names, num_bins):
    JEMMIGs_by_model = []
    ids_sorted_by_factor = []

    assert len(result_files) == 3, "These files must be the result files of FactorVAE, BetaVAE and AAEs!"

    for i in range(len(result_files)):
        results = np.load(result_files[i], "r")
        print("JEMMIG_results.keys: {}".format(list(results.keys())))

        MI_z_y = results['MI_z_y']
        H_z_y = results['H_z_y']
        H_y = results['H_y_4_diff_z'][:, 0]

        ids_sorted_by_MI = np.argsort(MI_z_y, axis=0)[::-1]
        MI_z_y_sorted = np.take_along_axis(MI_z_y, ids_sorted_by_MI, axis=0)
        H_z_y_sorted = np.take_along_axis(H_z_y, ids_sorted_by_MI, axis=0)

        H_diff = H_z_y_sorted[0, :] - MI_z_y_sorted[0, :]
        JEMMIG = H_diff + MI_z_y_sorted[1, :]

        JEMMIGs_by_model.append(JEMMIG)

        if i == 0:  # Sort for FactorVAE
            ids_sorted_by_factor = np.argsort(JEMMIG)

    #"""
    attr_names = [attr_names[i][:10] for i in ids_sorted_by_factor]
    for i in range(len(JEMMIGs_by_model)):
        JEMMIGs_by_model[i] = JEMMIGs_by_model[i][ids_sorted_by_factor]

    font = {'size': 12}
    matplotlib.rc('font', **font)

    width = 0.6
    plt.bar(2 * np.arange(len(attr_names)), JEMMIGs_by_model[0], width=width,
            align='center', label='FactorVAE')
    plt.bar(2 * np.arange(len(attr_names)) + width, JEMMIGs_by_model[1], width=width,
            align='center', label='BetaVAE')
    plt.bar(2 * np.arange(len(attr_names)) + 2 * width, JEMMIGs_by_model[2], width=width,
            align='center', label='AAE')

    plt.xticks(2 * np.arange(len(attr_names)) + 1.5 * width, attr_names, rotation=45, ha='right')
    plt.ylabel("JEMMIG (unnormalized)")
    plt.ylim(bottom=2)
    plt.legend()

    subplot_adjust = {'left': 0.03, 'right': 0.996, 'bottom': 0.34, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(16, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "JEMMIG_unorm_models.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    #"""


def compare_RMIG_norm_models(save_dir, result_files, attr_names, num_bins):
    RMIGs_norm_by_model = []
    ids_sorted_by_factor = []

    assert len(result_files) == 3, "These files must be the result files of FactorVAE, BetaVAE and AAEs!"

    for i in range(len(result_files)):
        results = np.load(result_files[i], "r")
        print("JEMMIG_results.keys: {}".format(list(results.keys())))

        RMIG_norm = results['MI_gap_y']
        RMIGs_norm_by_model.append(RMIG_norm)

        if i == 0:  # Sort for FactorVAE
            ids_sorted_by_factor = np.argsort(RMIG_norm)[::-1]

    #"""
    attr_names = [attr_names[i][:10] for i in ids_sorted_by_factor]
    for i in range(len(RMIGs_norm_by_model)):
        RMIGs_norm_by_model[i] = RMIGs_norm_by_model[i][ids_sorted_by_factor]

    font = {'size': 12}
    matplotlib.rc('font', **font)

    width = 0.6
    plt.bar(2 * np.arange(len(attr_names)), RMIGs_norm_by_model[0], width=width,
            align='center', label='FactorVAE')
    plt.bar(2 * np.arange(len(attr_names)) + width, RMIGs_norm_by_model[1], width=width,
            align='center', label='BetaVAE')
    plt.bar(2 * np.arange(len(attr_names)) + 2 * width, RMIGs_norm_by_model[2], width=width,
            align='center', label='AAE')

    plt.xticks(2 * np.arange(len(attr_names)) + 1.5 * width, attr_names, rotation=45, ha='right')
    plt.ylabel("RMIG (normalized)")
    # plt.ylim(bottom=0.4)
    plt.legend()

    subplot_adjust = {'left': 0.048, 'right': 0.996, 'bottom': 0.34, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(16, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_norm_models.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    #"""


def compare_RMIG_unnorm_models(save_dir, result_files, attr_names, num_bins):
    RMIGs_by_model = []
    ids_sorted_by_factor = []

    assert len(result_files) == 3, "These files must be the result files of FactorVAE, BetaVAE and AAEs!"

    for i in range(len(result_files)):
        results = np.load(result_files[i], "r")
        print("JEMMIG_results.keys: {}".format(list(results.keys())))

        RMIG_norm = results['MI_gap_y']
        H_y = results['H_y_4_diff_z'][:, 0]
        RMIG = RMIG_norm * H_y

        RMIGs_by_model.append(RMIG)

        if i == 0:  # Sort for FactorVAE
            ids_sorted_by_factor = np.argsort(RMIG)[::-1]

    #"""
    attr_names = [attr_names[i][:10] for i in ids_sorted_by_factor]
    for i in range(len(RMIGs_by_model)):
        RMIGs_by_model[i] = RMIGs_by_model[i][ids_sorted_by_factor]

    font = {'size': 12}
    matplotlib.rc('font', **font)

    width = 0.6
    plt.bar(2 * np.arange(len(attr_names)), RMIGs_by_model[0], width=width,
            align='center', label='FactorVAE')
    plt.bar(2 * np.arange(len(attr_names)) + width, RMIGs_by_model[1], width=width,
            align='center', label='BetaVAE')
    plt.bar(2 * np.arange(len(attr_names)) + 2 * width, RMIGs_by_model[2], width=width,
            align='center', label='AAE')

    plt.xticks(2 * np.arange(len(attr_names)) + 1.5 * width, attr_names, rotation=45, ha='right')
    plt.ylabel("RMIG (unnormalized)")
    # plt.ylim(bottom=0.4)
    plt.legend()

    subplot_adjust = {'left': 0.048, 'right': 0.996, 'bottom': 0.34, 'top': 0.98}
    plt.subplots_adjust(**subplot_adjust)
    plt.gcf().set_size_inches(16, 3)

    save_dir = make_dir_if_not_exist(save_dir)
    save_file = join(save_dir, "RMIG_unnorm_models.pdf")

    with PdfPages(save_file) as pdf_file:
        plt.savefig(pdf_file, format='pdf')

    plt.show()
    plt.close()
    #"""

def main():
    attr_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
                  'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                  'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                  'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                  'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                  'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                  'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                  'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

    enc_dec_model = "1Konny"

    model_and_run_ids = [
        ("FactorVAE", "1_tc50_multiSave"),
        ("FactorVAE", "6_VAE_beta50"),
        ("AAE", "0_Gz50"),
    ]

    labels = ["FactorVAE", "BetaVAE", "AAE"]

    save_dir = abspath(join(RESULTS_DIR, "celebA", "FactorVAE", "auxiliary", "JEMMIG_models_plot"))

    # interpretability result files
    # =========================================== #
    result_files = []

    num_bins = 100
    bin_limits = (-4.0, 4.0)
    data_proportion = 1.0

    bin_width = (bin_limits[1] - bin_limits[0]) / num_bins

    for model, run_id in model_and_run_ids:
        result_files.append(
            abspath(join(RESULTS_DIR, "celebA", model,
                         "auxiliary", "interpretability_metrics_v2",
                         "{}_{}".format(model, run_id),
                         "results[bins={},bin_limits={},data={}].npz".format(
                             num_bins, bin_limits, data_proportion))))
    # =========================================== #

    compare_RMIG_norm_models(save_dir, result_files, attr_names, num_bins)
    compare_RMIG_unnorm_models(save_dir, result_files, attr_names, num_bins)
    compare_JEMMIG_norm_models(save_dir, result_files, attr_names, num_bins)
    compare_JEMMIG_unnorm_models(save_dir, result_files, attr_names, num_bins)

if __name__ == "__main__":
    main()
