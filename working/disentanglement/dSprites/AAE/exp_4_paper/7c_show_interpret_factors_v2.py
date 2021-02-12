import argparse
from os.path import join
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist, print_both
from my_utils.python_utils.image import binary_float_to_uint8, uint8_to_binary_float

from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from global_settings import RAW_DATA_DIR

parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--interpretability_metrics_dir', type=str, required=True)
parser.add_argument('--num_bins', type=int, default=100)
parser.add_argument('--bin_limits', type=str, default="-4;4")
parser.add_argument('--data_proportion', type=float, default=0.1)


def main(args):
    # =====================================
    # Load config
    # =====================================
    with open(join(args.output_dir, 'config.json')) as f:
        config = json.load(f)
    args.__dict__.update(config)

    # =====================================
    # Dataset
    # =====================================
    data_file = join(RAW_DATA_DIR, "ComputerVision", "dSprites",
                     "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    # It is already in the range [0, 1]
    with np.load(data_file, encoding="latin1") as f:
        x_train = f['imgs']

    x_train = np.reshape(x_train, [3, 6, 40, 32, 32, 64, 64, 1])

    # =====================================
    # Instantiate model
    # =====================================
    if args.enc_dec_model == "1Konny":
        encoder = Encoder_1Konny(args.z_dim, stochastic=True)
        decoder = Decoder_1Konny()
        disc_z = DiscriminatorZ_1Konny()
    else:
        raise ValueError("Do not support enc_dec_model='{}'!".format(args.enc_dec_model))

    model = AAE([64, 64, 1], args.z_dim,
                encoder=encoder, decoder=decoder,
                discriminator_z=disc_z,
                rec_x_mode=args.rec_x_mode,
                stochastic_z=args.stochastic_z,
                use_gp0_z=True, gp0_z_mode=args.gp0_z_mode)

    loss_coeff_dict = {
        'rec_x': args.rec_x_coeff,
        'G_loss_z1_gen': args.G_loss_z1_gen_coeff,
        'D_loss_z1_gen': args.D_loss_z1_gen_coeff,
        'gp0_z': args.gp0_z_coeff,
    }

    model.build(loss_coeff_dict)
    SimpleParamPrinter.print_all_params_tf_slim()

    # =====================================
    # Load model
    # =====================================
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config_proto)

    model_dir = make_dir_if_not_exist(join(args.output_dir, "model_tf"))
    train_helper = SimpleTrainHelper(log_dir=None, save_dir=model_dir)

    # Load model
    train_helper.load(sess, load_step=args.load_step)

    # =====================================
    # Experiments
    # save_dir = remove_dir_if_exist(join(args.save_dir, "{}_{}".format(args.enc_dec_model, args.run)), ask_4_permission=True)
    # save_dir = make_dir_if_not_exist(save_dir)

    save_dir = make_dir_if_not_exist(join(args.save_dir, "{}_{}".format(args.enc_dec_model, args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)

    num_bins = args.num_bins
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    data_proportion = args.data_proportion

    # Logs
    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')
    print_ = functools.partial(print_both, file=f)

    print_("")
    print_("num_bins: {}".format(num_bins))
    print_("bin_limits: {}".format(bin_limits))
    print_("data_proportion: {}".format(data_proportion))

    # Results
    result_file = join(args.interpretability_metrics_dir,
                       "{}_{}".format(args.enc_dec_model, args.run),
                       "results[bins={},bin_limits={},data={}].npz".
                       format(num_bins, bin_limits, data_proportion))

    results = np.load(result_file, "r")
    print_("results.keys: {}".format(list(results.keys())))

    # Plotting
    # =========================================== #
    data = [x_train[0, 3, 20, 16, 16],
            x_train[1, 3, 20, 16, 16],
            x_train[2, 3, 20, 16, 16]]

    gt_factors = ['Shape', 'Scale', 'Rotation', 'Pos_x', 'Pos_y']
    ids_sorted = results['ids_sorted']
    MI_z_y_sorted = results['MI_z_y_sorted']
    H_z_y_sorted = results['H_z_y_sorted']
    H_y = results['H_y']
    RMIG = results['RMIG']
    JEMMI = results['JEMMI']

    print_("MI_z_y_sorted:\n{}".format(MI_z_y_sorted))

    print_("\nShow RMIG!")
    for k in range(len(gt_factors)):
        print_("{}, RMIG: {:.4f}, RMIG (unnorm): {:.4f}, H: {:.4f}, I1: {:.4f}, I2: {:.4f}".format(
            gt_factors[k], RMIG[k], RMIG[k] * H_y[k], H_y[k], MI_z_y_sorted[0, k], MI_z_y_sorted[1, k]))

    print_("\nShow JEMMI!")
    for k in range(len(gt_factors)):
        print_("{}, JEMMI: {:.4f}, JEMMI (unnorm): {:.4f}, H1: {:.4f}, H1-I1: {:.4f}, I2: {:.4f}, "
               "top2 ids: z{}, z{}".format(gt_factors[k], JEMMI[k], JEMMI[k] * (H_y[k] + np.log(num_bins)),
                                           H_z_y_sorted[0, k], H_z_y_sorted[0, k] - MI_z_y_sorted[0, k],
                                           MI_z_y_sorted[1, k], ids_sorted[0, k], ids_sorted[1, k]))

    span = 3
    points_one_side = 5

    for n in range(len(data)):
        for k in range(len(gt_factors)):
            print("x={}, y={}!".format(n, gt_factors[k]))

            img_file = join(save_dir, "{}[x={},bins={},bin_limits={},data={}].png".
                            format(gt_factors[k], n, num_bins, bin_limits, data_proportion))

            '''
            ids_top10 = ids_sorted[:10, k]
            MI_top10 = MI_z_y_sorted[:10, k]
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=ids_top10,
                                               z_comp_labels=["z[{}] ({:.4f})".format(comp, mi)
                                                              for comp, mi in zip(ids_top10, MI_top10)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               title="{} (RMIG={:.4f}, JEMMI={:.4f}, H={:.4f})".format(
                                                   gt_factors[k], RMIG[k], JEMMI[k], H_y[k]),
                                               title_font_scale=1.5,
                                               subplot_adjust={'left': 0.16, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.95},
                                               size_inches=(6.5, 5.2),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)
            '''

            ids_top3 = ids_sorted[:3, k]
            MI_top3 = MI_z_y_sorted[:3, k]
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=ids_top3,
                                               z_comp_labels=["z[{}] ({:.4f})".format(comp, mi)
                                                              for comp, mi in zip(ids_top3, MI_top3)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               title="{} (RMIG={:.4f}, JEMMI={:.4f}, H={:.4f})".format(
                                                   gt_factors[k], RMIG[k], JEMMI[k], H_y[k]),
                                               title_font_scale=1.5,
                                               subplot_adjust={'left': 0.16, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.88},
                                               size_inches=(6.2, 1.7),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
