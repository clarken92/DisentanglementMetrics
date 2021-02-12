import argparse
from os.path import join
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist, print_both
from my_utils.python_utils.image import binary_float_to_uint8

from my_utils.tensorflow_utils.image import TFCelebALoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--informativeness_metrics_dir', type=str, required=True)
parser.add_argument('--num_bins', type=int, default=100)
parser.add_argument('--bin_limits', type=str, default="-4;4")
parser.add_argument('--data_proportion', type=float, default=0.1)
parser.add_argument('--top_k', type=int, default=10)


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
    celebA_loader = TFCelebALoader(root_dir=args.celebA_root_dir)

    img_height, img_width = args.celebA_resize_size, args.celebA_resize_size
    celebA_loader.build_transformation_flow_tf(
        *celebA_loader.get_transform_fns("1Konny", resize_size=args.celebA_resize_size))
    num_train = celebA_loader.num_train_data

    # =====================================
    # Instantiate model
    # =====================================
    if args.activation == "relu":
        activation = tf.nn.relu
    elif args.activation == "leaky_relu":
        activation = tf.nn.leaky_relu
    else:
        raise ValueError("Do not support '{}' activation!".format(args.activation))

    if args.enc_dec_model == "1Konny":
        # assert args.z_dim == 65, "For 1Konny, z_dim must be 65. Found {}!".format(args.z_dim)

        encoder = Encoder_1Konny(args.z_dim, stochastic=True, activation=activation)
        decoder = Decoder_1Konny([img_height, img_width, 3], activation=activation,
                                 output_activation=tf.nn.sigmoid)
        disc_z = DiscriminatorZ_1Konny()
    else:
        raise ValueError("Do not support encoder/decoder model '{}'!".format(args.enc_dec_model))

    model = AAE([img_height, img_width, 3], args.z_dim,
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
    # save_dir = remove_dir_if_exist(join(args.save_dir, "AAE_{}".format(args.run)), ask_4_permission=True)
    # save_dir = make_dir_if_not_exist(save_dir)

    save_dir = make_dir_if_not_exist(join(args.save_dir, "AAE_{}".format(args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)

    num_bins = args.num_bins
    data_proportion = args.data_proportion
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    top_k = args.top_k

    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')
    print_ = functools.partial(print_both, file=f)

    result_file = join(args.informativeness_metrics_dir, "AAE_{}".format(args.run),
                       'results[bins={},bin_limits={},data={}].npz'.
                       format(num_bins, bin_limits, data_proportion))

    results = np.load(result_file, "r")

    print_("")
    print_("num_bins: {}".format(num_bins))
    print_("bin_limits: {}".format(bin_limits))
    print_("data_proportion: {}".format(data_proportion))
    print_("top_k: {}".format(top_k))

    # Plotting
    # =========================================== #
    # seed = 389
    # num_samples = 30
    seed = 398
    num_samples = 1

    ids = list(range(seed, seed + num_samples))
    print_("\nids: {}".format(ids))

    data = celebA_loader.sample_images_from_dataset(sess, 'train', ids)

    span = 3
    points_one_side = 5

    print_("sorted_MI: {}".format(results["sorted_MI_z_x"]))
    print_("sorted_z_ids: {}".format(results["sorted_z_comps"]))
    print_("sorted_norm_MI: {}".format(results["sorted_norm_MI_z_x"]))
    print_("sorted_norm_z_ids: {}".format(results["sorted_norm_z_comps"]))

    top_MI = results["sorted_MI_z_x"][:top_k]
    top_z_ids = results["sorted_z_comps"][:top_k]
    top_norm_MI = results["sorted_norm_MI_z_x"][:top_k]
    top_norm_z_ids = results["sorted_norm_z_comps"][:top_k]

    for n in range(len(ids)):
        if top_k == 10:
            print("Plot conditional all comps z traverse with train sample {}!".format(ids[n]))

            img_file = join(save_dir, "x_train[{}][bins={},bin_limits={},data={}].png".
                            format(ids[n], num_bins, bin_limits, data_proportion))
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=top_z_ids,
                                               z_comp_labels=["z[{}] ({:.2f})".format(comp, mi)
                                                              for comp, mi in zip(top_z_ids, top_MI)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               subplot_adjust={'left': 0.15, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.99},
                                               size_inches=(6.3, 4.9),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)

            img_file = join(save_dir, "x_train[{}][bins={},bin_limits={},data={},norm].png".
                            format(ids[n], num_bins, bin_limits, data_proportion))
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=top_norm_z_ids,
                                               z_comp_labels=["z[{}] ({:.2f})".format(comp, mi)
                                                              for comp, mi in zip(top_norm_z_ids, top_norm_MI)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               subplot_adjust={'left': 0.15, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.99},
                                               size_inches=(6.3, 4.9),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)
        elif top_k == 45:
            print("Plot conditional all comps z traverse with train sample {}!".format(ids[n]))

            img_file = join(save_dir, "x_train[{}][bins={},bin_limits={},data={}].png".
                            format(ids[n], num_bins, bin_limits, data_proportion))
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=top_z_ids,
                                               z_comp_labels=["z[{}] ({:.2f})".format(comp, mi)
                                                              for comp, mi in zip(top_z_ids, top_MI)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=5,
                                               subplot_adjust={'left': 0.19, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.99},
                                               size_inches=(2.98, 9.85),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)

            img_file = join(save_dir, "x_train[{}][bins={},bin_limits={},data={},norm].png".
                            format(ids[n], num_bins, bin_limits, data_proportion))
            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=top_norm_z_ids,
                                               z_comp_labels=["z[{}] ({:.2f})".format(comp, mi)
                                                              for comp, mi in zip(top_norm_z_ids, top_norm_MI)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=5,
                                               subplot_adjust={'left': 0.19, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.99},
                                               size_inches=(2.98, 9.85),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)
    # =========================================== #

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
