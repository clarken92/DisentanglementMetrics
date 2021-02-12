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

from my_utils.tensorflow_utils.image import TFCelebAWithAttrLoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter


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
    celebA_loader = TFCelebAWithAttrLoader(root_dir=args.celebA_root_dir)

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
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    data_proportion = args.data_proportion

    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')
    print_ = functools.partial(print_both, file=f)

    result_file = join(args.interpretability_metrics_dir, "AAE_{}".format(args.run),
                       "results[bins={},bin_limits={},data={}].npz".
                       format(num_bins, bin_limits, data_proportion))

    results = np.load(result_file, "r")

    print_("")
    print_("num_bins: {}".format(num_bins))
    print_("bin_limits: {}".format(bin_limits))
    print_("data_proportion: {}".format(data_proportion))

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

    attr_names = celebA_loader.attributes
    print_("attr_names: {}".format(attr_names))
    print_("results.keys: {}".format(list(results.keys())))

    # (z_dim, num_attrs)
    MI_ids_sorted = results['MI_ids_sorted']
    MI_sorted = results['MI_sorted']

    MI_gap_y = results['MI_gap_y']
    H_y = results['H_y_4_diff_z'][:, 0]
    assert MI_ids_sorted.shape[1] == len(attr_names) == len(MI_gap_y) == len(H_y), \
        "MI_ids_sorted.shape: {}, len(attr_names): {}, len(MI_gap_y): {}, len(H_y): {}".format(
            MI_ids_sorted.shape, len(attr_names), len(MI_gap_y), len(H_y))

    print_("\nShow RMIG!")
    for i in range(len(attr_names)):
        print("{}: RMIG: {:.4f}, RMIG (unnorm): {:.4f}, H: {:.4f}".format(
            attr_names[i], MI_gap_y[i], MI_gap_y[i] * H_y[i], H_y[i]))

    print_("\nShow JEMMI!")
    H_z_y = results['H_z_y']
    MI_z_y = results['MI_z_y']

    ids_sorted_by_MI = np.argsort(MI_z_y, axis=0)[::-1]
    MI_z_y_sorted = np.take_along_axis(MI_z_y, ids_sorted_by_MI, axis=0)
    H_z_y_sorted = np.take_along_axis(H_z_y, ids_sorted_by_MI, axis=0)

    H_diff = H_z_y_sorted[0, :] - MI_z_y_sorted[0, :]
    JEMMI_unnorm = H_diff + MI_z_y_sorted[1, :]
    JEMMI_norm = JEMMI_unnorm / (np.log(num_bins) + H_y)

    for i in range(len(attr_names)):
        print("{}: JEMMI: {:.4f}, JEMMI (unnorm): {:.4f}, H_diff: {:.4f}, I2: {:.4f}, top 2 latents: z{}, z{}".format(
            attr_names[i], JEMMI_norm[i], JEMMI_unnorm[i], H_diff[i], MI_z_y_sorted[1, i],
            ids_sorted_by_MI[0, i], ids_sorted_by_MI[1, i]))

    # Uncomment if you want
    '''
    for n in range(len(ids)):
        for k in range(len(attr_names)):
            MI_ids_top10 = MI_ids_sorted[:10, k]
            MI_top10 = MI_sorted[:10, k]
            print("Plot top 10 latents for factor '{}'!".format(attr_names[k]))

            img_file = join(save_dir, "x_train[{}][attr={}][bins={},bin_limits={},data={}].png".
                            format(ids[n], attr_names[k], num_bins, bin_limits, data_proportion))

            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=MI_ids_top10,
                                               z_comp_labels=["z[{}] ({:.4f})".format(comp, mi)
                                                              for comp, mi in zip(MI_ids_top10, MI_top10)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               title="{} (MI gap={:.4f}, H={:.4f})".format(
                                                   attr_names[k], MI_gap_y[k], H_y[k]),
                                               title_font_scale=1.5,
                                               subplot_adjust={'left': 0.16, 'right': 0.99,
                                                               'bottom': 0.01, 'top': 0.95},
                                               size_inches=(6.5, 5.2),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)
    '''

    # Top 5 only
    for n in range(len(ids)):
        for k in range(len(attr_names)):
            MI_ids_top10 = MI_ids_sorted[:5, k]
            MI_top10 = MI_sorted[:5, k]
            print("Plot top 5 latents for factor '{}'!".format(attr_names[k]))

            img_file = join(save_dir, "train{}_attr={}_bins={}_data={}.png".
                            format(ids[n], attr_names[k], num_bins, data_proportion))

            model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                               z_comps=MI_ids_top10,
                                               z_comp_labels=["z[{}] ({:.4f})".format(comp, mi)
                                                              for comp, mi in zip(MI_ids_top10, MI_top10)],
                                               span=span, points_1_side=points_one_side,
                                               hl_x=True,
                                               font_size=9,
                                               title="{} (MI gap={:.4f}, H={:.4f})".format(
                                                   attr_names[k], MI_gap_y[k], H_y[k]),
                                               title_font_scale=1.5,
                                               subplot_adjust={'left': 0.16, 'right': 0.99,
                                                               'bottom': 0.005, 'top': 0.93},
                                               size_inches=(6.5, 2.8),
                                               batch_size=args.batch_size,
                                               dec_output_2_img_func=binary_float_to_uint8)

    '''
    # Sorted by y
    y_ids_sorted = np.argsort(MI_gap_y / H_y, axis=0)[::-1]
    sorted_attrs_names = [attr_names[idx] for idx in y_ids_sorted]

    MIG_sorted_by_y = np.take_along_axis(MI_gap_y, y_ids_sorted, axis=0) / H_y
    print("MIG_sorted_by_y: {}".format(MIG_sorted_by_y))

    # (z_dim, num_attrs)
    MI_ids_sorted_by_y = MI_ids_sorted[:, y_ids_sorted]
    MI_sorted_by_y = MI_sorted[:, y_ids_sorted] / H_y

    z_ids = [MI_ids_sorted_by_y[:2, i] for i in range(5)]
    z_ids = np.concatenate(z_ids, axis=0)

    z_comp_labels = []
    for i in range(5):
        z_comp_labels.append("{} z[{}] ({:.3f})".format(
            sorted_attrs_names[i], MI_ids_sorted_by_y[0, i], MI_sorted_by_y[0, i]))
        z_comp_labels.append("{} z[{}] ({:.3f})".format(
            sorted_attrs_names[i], MI_ids_sorted_by_y[1, i], MI_sorted_by_y[1, i]))

    for n in range(len(ids)):
        img_file = join(save_dir, "x_train[{}][bins={},bin_limits={},data={}]_top5attr.png".
                        format(ids[n], num_bins, bin_limits, data_proportion))

        model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                           z_comps=z_ids,
                                           z_comp_labels=z_comp_labels,
                                           span=span, points_1_side=points_one_side,
                                           hl_x=True,
                                           font_size=5,
                                           title=None,
                                           title_font_scale=1.5,
                                           subplot_adjust={'left': 0.23, 'right': 0.99,
                                                           'bottom': 0.01, 'top': 0.99},
                                           size_inches=(5.3, 3.8),
                                           batch_size=args.batch_size,
                                           dec_output_2_img_func=binary_float_to_uint8)

    f.close()
    '''

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
