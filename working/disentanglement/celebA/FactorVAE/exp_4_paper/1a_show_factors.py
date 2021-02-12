import argparse
from os.path import join
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist, print_both
from my_utils.python_utils.training import iterate_data
from my_utils.python_utils.image import binary_float_to_uint8

from my_utils.tensorflow_utils.image import TFCelebALoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)


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
        disc_z = DiscriminatorZ_1Konny(num_outputs=2)
    else:
        raise ValueError("Do not support encoder/decoder model '{}'!".format(args.enc_dec_model))

    model = FactorVAE([img_height, img_width, 3], args.z_dim,
                      encoder=encoder, decoder=decoder,
                      discriminator_z=disc_z,
                      rec_x_mode=args.rec_x_mode,
                      use_gp0_z_tc=True, gp0_z_tc_mode=args.gp0_z_tc_mode)

    loss_coeff_dict = {
        'rec_x': args.rec_x_coeff,
        'kld_loss': args.kld_loss_coeff,
        'tc_loss': args.tc_loss_coeff,
        'gp0_z_tc': args.gp0_z_tc_coeff,
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
    save_dir = remove_dir_if_exist(join(args.save_dir, "FactorVAE_{}".format(args.run)), ask_4_permission=False)
    save_dir = make_dir_if_not_exist(save_dir)

    # save_dir = make_dir_if_not_exist(join(args.save_dir, "FactorVAE_{}".format(args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)
    f = open(join(save_dir, 'log.txt'), mode='w')
    print_ = functools.partial(print_both, file=f)

    # z gaussian stddev
    # ======================================= #
    all_z_mean = []
    all_z_stddev = []

    count = 0
    for batch_ids in iterate_data(int(0.05 * num_train), 10 * args.batch_size, shuffle=False):
        x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)

        z_mean, z_stddev = sess.run(model.get_output(['z_mean', 'z_stddev']),
                                    feed_dict={model.is_train: False, model.x_ph: x})

        all_z_mean.append(z_mean)
        all_z_stddev.append(z_stddev)

        count += len(batch_ids)
        print("\rProcessed {} samples!".format(count), end="")
    print()

    all_z_mean = np.concatenate(all_z_mean, axis=0)
    all_z_stddev = np.concatenate(all_z_stddev, axis=0)
    # ======================================= #

    z_std_error = np.std(all_z_mean, axis=0, ddof=0)
    z_sorted_comps = np.argsort(z_std_error)[::-1]
    top10_z_comps = z_sorted_comps[:10]

    print_("")
    print_("z_std_error: {}".format(z_std_error))
    print_("z_sorted_std_error: {}".format(z_std_error[z_sorted_comps]))
    print_("z_sorted_comps: {}".format(z_sorted_comps))
    print_("top10_z_comps: {}".format(top10_z_comps))

    z_stddev_mean = np.mean(all_z_stddev, axis=0)
    info_z_comps = [idx for idx in range(len(z_stddev_mean)) if z_stddev_mean[idx] < 0.4]
    print_("info_z_comps: {}".format(info_z_comps))
    print_("len(info_z_comps): {}".format(len(info_z_comps)))

    # Plotting
    # =========================================== #
    seed = 389
    num_samples = 30
    ids = list(range(seed, seed + num_samples))
    print("\nids: {}".format(ids))

    data = celebA_loader.sample_images_from_dataset(sess, 'train', ids)

    span = 3
    points_one_side = 5

    # span = 8
    # points_one_side = 12

    for n in range(len(ids)):
        print("Plot conditional all comps z traverse with train sample {}!".format(ids[n]))
        img_file = join(save_dir, "x_train[{}]_[span={}]_hl.png".format(ids[n], span))
        # model.cond_all_latents_traverse_v2(img_file, sess, data[n],
        #                                 z_comps=top10_z_comps,
        #                                 z_comp_labels=None,
        #                                 span=span, points_1_side=points_one_side,
        #                                 hl_x=True,
        #                                 batch_size=args.batch_size,
        #                                 dec_output_2_img_func=binary_float_to_uint8)

        img_file = join(save_dir, "x_train[{}]_[span={}]_hl_labeled.png".format(ids[n], span))
        model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                           z_comps=top10_z_comps,
                                           z_comp_labels=["z[{}]".format(comp) for comp in top10_z_comps],
                                           span=span, points_1_side=points_one_side,
                                           hl_x=True,
                                           subplot_adjust={'left': 0.09, 'right': 0.98, 'bottom': 0.02, 'top': 0.98},
                                           size_inches=(6, 5),
                                           batch_size=args.batch_size,
                                           dec_output_2_img_func=binary_float_to_uint8)

        # img_file = join(save_dir, "x_train[{}]_[span={}].png".format(ids[n], span))
        # model.cond_all_latents_traverse_v2(img_file, sess, data[n],
        #                                    z_comps=top10_z_comps,
        #                                    z_comp_labels=None,
        #                                    span=span, points_1_side=points_one_side,
        #                                    hl_x=False,
        #                                    batch_size=args.batch_size,
        #                                    dec_output_2_img_func=binary_float_to_uint8)
        #
        # img_file = join(save_dir, "x_train[{}]_[span={}]_labeled.png".format(ids[n], span))
        # model.cond_all_latents_traverse_v2(img_file, sess, data[n],
        #                                    z_comps=top10_z_comps,
        #                                    z_comp_labels=["z[{}]".format(comp) for comp in top10_z_comps],
        #                                    span=span, points_1_side=points_one_side,
        #                                    hl_x=False,
        #                                    subplot_adjust={'left': 0.09, 'right': 0.98, 'bottom': 0.02, 'top': 0.98},
        #                                    size_inches=(6, 5),
        #                                    batch_size=args.batch_size,
        #                                    dec_output_2_img_func=binary_float_to_uint8)

        img_file = join(save_dir, "x_train[{}]_[span={}]_info_hl.png".format(ids[n], span))
        model.cond_all_latents_traverse_v2(img_file, sess, data[n],
                                           z_comps=info_z_comps,
                                           z_comp_labels=None,
                                           span=span, points_1_side=points_one_side,
                                           hl_x=True,
                                           batch_size=args.batch_size,
                                           dec_output_2_img_func=binary_float_to_uint8)
    # =========================================== #

    f.close()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
