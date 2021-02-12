import argparse
from os.path import join, exists
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

from my_utils.tensorflow_utils.image import TFCelebAWithAttrLoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from utils.general import normal_density, at_bin


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--num_bins', type=int, default=100)
parser.add_argument('--bin_limits', type=str, default="-4;4")
parser.add_argument('--data_proportion', type=float, default=1.0)
parser.add_argument('--informativeness_metrics_dir', type=str, required=True)
parser.add_argument('--top_k', type=int, required=True)

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
    # save_dir = remove_dir_if_exist(join(args.save_dir, "FactorVAE_{}".format(args.run)), ask_4_permission=False)
    # save_dir = make_dir_if_not_exist(save_dir)

    save_dir = make_dir_if_not_exist(join(args.save_dir, "FactorVAE_{}".format(args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)

    num_bins = args.num_bins
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    data_proportion = args.data_proportion
    num_data = int(data_proportion * celebA_loader.num_train_data)
    top_k = args.top_k
    eps = 1e-8

    # file
    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')

    # print function
    print_ = functools.partial(print_both, file=f)

    print_("num_bins: {}".format(num_bins))
    print_("bin_limits: {}".format(bin_limits))
    print_("data_proportion: {}".format(data_proportion))
    print_("top_k: {}".format(top_k))

    # Compute bins
    # ================================= #
    print_("")
    print_("bin_limits: {}".format(bin_limits))
    assert len(bin_limits) == 2 and bin_limits[0] < bin_limits[1], "bin_limits={}".format(bin_limits)

    bins = np.linspace(bin_limits[0], bin_limits[1], num_bins + 1, endpoint=True)
    print_("bins: {}".format(bins))
    assert len(bins) == num_bins + 1

    bin_widths = [bins[b] - bins[b - 1] for b in range(1, len(bins))]
    print_("bin_widths: {}".format(bin_widths))
    assert len(bin_widths) == num_bins, "len(bin_widths)={} while num_bins={}!".format(len(bin_widths), num_bins)
    assert np.all(np.greater(bin_widths, 0)), "bin_widths: {}".format(bin_widths)

    bin_centers = [(bins[b] + bins[b - 1]) * 0.5 for b in range(1, len(bins))]
    print_("bin_centers: {}".format(bin_centers))
    assert len(bin_centers) == num_bins, "len(bin_centers)={} while num_bins={}!".format(len(bin_centers), num_bins)
    # ================================= #

    # Compute representations
    # ================================= #
    z_data_file = join(args.informativeness_metrics_dir, "FactorVAE_{}".format(args.run),
                       "z_data[data={}].npz".format(data_proportion))

    with np.load(z_data_file, "r") as f:
        all_z_mean = f['all_z_mean']
        all_z_stddev = f['all_z_stddev']

    print_("")
    print_("all_z_mean.shape: {}".format(all_z_mean.shape))
    print_("all_z_stddev.shape: {}".format(all_z_stddev.shape))
    # ================================= #

    # Compute the mutual information
    # ================================= #
    mi_file = join(args.informativeness_metrics_dir, "FactorVAE_{}".format(args.run),
                   'results[bins={},bin_limits={},data={}].npz'.
                   format(num_bins, bin_limits, data_proportion))
    with np.load(mi_file, "r") as f:
        sorted_MI_z_x = f['sorted_MI_z_x']
        sorted_z_ids = f['sorted_z_comps']
        H_z = f['H_z']

    if top_k > 0:
        top_MI = sorted_MI_z_x[:top_k]
        top_z_ids = sorted_z_ids[:top_k]

        bot_MI = sorted_MI_z_x[-top_k:]
        bot_z_ids = sorted_z_ids[-top_k:]

        top_bot_MI = np.concatenate([top_MI, bot_MI], axis=0)
        top_bot_z_ids = np.concatenate([top_z_ids, bot_z_ids], axis=0)

        print_("top MI: {}".format(top_MI))
        print_("top_z_ids: {}".format(top_z_ids))
        print_("bot MI: {}".format(bot_MI))
        print_("bot_z_ids: {}".format(bot_z_ids))

    else:
        top_bot_MI = sorted_MI_z_x
        top_bot_z_ids = sorted_z_ids
    # ================================= #

    H_z1z2_mean_mat = np.full([len(top_bot_z_ids), len(top_bot_z_ids)], -1, dtype=np.float32)
    MI_z1z2_mean_mat = np.full([len(top_bot_z_ids), len(top_bot_z_ids)], -1, dtype=np.float32)
    H_z1z2_mean = []
    MI_z1z2_mean = []
    z1z2_ids = []

    # Compute the mutual information
    # ================================= #
    for i in range(len(top_bot_z_ids)):
        z_idx1 = top_bot_z_ids[i]
        H_s1 = H_z[z_idx1]

        for j in range(i+1, len(top_bot_z_ids)):
            z_idx2 = top_bot_z_ids[j]
            H_s2 = H_z[z_idx2]

            print_("")
            print_("Compute MI(z{}_mean, z{}_mean)!".format(z_idx1, z_idx2))

            s1s2_mean_counter = np.zeros([num_bins, num_bins], dtype=np.int32)

            for batch_ids in iterate_data(len(all_z_mean), 100, shuffle=False, include_remaining=True):
                s1 = at_bin(all_z_mean[batch_ids, z_idx1], bins, one_hot=False)
                s2 = at_bin(all_z_mean[batch_ids, z_idx2], bins, one_hot=False)

                for s1_, s2_ in zip(s1, s2):
                    s1s2_mean_counter[s1_, s2_] += 1

            # I(s1, s2) = Q(s1, s2) * (log Q(s1, s2) - log Q(s1) log Q(s2))
            # ---------------------------------- #
            Q_s1s2_mean = (s1s2_mean_counter * 1.0) / np.sum(s1s2_mean_counter).astype(np.float32)
            log_Q_s1s2_mean = np.log(np.maximum(Q_s1s2_mean, eps))

            H_s1s2_mean = -np.sum(Q_s1s2_mean * log_Q_s1s2_mean)
            MI_s1s2_mean = H_s1 + H_s2 - H_s1s2_mean

            print_("H_s1: {}".format(H_s1))
            print_("H_s2: {}".format(H_s2))
            print_("H_s1s2_mean: {}".format(H_s1s2_mean))
            print_("MI_s1s2_mean: {}".format(MI_s1s2_mean))

            H_z1z2_mean.append(H_s1s2_mean)
            MI_z1z2_mean.append(MI_s1s2_mean)
            z1z2_ids.append((z_idx1, z_idx2))

            H_z1z2_mean_mat[i, j] = H_s1s2_mean
            H_z1z2_mean_mat[j, i] = H_s1s2_mean
            MI_z1z2_mean_mat[i, j] = MI_s1s2_mean
            MI_z1z2_mean_mat[j, i] = MI_s1s2_mean

    H_z1z2_mean = np.asarray(H_z1z2_mean, dtype=np.float32)
    MI_z1z2_mean = np.asarray(MI_z1z2_mean, dtype=np.float32)
    z1z2_ids = np.asarray(z1z2_ids, dtype=np.int32)

    result_file = join(save_dir, "results[bins={},bin_limits={},data={},k={}].npz".
                       format(num_bins, bin_limits, data_proportion, top_k))
    results = {
        'H_z1z2_mean': H_z1z2_mean,
        'MI_z1z2_mean': MI_z1z2_mean,
        'H_z1z2_mean_mat': H_z1z2_mean_mat,
        'MI_z1z2_mean_mat': MI_z1z2_mean_mat,
        'z1z2_ids': z1z2_ids,
    }

    np.savez_compressed(result_file, **results)
    # ================================= #
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
