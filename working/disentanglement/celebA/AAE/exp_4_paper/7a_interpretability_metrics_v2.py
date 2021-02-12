import argparse
from os.path import join, exists
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, print_both
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
    # save_dir = remove_dir_if_exist(join(args.save_dir, "AAE_{}".format(args.run)), ask_4_permission=False)
    # save_dir = make_dir_if_not_exist(save_dir)

    save_dir = make_dir_if_not_exist(join(args.save_dir, "AAE_{}".format(args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3, suppress=True)

    num_bins = args.num_bins
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    data_proportion = args.data_proportion
    num_data = int(data_proportion * celebA_loader.num_train_data)
    eps = 1e-8

    # file
    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')

    # print function
    print_ = functools.partial(print_both, file=f)

    '''
    if attr_type == 0:
        attr_names = celebA_loader.attributes
    elif attr_type == 1:
        attr_names = ['Male', 'Black_Hair', 'Blond_Hair', 'Straight_Hair', 'Wavy_Hair', 'Bald',
                      'Oval_Face', 'Big_Nose', 'Chubby', 'Double_Chin', 'Goatee', 'No_Beard',
                      'Mouth_Slightly_Open', 'Smiling',
                      'Eyeglasses', 'Pale_Skin']
    else:
        raise ValueError("Only support factor_type=0 or 1!")
    '''

    print_("num_bins: {}".format(num_bins))
    print_("bin_limits: {}".format(bin_limits))
    print_("data_proportion: {}".format(data_proportion))

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
    z_data_attr_file = join(save_dir, "z_data[data={}].npz".format(data_proportion))

    if not exists(z_data_attr_file):
        all_z_mean = []
        all_z_stddev = []
        all_attrs = []

        print("")
        print("Compute all_z_mean, all_z_stddev and all_attrs!")

        count = 0
        for batch_ids in iterate_data(num_data, 10 * args.batch_size, shuffle=False):
            x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)
            attrs = celebA_loader.sample_attrs_from_dataset('train', batch_ids)
            assert attrs.shape[1] == celebA_loader.num_attributes

            z_mean, z_stddev = sess.run(
                model.get_output(['z_mean', 'z_stddev']),
                feed_dict={model.is_train: False, model.x_ph: x})

            all_z_mean.append(z_mean)
            all_z_stddev.append(z_stddev)
            all_attrs.append(attrs)

            count += len(batch_ids)
            print("\rProcessed {} samples!".format(count), end="")
        print()

        all_z_mean = np.concatenate(all_z_mean, axis=0)
        all_z_stddev = np.concatenate(all_z_stddev, axis=0)
        all_attrs = np.concatenate(all_attrs, axis=0)

        np.savez_compressed(z_data_attr_file, all_z_mean=all_z_mean,
                            all_z_stddev=all_z_stddev, all_attrs=all_attrs)
    else:
        print("{} exists. Load data from file!".format(z_data_attr_file))
        with np.load(z_data_attr_file, "r") as f:
            all_z_mean = f['all_z_mean']
            all_z_stddev = f['all_z_stddev']
            all_attrs = f['all_attrs']

    print_("")
    print_("all_z_mean.shape: {}".format(all_z_mean.shape))
    print_("all_z_stddev.shape: {}".format(all_z_stddev.shape))
    print_("all_attrs.shape: {}".format(all_attrs.shape))
    # ================================= #

    # Compute the probability mass function for ground truth factors
    # ================================= #
    num_attrs = all_attrs.shape[1]

    assert all_attrs.dtype == np.bool
    all_attrs = all_attrs.astype(np.int32)

    # (num_samples, num_attrs, 2)    # The first component is 1 and the last component is 0
    all_Q_y_cond_x = np.stack([all_attrs, 1 - all_attrs], axis=-1)
    # ================================= #

    # Compute Q(zi|x)
    # Compute I(zi, yk)
    # ================================= #
    Q_z_y = np.zeros([args.z_dim, num_attrs, num_bins, 2], dtype=np.float32)
    MI_z_y = np.zeros([args.z_dim, num_attrs], dtype=np.float32)
    H_z_y = np.zeros([args.z_dim, num_attrs], dtype=np.float32)
    H_z_4_diff_y = np.zeros([args.z_dim, num_attrs], dtype=np.float32)
    H_y_4_diff_z = np.zeros([num_attrs, args.z_dim], dtype=np.float32)

    for i in range(args.z_dim):
        print_("")
        print_("Compute all_Q_z{}_cond_x!".format(i))

        # Q_s_cond_x
        all_Q_s_cond_x = []
        for batch_ids in iterate_data(len(all_z_mean), 500, shuffle=False, include_remaining=True):
            # (batch_size, num_bins)
            q_s_cond_x = normal_density(np.expand_dims(bin_centers, axis=0),
                                        mean=np.expand_dims(all_z_mean[batch_ids, i], axis=-1),
                                        stddev=np.expand_dims(all_z_stddev[batch_ids, i], axis=-1))

            # (batch_size, num_bins)
            max_q_s_cond_x = np.max(q_s_cond_x, axis=-1)
            # print("\nmax_q_s_cond_x: {}".format(np.sort(max_q_s_cond_x)))

            # (batch_size, num_bins)
            deter_s_cond_x = at_bin(all_z_mean[batch_ids, i], bins).astype(np.float32)

            # (batch_size, num_bins)
            Q_s_cond_x = q_s_cond_x * np.expand_dims(bin_widths, axis=0)
            Q_s_cond_x = Q_s_cond_x / np.maximum(np.sum(Q_s_cond_x, axis=1, keepdims=True), eps)
            # print("sort(sum(Q_s_cond_x)) (before): {}".format(np.sort(np.sum(Q_s_cond_x, axis=-1))))

            Q_s_cond_x = np.where(np.expand_dims(np.less(max_q_s_cond_x, 1e-5), axis=-1),
                                  deter_s_cond_x, Q_s_cond_x)
            # print("sort(sum(Q_s_cond_x)) (after): {}".format(np.sort(np.sum(Q_s_cond_x, axis=-1))))

            all_Q_s_cond_x.append(Q_s_cond_x)

        # (num_samples, num_bins)
        all_Q_s_cond_x = np.concatenate(all_Q_s_cond_x, axis=0)
        assert np.all(all_Q_s_cond_x >= 0), "'all_Q_s_cond_x' contains negative values. " \
                                            "sorted_all_Q_s_cond_x[:30]:\n{}!".format(
            np.sort(all_Q_s_cond_x[:30], axis=None))

        assert len(all_Q_s_cond_x) == len(all_attrs), "all_Q_s_cond_x.shape={}, all_attrs.shape={}".format(
            all_Q_s_cond_x.shape, all_attrs.shape)

        # I(z, y)
        for k in range(num_attrs):
            # Compute Q(zi, yk)
            # -------------------------------- #
            # (z_dim, 2)
            Q_zi_yk = np.matmul(np.transpose(all_Q_s_cond_x, axes=[1, 0]), all_Q_y_cond_x[:, k, :])
            Q_zi_yk = Q_zi_yk / len(all_Q_y_cond_x)
            Q_zi_yk = Q_zi_yk / np.maximum(np.sum(Q_zi_yk), eps)

            assert np.all(Q_zi_yk >= 0), "'Q_zi_yk' contains negative values. " \
                "sorted_Q_zi_yk[:10]:\n{}!".format(np.sort(Q_zi_yk, axis=None))

            log_Q_zi_yk = np.log(np.clip(Q_zi_yk, eps, 1 - eps))

            Q_z_y[i, k] = Q_zi_yk
            print_("sum(Q_zi_yk): {}".format(np.sum(Q_zi_yk)))
            # -------------------------------- #

            # Compute Q_z
            # -------------------------------- #
            Q_zi = np.sum(Q_zi_yk, axis=1)
            log_Q_zi = np.log(np.clip(Q_zi, eps, 1-eps))
            print_("sum(Q_z{}): {}".format(i, np.sum(Q_zi)))
            print_("Q_z{}: {}".format(i, Q_zi))
            # -------------------------------- #

            # Compute Q_y
            # -------------------------------- #
            Q_yk = np.sum(Q_zi_yk, axis=0)
            log_Q_yk = np.log(np.clip(Q_yk, eps, 1-eps))
            print_("sum(Q_y{}): {}".format(k, np.sum(Q_yk)))
            print_("Q_y{}: {}".format(k, np.sum(Q_yk)))
            # -------------------------------- #

            MI_zi_yk = Q_zi_yk * (log_Q_zi_yk -
                                  np.expand_dims(log_Q_zi, axis=-1) -
                                  np.expand_dims(log_Q_yk, axis=0))

            MI_zi_yk = np.sum(MI_zi_yk)
            H_zi_yk = -np.sum(Q_zi_yk * log_Q_zi_yk)
            H_zi = -np.sum(Q_zi * log_Q_zi)
            H_yk = -np.sum(Q_yk * log_Q_yk)

            MI_z_y[i, k] = MI_zi_yk
            H_z_y[i, k] = H_zi_yk
            H_z_4_diff_y[i, k] = H_zi
            H_y_4_diff_z[k, i] = H_yk
    # ================================= #

    print_("")
    print_("MI_z_y:\n{}".format(MI_z_y))
    print_("H_z_y:\n{}".format(H_z_y))
    print_("H_z_4_diff_y:\n{}".format(H_z_4_diff_y))
    print_("H_y_4_diff_z:\n{}".format(H_y_4_diff_z))

    # Compute metric
    # ================================= #
    # Sorted in decreasing order
    MI_ids_sorted = np.argsort(MI_z_y, axis=0)[::-1]
    MI_sorted = np.take_along_axis(MI_z_y, MI_ids_sorted, axis=0)

    MI_gap_y = np.divide(MI_sorted[0, :] - MI_sorted[1, :], H_y_4_diff_z[:, 0])
    MIG = np.mean(MI_gap_y)

    print_("")
    print_("MI_sorted: {}".format(MI_sorted))
    print_("MI_ids_sorted: {}".format(MI_ids_sorted))
    print_("MI_gap_y: {}".format(MI_gap_y))
    print_("MIG: {}".format(MIG))

    results = {
        'Q_z_y': Q_z_y,
        'MI_z_y': MI_z_y,
        'H_z_y': H_z_y,
        'H_z_4_diff_y': H_z_4_diff_y,
        'H_y_4_diff_z': H_y_4_diff_z,
        'MI_sorted': MI_sorted,
        'MI_ids_sorted': MI_ids_sorted,
        'MI_gap_y': MI_gap_y,
        'MIG': MIG,
    }

    result_file = join(save_dir, 'results[bins={},bin_limits={},data={}].npz'.
                       format(num_bins, bin_limits, data_proportion))
    np.savez_compressed(result_file, **results)
    # ================================= #

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
