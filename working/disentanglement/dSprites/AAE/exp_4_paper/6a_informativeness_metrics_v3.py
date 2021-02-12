import argparse
from os.path import join, exists
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, print_both
from my_utils.python_utils.training import iterate_data
from my_utils.tensorflow_utils.training.helper import \
    SimpleTrainHelper, SimpleParamPrinter

from utils.general import normal_density, at_bin
from global_settings import RAW_DATA_DIR


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
    data_file = join(RAW_DATA_DIR, "ComputerVision", "dSprites",
                     "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    # It is already in the range [0, 1]
    with np.load(data_file, encoding="latin1") as f:
        x_train = f['imgs']
        # 3 shape * 6 scale * 40 rotation * 32 pos X * 32 pos Y
        y_train = f['latents_classes']

    x_train = np.expand_dims(x_train.astype(np.float32), axis=-1)
    num_train = len(x_train)
    print("num_train: {}".format(num_train))

    print("y_train[:10]: {}".format(y_train[:10]))

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
    save_dir = make_dir_if_not_exist(join(args.save_dir, "{}_{}".format(args.enc_dec_model, args.run)))
    # =====================================

    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=5, suppress=True)

    num_bins = args.num_bins
    bin_limits = tuple([float(s) for s in args.bin_limits.split(";")])
    data_proportion = args.data_proportion
    num_data = int(data_proportion * num_train)
    assert num_data == num_train, "For dSprites, you must use all data!"
    eps = 1e-8

    # file
    f = open(join(save_dir, 'log[bins={},bin_limits={},data={}].txt'.
                  format(num_bins, bin_limits, data_proportion)), mode='w')

    # print function
    print_ = functools.partial(print_both, file=f)

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
    z_data_file = join(save_dir, "z_data[data={}].npz".format(data_proportion))

    if not exists(z_data_file):
        all_z_mean = []
        all_z_stddev = []

        print("")
        print("Compute all_z_mean, all_z_stddev and all_attrs!")

        count = 0
        for batch_ids in iterate_data(num_data, 10 * args.batch_size, shuffle=False):
            x = x_train[batch_ids]

            z_mean, z_stddev = sess.run(
                model.get_output(['z_mean', 'z_stddev']),
                feed_dict={model.is_train: False, model.x_ph: x})

            all_z_mean.append(z_mean)
            all_z_stddev.append(z_stddev)

            count += len(batch_ids)
            print("\rProcessed {} samples!".format(count), end="")
        print()

        all_z_mean = np.concatenate(all_z_mean, axis=0)
        all_z_stddev = np.concatenate(all_z_stddev, axis=0)

        np.savez_compressed(z_data_file, all_z_mean=all_z_mean,
                            all_z_stddev=all_z_stddev)
    else:
        print("{} exists. Load data from file!".format(z_data_file))
        with np.load(z_data_file, "r") as f:
            all_z_mean = f['all_z_mean']
            all_z_stddev = f['all_z_stddev']
    # ================================= #

    # Compute mutual information
    # ================================= #
    H_z = []
    H_z_cond_x = []
    MI_z_x = []
    norm_MI_z_x = []
    Q_z_cond_x = []

    for i in range(args.z_dim):
        print_("")
        print_("Compute I(z{}, x)!".format(i))

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

        all_Q_s_cond_x = np.concatenate(all_Q_s_cond_x, axis=0)
        print_("sort(sum(all_Q_s_cond_x))[:10]: {}".format(
            np.sort(np.sum(all_Q_s_cond_x, axis=-1), axis=0)[:100]))
        assert np.all(all_Q_s_cond_x >= 0), "'all_Q_s_cond_x' contains negative values. " \
                                            "sorted_all_Q_s_cond_x[:30]:\n{}!".format(
            np.sort(all_Q_s_cond_x[:30], axis=None))
        Q_z_cond_x.append(all_Q_s_cond_x)

        H_zi_cond_x = -np.mean(np.sum(all_Q_s_cond_x * np.log(np.maximum(all_Q_s_cond_x, eps)), axis=1), axis=0)

        # Q_s
        Q_s = np.mean(all_Q_s_cond_x, axis=0)
        print_("Q_s: {}".format(Q_s))
        print_("sum(Q_s): {}".format(sum(Q_s)))
        assert np.all(Q_s >= 0), "'Q_s' contains negative values. " \
                                 "sorted_Q_s[:10]:\n{}!".format(np.sort(Q_s, axis=None))

        Q_s = Q_s / np.sum(Q_s, axis=0)
        print_("sum(Q_s) (normalized): {}".format(sum(Q_s)))

        H_zi = -np.sum(Q_s * np.log(np.maximum(Q_s, eps)), axis=0)

        MI_zi_x = H_zi - H_zi_cond_x
        normalized_MI_zi_x = (1.0 * MI_zi_x) / (H_zi + eps)

        print_("H_zi: {}".format(H_zi))
        print_("H_zi_cond_x: {}".format(H_zi_cond_x))
        print_("MI_zi_x: {}".format(MI_zi_x))
        print_("normalized_MI_zi_x: {}".format(normalized_MI_zi_x))

        H_z.append(H_zi)
        H_z_cond_x.append(H_zi_cond_x)
        MI_z_x.append(MI_zi_x)
        norm_MI_z_x.append(normalized_MI_zi_x)

    H_z = np.asarray(H_z, dtype=np.float32)
    H_z_cond_x = np.asarray(H_z_cond_x, dtype=np.float32)
    MI_z_x = np.asarray(MI_z_x, dtype=np.float32)
    norm_MI_z_x = np.asarray(norm_MI_z_x, dtype=np.float32)

    print_("")
    print_("H_z: {}".format(H_z))
    print_("H_z_cond_x: {}".format(H_z_cond_x))
    print_("MI_z_x: {}".format(MI_z_x))
    print_("norm_MI_z_x: {}".format(norm_MI_z_x))

    sorted_z_comps = np.argsort(MI_z_x, axis=0)[::-1]
    sorted_MI_z_x = np.take_along_axis(MI_z_x, sorted_z_comps, axis=0)
    print_("sorted_MI_z_x: {}".format(sorted_MI_z_x))
    print_("sorted_z_comps: {}".format(sorted_z_comps))

    sorted_norm_z_comps = np.argsort(norm_MI_z_x, axis=0)[::-1]
    sorted_norm_MI_z_x = np.take_along_axis(norm_MI_z_x, sorted_norm_z_comps, axis=0)
    print_("sorted_norm_MI_z_x: {}".format(sorted_norm_MI_z_x))
    print_("sorted_norm_z_comps: {}".format(sorted_norm_z_comps))

    result_file = join(save_dir, 'results[bins={},bin_limits={},data={}].npz'.
                       format(num_bins, bin_limits, data_proportion))

    np.savez_compressed(result_file,
                        H_z=H_z, H_z_cond_x=H_z_cond_x, MI_z_x=MI_z_x, norm_MI_z_x=norm_MI_z_x,
                        sorted_MI_z_x=sorted_MI_z_x, sorted_z_comps=sorted_z_comps,
                        sorted_norm_MI_z_x=sorted_norm_MI_z_x,
                        sorted_norm_z_comps=sorted_norm_z_comps)

    Q_z_cond_x = np.asarray(Q_z_cond_x, dtype=np.float32)
    z_prob_file = join(save_dir, 'z_prob[bins={},bin_limits={},data={}].npz'.
                       format(num_bins, bin_limits, data_proportion))
    np.savez_compressed(z_prob_file, Q_z_cond_x=Q_z_cond_x)
    # ================================= #

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
