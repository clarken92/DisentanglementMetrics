import argparse
from os.path import join, exists
import json
import functools

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, \
    DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist, print_both
from my_utils.python_utils.training import iterate_data
from my_utils.python_utils.general import one_hot
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

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
    save_dir = make_dir_if_not_exist(join(args.save_dir, "{}_{}".format(
        args.enc_dec_model, args.run)))
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

    print_("")
    all_Q_z_cond_x = []
    for i in range(args.z_dim):
        print_("\nCompute all_Q_z{}_cond_x!".format(i))

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
            "sorted_all_Q_s_cond_x[:30]:\n{}!".format(np.sort(all_Q_s_cond_x[:30], axis=None))
        assert len(all_Q_s_cond_x) == num_train

        all_Q_z_cond_x.append(all_Q_s_cond_x)

    # (z_dim, num_samples, num_bins)
    all_Q_z_cond_x = np.asarray(all_Q_z_cond_x, dtype=np.float32)
    print_("all_Q_z_cond_x.shape: {}".format(all_Q_z_cond_x.shape))
    print_("sum(all_Q_z_cond_x)[:, :10]:\n{}".format(np.sum(all_Q_z_cond_x, axis=-1)[:, :10]))

    # (z_dim, num_bins)
    Q_z = np.mean(all_Q_z_cond_x, axis=1)
    log_Q_z = np.log(np.clip(Q_z, eps, 1-eps))
    print_("Q_z.shape: {}".format(Q_z.shape))
    print_("sum(Q_z): {}".format(np.sum(Q_z, axis=-1)))

    # (z_dim, )
    H_z = -np.sum(Q_z * log_Q_z, axis=-1)

    # Factors
    gt_factors = ['shape', 'scale', 'rotation', 'pos_x', 'pos_y']
    gt_num_values = [3, 6, 40, 32, 32]

    MI_z_y = np.zeros([args.z_dim, len(gt_factors)], dtype=np.float32)
    H_z_y = np.zeros([args.z_dim, len(gt_factors)], dtype=np.float32)

    ids_sorted = np.zeros([args.z_dim, len(gt_factors)], dtype=np.int32)
    MI_z_y_sorted = np.zeros([args.z_dim, len(gt_factors)], dtype=np.float32)
    H_z_y_sorted = np.zeros([args.z_dim, len(gt_factors)], dtype=np.float32)

    H_y = []
    RMIG = []
    JEMMI = []

    for k, (factor, num_values) in enumerate(zip(gt_factors, gt_num_values)):
        print_("\n#" + "=" * 50 + "#")
        print_("The {}-th gt factor '{}' has {} values!".format(k, factor, num_values))

        print_("")
        # (num_samples, num_categories)
        # NOTE: We must use k+1 to account for the 'color' attribute, which is always white
        all_Q_yk_cond_x = one_hot(y_train[:, k+1], num_categories=num_values, dtype=np.float32)
        print_("all_Q_yk_cond_x.shape: {}".format(all_Q_yk_cond_x.shape))

        # (num_categories)
        Q_yk = np.mean(all_Q_yk_cond_x, axis=0)
        log_Q_yk = np.log(np.clip(Q_yk, eps, 1-eps))
        print_("Q_yk.shape: {}".format(Q_yk.shape))

        H_yk = -np.sum(Q_yk * log_Q_yk)
        print_("H_yk: {}".format(H_yk))
        H_y.append(H_yk)

        Q_z_yk = np.zeros([args.z_dim, num_bins, num_values], dtype=np.float32)

        # Compute I(zi, yk)
        for i in range(args.z_dim):
            print_("\n#" + "-" * 50 + "#")
            all_Q_zi_cond_x = all_Q_z_cond_x[i]
            assert len(all_Q_zi_cond_x) == len(all_Q_yk_cond_x) == num_train, \
                "all_Q_zi_cond_x.shape: {}, all_Q_yk_cond_x.shape: {}".format(
                    all_Q_zi_cond_x.shape, all_Q_yk_cond_x.shape)

            # (num_bins, num_categories)
            Q_zi_yk = np.matmul(np.transpose(all_Q_zi_cond_x, axes=[1, 0]), all_Q_yk_cond_x)
            Q_zi_yk = Q_zi_yk / num_train
            print_("np.sum(Q_zi_yk): {}".format(np.sum(Q_zi_yk)))
            Q_zi_yk = Q_zi_yk / np.maximum(np.sum(Q_zi_yk), eps)
            print_("np.sum(Q_zi_yk) (normalized): {}".format(np.sum(Q_zi_yk)))

            assert np.all(Q_zi_yk >= 0), "'Q_zi_yk' contains negative values. " \
                "sorted_Q_zi_yk[:10]:\n{}!".format(np.sort(Q_zi_yk, axis=None))

            # (num_bins, num_categories)
            log_Q_zi_yk = np.log(np.clip(Q_zi_yk, eps, 1 - eps))

            print_("")
            print_("Q_zi (default): {}".format(Q_z[i]))
            print_("Q_zi (sum of Q_zi_yk over yk): {}".format(np.sum(Q_zi_yk, axis=-1)))

            print_("")
            print_("Q_yk (default): {}".format(Q_yk))
            print_("Q_yk (sum of Q_zi_yk over zi): {}".format(np.sum(Q_zi_yk, axis=0)))

            MI_zi_yk = Q_zi_yk * (log_Q_zi_yk -
                                  np.expand_dims(log_Q_z[i], axis=-1) -
                                  np.expand_dims(log_Q_yk, axis=0))

            MI_zi_yk = np.sum(MI_zi_yk)
            H_zi_yk = -np.sum(Q_zi_yk * log_Q_zi_yk)

            Q_z_yk[i] = Q_zi_yk
            MI_z_y[i, k] = MI_zi_yk
            H_z_y[i, k] = H_zi_yk

            print_("#" + "-" * 50 + "#")

        # Print statistics for all z
        print_("")
        print_("MI_z_yk:\n{}".format(MI_z_y[:, k]))
        print_("H_z_yk:\n{}".format(H_z_y[:, k]))
        print_("H_z:\n{}".format(H_z))
        print_("H_yk:\n{}".format(H_yk))

        # Compute RMIG and JEMMI
        ids_yk_sorted = np.argsort(MI_z_y[:, k], axis=0)[::-1]
        MI_z_yk_sorted = np.take_along_axis(MI_z_y[:, k], ids_yk_sorted, axis=0)
        H_z_yk_sorted = np.take_along_axis(H_z_y[:, k], ids_yk_sorted, axis=0)

        RMIG_yk = np.divide(MI_z_yk_sorted[0] - MI_z_yk_sorted[1], H_yk)
        JEMMI_yk = np.divide(H_z_yk_sorted[0] - MI_z_yk_sorted[0] + MI_z_yk_sorted[1],
                             H_yk + np.log(num_bins))

        ids_sorted[:, k] = ids_yk_sorted
        MI_z_y_sorted[:, k] = MI_z_yk_sorted
        H_z_y_sorted[:, k] = H_z_yk_sorted

        RMIG.append(RMIG_yk)
        JEMMI.append(JEMMI_yk)

        print_("")
        print_("ids_sorted: {}".format(ids_sorted))
        print_("MI_z_yk_sorted: {}".format(MI_z_yk_sorted))
        print_("RMIG_yk: {}".format(RMIG_yk))
        print_("JEMMI_yk: {}".format(JEMMI_yk))

        z_yk_prob_file = join(save_dir, "z_yk_prob_4_{}[bins={},bin_limits={},data={}].npz".
                              format(factor, num_bins, bin_limits, data_proportion))
        np.savez_compressed(z_yk_prob_file, Q_z_yk=Q_z_yk)
        print_("#" + "=" * 50 + "#")

    results = {
        "MI_z_y": MI_z_y,
        "H_z_y": H_z_y,
        "ids_sorted": ids_sorted,
        "MI_z_y_sorted": MI_z_y_sorted,
        "H_z_y_sorted": H_z_y_sorted,
        "H_z": H_z,
        "H_y": np.asarray(H_y, dtype=np.float32),
        "RMIG": np.asarray(RMIG, dtype=np.float32),
        "JEMMI": np.asarray(JEMMI, dtype=np.float32),
    }
    result_file = join(save_dir, "results[bins={},bin_limits={},data={}].npz".
                       format(num_bins, bin_limits, data_proportion))
    np.savez_compressed(result_file, **results)

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
