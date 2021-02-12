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
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.general import make_dir_if_not_exist, print_both
from my_utils.python_utils.training import iterate_data
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from utils.metrics.mig_score_locatello import compute_mig
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
        y_train = f['latents_classes'][:, 1:]

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
        disc_z = DiscriminatorZ_1Konny(num_outputs=2)
    else:
        raise ValueError("Do not support enc_dec_model='{}'!".format(args.enc_dec_model))

    model = FactorVAE([64, 64, 1], args.z_dim,
                      encoder=encoder, decoder=decoder,
                      discriminator_z=disc_z,
                      rec_x_mode=args.rec_x_mode,
                      use_gp0_z_tc=True, gp0_z_tc_mode=args.gp0_z_tc_mode)

    loss_coeff_dict = {
        'rec_x': args.rec_x_coeff,
        'kld_loss': args.kld_loss_coeff,
        'tc_loss': args.tc_loss_coeff,
        'gp0_z_tc': args.gp0_z_tc_coeff,
        'Dz_tc_loss_coeff': args.Dz_tc_loss_coeff,
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
    data_proportion = args.data_proportion
    num_data = int(data_proportion * num_train)
    assert num_data == num_train, "For dSprites, you must use all data!"

    # file
    f = open(join(save_dir, 'log[bins={},data={}].txt'.
                  format(num_bins, data_proportion)), mode='w')

    # print function
    print_ = functools.partial(print_both, file=f)

    print_("num_bins: {}".format(num_bins))
    print_("data_proportion: {}".format(data_proportion))

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

    print_("")
    print_("all_z_mean.shape: {}".format(all_z_mean.shape))
    print_("all_z_stddev.shape: {}".format(all_z_stddev.shape))
    # ================================= #

    # Transpose and compute MIG score
    # ================================= #
    assert len(all_z_mean) == len(y_train)

    # (num_latents, num_samples)
    all_z_mean = np.transpose(all_z_mean, [1, 0])
    print_("")
    print_("all_z_mean.shape: {}".format(all_z_mean.shape))

    y_train = np.transpose(y_train, [1, 0])
    print_("")
    print_("y_train.shape: {}".format(y_train.shape))

    # All
    # --------------------------------- #
    result_all = compute_mig(all_z_mean, y_train, is_discrete_z=False,
                             is_discrete_y=True, num_bins=num_bins)

    # (num_latents, num_factors)
    MI_gap_y = result_all['MI_gap_y']
    attr_ids_sorted = np.argsort(MI_gap_y, axis=0)[::-1].tolist()
    MI_gap_y_sorted = MI_gap_y[attr_ids_sorted].tolist()

    print_("")
    print_("MIG: {}".format(result_all['MIG']))
    print_("Sorted factors:\n{}".format(list(zip(attr_ids_sorted, MI_gap_y_sorted))))

    save_file = join(save_dir, "results[bins={},data={}].npz".format(num_bins, data_proportion))
    np.savez_compressed(save_file, **result_all)
    # --------------------------------- #
    # ================================= #

    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
