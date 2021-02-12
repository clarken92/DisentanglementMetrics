import argparse
from os.path import join, exists
import json

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist, print_both
from my_utils.python_utils.training import iterate_data
from my_utils.python_utils.arg_parsing import str2bool
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from utils.metrics.sampling import estimate_SEPIN_cupy
from global_settings import RAW_DATA_DIR


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--num_samples', type=int, default=10000)
parser.add_argument('--batch', type=int, default=5)
parser.add_argument('--gpu_support', type=str, default='')
parser.add_argument('--gpu_id', type=int, default=0)


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
    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=5, suppress=True)
    # =====================================

    # Compute representations
    # ================================= #
    z_data_file = join(save_dir, "z_data.npz")

    if not exists(z_data_file):
        all_z_mean = []
        all_z_stddev = []

        print("")
        print("Compute all_z_mean, all_z_stddev!")

        count = 0
        for batch_ids in iterate_data(num_train, 10 * args.batch_size, shuffle=False):
            x = x_train[batch_ids]

            z_samples, z_mean, z_stddev = sess.run(
                model.get_output(['z1_gen', 'z_mean', 'z_stddev']),
                feed_dict={model.is_train: False, model.x_ph: x})

            all_z_mean.append(z_mean)
            all_z_stddev.append(z_stddev)

            count += len(batch_ids)
            print("\rProcessed {} samples!".format(count), end="")
        print()

        all_z_mean = np.concatenate(all_z_mean, axis=0)
        all_z_stddev = np.concatenate(all_z_stddev, axis=0)

        np.savez_compressed(z_data_file, all_z_mean=all_z_mean, all_z_stddev=all_z_stddev)
    else:
        print("{} exists. Load data from file!".format(z_data_file))
        with np.load(z_data_file, "r") as f:
            all_z_mean = f['all_z_mean']
            all_z_stddev = f['all_z_stddev']
    # ================================= #

    if args.gpu_support == 'cupy':
        print("Use cupy instead of numpy!")
        results = estimate_SEPIN_cupy(all_z_mean, all_z_stddev, num_samples=args.num_samples,
                                      batch_size=args.batch, gpu=args.gpu_id)
    else:
        raise NotImplementedError("There is no numpy support since it is too slow!")

    result_file = join(save_dir, "results[num_samples={}].npz".format(args.num_samples))
    np.savez_compressed(result_file, **results)
    f.close()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
