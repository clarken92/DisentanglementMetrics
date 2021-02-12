import argparse
from os.path import join, exists
import json

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.training import iterate_data
from my_utils.python_utils.arg_parsing import str2bool
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from utils.metrics.metrics_eastwood import compute_metrics_with_LASSO, compute_metrics_with_RandomForest
from global_settings import RAW_DATA_DIR


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)

parser.add_argument('--save_dir', type=str, required=True)
parser.add_argument('--continuous_only', type=str2bool, required=True)
parser.add_argument('--classifier', type=str, required=True)
parser.add_argument('--LASSO_alpha', type=float, default=0.02)
parser.add_argument('--LASSO_iters', type=int, default=1000)
parser.add_argument('--RF_trees', type=int, default=10)
parser.add_argument('--RF_depth', type=int, default=8)


def main(args):
    # Load config
    # ===================================== #
    with open(join(args.output_dir, 'config.json')) as f:
        config = json.load(f)
    args.__dict__.update(config)
    # ===================================== #

    # Load dataset
    # ===================================== #
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
    # ===================================== #

    # Build model
    # ===================================== #
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
    # ===================================== #

    # Initialize session
    # ===================================== #
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config_proto)

    model_dir = make_dir_if_not_exist(join(args.output_dir, "model_tf"))
    train_helper = SimpleTrainHelper(log_dir=None, save_dir=model_dir)
    train_helper.load(sess, load_step=args.load_step)
    # ===================================== #

    # Experiments
    # ===================================== #
    save_dir = make_dir_if_not_exist(join(args.save_dir, "{}_{}".format(args.enc_dec_model, args.run)))
    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=5, suppress=True)
    # ===================================== #

    # Compute representations
    # ===================================== #
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
    # ===================================== #

    cont_mask = [False, True, True, True, True] if args.continuous_only else None

    if args.classifier == "LASSO":
        results = compute_metrics_with_LASSO(latents=all_z_mean, factors=y_train,
                                             params={'alpha': args.LASSO_alpha,
                                                     'max_iter': args.LASSO_iters},
                                             cont_mask=cont_mask)
        result_file = join(save_dir, "results[LASSO,{},alpha={},iters={}].npz".format(
            "cont" if args.continuous_only else "all", args.LASSO_alpha, args.LASSO_iters))
    else:
        results = compute_metrics_with_RandomForest(latents=all_z_mean, factors=y_train,
                                                    params={'n_estimators': args.RF_trees,
                                                            'max_depth': args.RF_depth})
        result_file = join(save_dir, "results[RF,{},trees={},depth={}].npz".format(
            "cont" if args.continuous_only else "all", args.RF_trees, args.RF_depth))

    np.savez_compressed(result_file, **results)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
