import argparse
import os
import json

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.image import binary_float_to_uint8

from my_utils.python_utils.training import iterate_data
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist
from utils.visualization import plot_comp_dist, plot_corrmat_with_histogram

from global_settings import RAW_DATA_DIR


parser = argparse.ArgumentParser()

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--load_step', type=int, default=-1)


def main(args):
    # =====================================
    # Load config
    # =====================================
    with open(os.path.join(args.output_dir, 'config.json')) as f:
        config = json.load(f)
    args.__dict__.update(config)

    # =====================================
    # Preparation
    # =====================================
    data_file = os.path.join(RAW_DATA_DIR, "ComputerVision", "dSprites",
                             "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    # It is already in the range [0, 1]
    with np.load(data_file, encoding="latin1") as f:
        x_train = f['imgs']

    x_train = np.expand_dims(x_train.astype(np.float32), axis=-1)

    # =====================================
    # Instantiate models
    # =====================================
    encoder = Encoder_1Konny(args.z_dim, stochastic=True)
    decoder = Decoder_1Konny()
    disc_z = DiscriminatorZ_1Konny(num_outputs=2)

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
    }

    model.build(loss_coeff_dict)
    SimpleParamPrinter.print_all_params_list()

    # =====================================
    # TF Graph Handler
    asset_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "asset"))
    img_eval = remove_dir_if_exist(os.path.join(asset_dir, "img_eval"), ask_4_permission=False)
    img_eval = make_dir_if_not_exist(img_eval)

    img_x_rec = make_dir_if_not_exist(os.path.join(img_eval, "x_rec"))
    img_z_rand_2_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_rand_2_traversal"))
    img_z_cond_all_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_cond_all_traversal"))
    img_z_cond_1_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_cond_1_traversal"))
    img_z_corr = make_dir_if_not_exist(os.path.join(img_eval, "z_corr"))
    img_z_dist = make_dir_if_not_exist(os.path.join(img_eval, "z_dist"))
    img_z_stat_dist = make_dir_if_not_exist(os.path.join(img_eval, "z_stat_dist"))
    img_rec_error_dist = make_dir_if_not_exist(os.path.join(img_eval, "rec_error_dist"))

    model_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "model_tf"))

    train_helper = SimpleTrainHelper(log_dir=None, save_dir=model_dir)
    # =====================================

    # =====================================
    # Training Loop
    # =====================================
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config_proto)

    # Load model
    train_helper.load(sess, load_step=args.load_step)

    #'''
    # Reconstruction
    # ======================================= #
    seed = 389
    x = x_train[np.arange(seed, seed + 64)]

    img_file = os.path.join(img_x_rec, 'x_rec_train.png')
    model.reconstruct_images(img_file, sess, x, block_shape=[8, 8], batch_size=-1,
                             dec_output_2_img_func=binary_float_to_uint8)
    # ======================================= #

    # z random/conditional traversal
    # ======================================= #
    # Plot z cont with z cont
    z_zero = np.zeros([args.z_dim], dtype=np.float32)
    z_rand = np.random.randn(args.z_dim)
    z_start, z_stop = -4, 4
    num_points = 8

    for i in range(args.z_dim):
        for j in range(i + 1, args.z_dim):
            print("Plot random 2 comps z traverse with {} and {} components!".format(i, j))

            img_file = os.path.join(img_z_rand_2_traversal, 'z[{},{},zero].png'.format(i, j))
            model.rand_2_latents_traverse(img_file, sess, default_z=z_zero,
                                          z_comp1=i, start1=z_start, stop1=z_stop, num_points1=num_points,
                                          z_comp2=j, start2=z_start, stop2=z_stop, num_points2=num_points,
                                          batch_size=args.batch_size,
                                          dec_output_2_img_func=binary_float_to_uint8)

            img_file = os.path.join(img_z_rand_2_traversal, 'z[{},{},rand].png'.format(i, j))
            model.rand_2_latents_traverse(img_file, sess, default_z=z_rand,
                                          z_comp1=i, start1=z_start, stop1=z_stop, num_points1=num_points,
                                          z_comp2=j, start2=z_stop, stop2=z_stop, num_points2=num_points,
                                          batch_size=args.batch_size,
                                          dec_output_2_img_func=binary_float_to_uint8)

    seed = 389
    z_start, z_stop = -4, 4
    num_itpl_points = 8

    for n in range(seed, seed+30):
        print("Plot conditional all comps z traverse with test sample {}!".format(n))

        x = x_train[n]
        img_file = os.path.join(img_z_cond_all_traversal, 'x_train{}.png'.format(n))
        model.cond_all_latents_traverse(img_file, sess, x,
                                        start=z_start, stop=z_stop,
                                        num_itpl_points=num_itpl_points,
                                        batch_size=args.batch_size,
                                        dec_output_2_img_func=binary_float_to_uint8)

    seed = 64
    z_start, z_stop = -4, 4
    num_itpl_points = 8
    print("Plot conditional 1 comp z traverse!")
    for i in range(args.z_dim):
        x = x_train[seed: seed+64]
        img_file = os.path.join(img_z_cond_1_traversal, 'x_train[{},{}]_z{}.png'.format(seed, seed+64, i))
        model.cond_1_latent_traverse(img_file, sess, x,
                                     z_comp=i, start=z_start, stop=z_stop,
                                     num_itpl_points=num_itpl_points,
                                     batch_size=args.batch_size,
                                     dec_output_2_img_func=binary_float_to_uint8)
    # ======================================= #
    # '''

    # z correlation matrix
    # ======================================= #
    data = x_train

    all_z = []
    for batch_ids in iterate_data(len(data), args.batch_size, shuffle=False):
        x = data[batch_ids]

        z = model.encode(sess, x)
        assert len(z.shape) == 2 and z.shape[1] == args.z_dim, "z.shape: {}".format(z.shape)

        all_z.append(z)

    all_z = np.concatenate(all_z, axis=0)

    print("Start plotting!")
    plot_corrmat_with_histogram(os.path.join(img_z_corr, "corr_mat.png"), all_z)
    plot_comp_dist(os.path.join(img_z_dist, 'z_{}'), all_z, x_lim=(-5, 5))
    print("Done!")
    # ======================================= #

    # '''
    # z gaussian stddev
    # ======================================= #
    print("\nPlot z mean and stddev!")
    data = x_train

    all_z_mean = []
    all_z_stddev = []

    for batch_ids in iterate_data(len(data), args.batch_size, shuffle=False):
        x = data[batch_ids]

        z_mean, z_stddev = sess.run(model.get_output(['z_mean', 'z_stddev']),
                                    feed_dict={model.is_train: False, model.x_ph: x})

        all_z_mean.append(z_mean)
        all_z_stddev.append(z_stddev)

    all_z_mean = np.concatenate(all_z_mean, axis=0)
    all_z_stddev = np.concatenate(all_z_stddev, axis=0)

    plot_comp_dist(os.path.join(img_z_stat_dist, 'z_mean_{}.png'), all_z_mean, x_lim=(-5, 5))
    plot_comp_dist(os.path.join(img_z_stat_dist, 'z_stddev_{}.png'), all_z_stddev, x_lim=(0, 3))
    # ======================================= #
    # '''

    # Decoder sensitivity
    # ======================================= #
    z_start = -3
    z_stop = 3
    for i in range(args.z_dim):
        print("\nPlot rec error distribution for z component {}!".format(i))

        all_z1 = np.array(all_z, copy=True, dtype=np.float32)
        all_z2 = np.array(all_z, copy=True, dtype=np.float32)

        all_z1[:, i] = z_start
        all_z2[:, i] = z_stop

        all_x_rec1 = []
        all_x_rec2 = []
        for batch_ids in iterate_data(len(x_train), args.batch_size, shuffle=False):
            z1 = all_z1[batch_ids]
            z2 = all_z2[batch_ids]

            x1 = model.decode(sess, z1)
            x2 = model.decode(sess, z2)

            all_x_rec1.append(x1)
            all_x_rec2.append(x2)

        all_x_rec1 = np.concatenate(all_x_rec1, axis=0)
        all_x_rec2 = np.concatenate(all_x_rec2, axis=0)

        rec_errors = np.sum(np.reshape((all_x_rec1 - all_x_rec2)**2, [len(x_train), 28 * 28]), axis=1)
        plot_comp_dist(os.path.join(img_rec_error_dist, 'rec_error[zi={},{},{}].png'.format(
            i, z_start, z_stop)), rec_errors)
    # ======================================= #


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
