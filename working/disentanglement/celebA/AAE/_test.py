import argparse
import os
import json

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.tensorflow_utils.image import TFCelebALoader
from my_utils.python_utils.image import binary_float_to_uint8
from my_utils.python_utils.training import iterate_data
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist
from utils.visualization import plot_comp_dist, plot_corrmat_with_histogram

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
    celebA_loader = TFCelebALoader(root_dir=args.celebA_root_dir)
    num_train = celebA_loader.num_train_data
    num_test = celebA_loader.num_test_data

    img_height, img_width = args.celebA_resize_size, args.celebA_resize_size
    celebA_loader.build_transformation_flow_tf(
        *celebA_loader.get_transform_fns("1Konny", resize_size=args.celebA_resize_size))

    # =====================================
    # Instantiate models
    # =====================================
    # Only use activation for encoder and decoder
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
    SimpleParamPrinter.print_all_params_list()

    # =====================================
    # TF Graph Handler
    asset_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "asset"))
    img_eval = remove_dir_if_exist(os.path.join(asset_dir, "img_eval"), ask_4_permission=False)
    img_eval = make_dir_if_not_exist(img_eval)

    img_x_gen = make_dir_if_not_exist(os.path.join(img_eval, "x_gen"))
    img_x_rec = make_dir_if_not_exist(os.path.join(img_eval, "x_rec"))
    img_z_rand_2_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_rand_2_traversal"))
    img_z_cond_all_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_cond_all_traversal"))
    img_z_cond_1_traversal = make_dir_if_not_exist(os.path.join(img_eval, "z_cond_1_traversal"))
    img_z_corr = make_dir_if_not_exist(os.path.join(img_eval, "z_corr"))
    img_z_dist = make_dir_if_not_exist(os.path.join(img_eval, "z_dist"))
    img_z_stat_dist = make_dir_if_not_exist(os.path.join(img_eval, "z_stat_dist"))
    # img_rec_error_dist = make_dir_if_not_exist(os.path.join(img_eval, "rec_error_dist"))

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

    # '''
    # Generation
    # ======================================= #
    z = np.random.randn(64, args.z_dim)

    img_file = os.path.join(img_x_gen, 'x_gen_test.png')
    model.generate_images(img_file, sess, z, block_shape=[8, 8],
                          batch_size=args.batch_size,
                          dec_output_2_img_func=binary_float_to_uint8)
    # ======================================= #
    # '''

    # '''
    # Reconstruction
    # ======================================= #
    seed = 389
    x = celebA_loader.sample_images_from_dataset(sess, 'test', list(range(seed, seed+64)))

    img_file = os.path.join(img_x_rec, 'x_rec_test.png')
    model.reconstruct_images(img_file, sess, x, block_shape=[8, 8], batch_size=-1,
                             dec_output_2_img_func=binary_float_to_uint8)
    # ======================================= #
    # '''

    # '''
    # z random traversal
    # ======================================= #
    if args.z_dim <= 5:
        print("z_dim = {}, perform random traversal!".format(args.z_dim))

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
    # ======================================= #
    # '''

    # z conditional traversal (all features + one feature)
    # ======================================= #
    seed = 389
    num_samples = 30
    data = celebA_loader.sample_images_from_dataset(sess, 'train', list(range(seed, seed + num_samples)))

    z_start, z_stop = -4, 4
    num_itpl_points = 8
    for n in range(num_samples):
        print("Plot conditional all comps z traverse with test sample {}!".format(n))
        img_file = os.path.join(img_z_cond_all_traversal, 'x_train{}.png'.format(n))
        model.cond_all_latents_traverse(img_file, sess, data[n],
                                        start=z_start, stop=z_stop,
                                        num_itpl_points=num_itpl_points,
                                        batch_size=args.batch_size,
                                        dec_output_2_img_func=binary_float_to_uint8)

    z_start, z_stop = -4, 4
    num_itpl_points = 8
    for i in range(args.z_dim):
        print("Plot conditional z traverse with comp {}!".format(i))
        img_file = os.path.join(img_z_cond_1_traversal, 'x_train[{},{}]_z{}.png'.format(seed, seed + num_samples, i))
        model.cond_1_latent_traverse(img_file, sess, data,
                                     z_comp=i, start=z_start, stop=z_stop,
                                     num_itpl_points=num_itpl_points,
                                     batch_size=args.batch_size,
                                     dec_output_2_img_func=binary_float_to_uint8)
    # ======================================= #
    # '''

    # '''
    # z correlation matrix
    # ======================================= #
    all_z = []
    for batch_ids in iterate_data(num_train, args.batch_size, shuffle=False):
        x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)

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

    # '''
    # z gaussian stddev
    # ======================================= #
    print("\nPlot z mean and stddev!")
    all_z_mean = []
    all_z_stddev = []

    for batch_ids in iterate_data(num_train, args.batch_size, shuffle=False):
        x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)

        z_mean, z_stddev = sess.run(model.get_output(['z_mean', 'z_stddev']),
                                    feed_dict={model.is_train: False, model.x_ph: x})

        all_z_mean.append(z_mean)
        all_z_stddev.append(z_stddev)

    all_z_mean = np.concatenate(all_z_mean, axis=0)
    all_z_stddev = np.concatenate(all_z_stddev, axis=0)

    plot_comp_dist(os.path.join(img_z_stat_dist, 'z_mean_{}.png'), all_z_mean, x_lim=(-5, 5))
    plot_comp_dist(os.path.join(img_z_stat_dist, 'z_stddev_{}.png'), all_z_stddev, x_lim=(-0.5, 3))
    # ======================================= #
    # '''


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
