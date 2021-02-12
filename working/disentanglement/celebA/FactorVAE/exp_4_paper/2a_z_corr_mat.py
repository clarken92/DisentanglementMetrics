import argparse
from os.path import join
import json

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.general import make_dir_if_not_exist, remove_dir_if_exist
from my_utils.python_utils.training import iterate_data

from my_utils.tensorflow_utils.image import TFCelebALoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter

from utils.visualization import plot_corrmat, plot_corrmat_with_histogram


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
        assert args.z_dim == 65, "For 1Konny, z_dim must be 65. Found {}!".format(args.z_dim)

        encoder = Encoder_1Konny(args.z_dim, stochastic=True, activation=activation)
        decoder = Decoder_1Konny([img_height, img_width, 3], activation=activation,
                                 output_activation=tf.nn.sigmoid)
        disc_z = DiscriminatorZ_1Konny(num_outputs=2)
    elif args.enc_dec_model == "my":
        assert args.z_dim == 150, "For 1Konny, z_dim must be 150. Found {}!".format(args.z_dim)

        encoder = Encoder_My(args.z_dim, stochastic=True, activation=activation)
        decoder = Decoder_My([img_height, img_width, 3], activation=activation,
                             output_activation=tf.nn.sigmoid)
        disc_z = DiscriminatorZ_My(num_outputs=2)
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
    # =====================================

    # z correlation matrix
    # ======================================= #
    for deterministic in [True, False]:
        all_z = []

        for batch_ids in iterate_data(celebA_loader.num_train_data, args.batch_size, shuffle=False):
            x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)

            z = model.encode(sess, x, deterministic=deterministic)
            assert len(z.shape) == 2 and z.shape[1] == args.z_dim, "z.shape: {}".format(z.shape)

            all_z.append(z)

        all_z = np.concatenate(all_z, axis=0)

        # plot_corrmat(join(save_dir, "corr_mat[deter={}].png".format(deterministic)), all_z,
        #              font={'size': 14},
        #              subplot_adjust={'left': 0.04, 'right': 0.96, 'bottom': 0.02, 'top': 0.98},
        #              size_inches=(7.2, 6))

        plot_corrmat_with_histogram(join(save_dir, "corr_mat_hist[deter={}].png".format(deterministic)), all_z,
                                    font={'size': 14},
                                    subplot_adjust={'left': 0.04, 'right': 0.96, 'bottom': 0.02, 'top': 0.98},
                                    size_inches=(10, 3))
    # ======================================= #


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
