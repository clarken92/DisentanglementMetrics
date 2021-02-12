import argparse
import os

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.celebA import Decoder_1Konny, Encoder_1Konny, DiscriminatorZ_1Konny
from models.generative.aae import AAE

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.arg_parsing import save_args, str2bool
from my_utils.python_utils.image import binary_float_to_uint8
from my_utils.python_utils.training import ContinuousIndexSampler

from my_utils.tensorflow_utils.image import TFCelebALoader
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter
from my_utils.tensorflow_utils.training.summary import custom_tf_scalar_summary


parser = argparse.ArgumentParser()

parser.add_argument('--celebA_root_dir', required=True, type=str)
parser.add_argument('--celebA_crop_size', type=int, default=128)
parser.add_argument('--celebA_resize_size', type=int, default=64)

parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--activation', type=str, default="relu")
parser.add_argument('--z_dim', type=int, default=65)

parser.add_argument('--lr_AE', type=float, default=1e-3)
parser.add_argument('--beta1_AE', type=float, default=0.5)
parser.add_argument('--beta2_AE', type=float, default=0.99)

parser.add_argument('--lr_Dz', type=float, default=1e-4)
parser.add_argument('--beta1_Dz', type=float, default=0.5)
parser.add_argument('--beta2_Dz', type=float, default=0.99)

parser.add_argument('--stochastic_z', type=str2bool, default="True")
parser.add_argument('--rec_x_mode', type=str, default='l1')
parser.add_argument('--rec_x_coeff', type=float, default=1.0)

parser.add_argument('--D_loss_z1_gen_coeff', type=float, default=1.0)
parser.add_argument('--G_loss_z1_gen_coeff', type=float, default=10.0)

parser.add_argument('--gp0_z_mode', type=str, default="interpolation")
parser.add_argument('--gp0_z_coeff', type=float, default=0.0)

parser.add_argument('--Dz_steps', type=int, default=1)
parser.add_argument('--AE_steps', type=int, default=1)

parser.add_argument('--log_freq', type=int, default=100)
parser.add_argument('--save_freq', type=int, default=500)

parser.add_argument('--viz_gen_freq', type=int, default=5000)
parser.add_argument('--viz_rec_freq', type=int, default=5000)
parser.add_argument('--viz_itpl_freq', type=int, default=5000)

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--enc_dec_model', required=True, type=str)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--force_rm_dir', type=str2bool, default="False")


def main(args):
    # =====================================
    # Preparation
    # =====================================
    celebA_loader = TFCelebALoader(root_dir=args.celebA_root_dir)
    num_train = celebA_loader.num_train_data
    num_test = celebA_loader.num_test_data

    img_height, img_width = args.celebA_resize_size, args.celebA_resize_size
    celebA_loader.build_transformation_flow_tf(
        *celebA_loader.get_transform_fns("1Konny", resize_size=args.celebA_resize_size))

    args.output_dir = os.path.join(args.output_dir, args.enc_dec_model, args.run)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        if args.force_rm_dir:
            import shutil
            shutil.rmtree(args.output_dir, ignore_errors=True)
            print("Removed '{}'".format(args.output_dir))
        else:
            raise ValueError("Output directory '{}' existed. 'force_rm_dir' "
                             "must be set to True!".format(args.output_dir))
        os.mkdir(args.output_dir)

    save_args(os.path.join(args.output_dir, 'config.json'), args)
    # pp.pprint(args.__dict__)

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

    loss = model.get_loss()
    train_params = model.get_train_params()

    opt_Dz = tf.train.AdamOptimizer(learning_rate=args.lr_Dz, beta1=args.beta1_Dz, beta2=args.beta2_Dz)
    opt_AE = tf.train.AdamOptimizer(learning_rate=args.lr_AE, beta1=args.beta1_AE, beta2=args.beta2_AE)

    with tf.control_dependencies(model.get_all_update_ops()):
        train_op_Dz = opt_Dz.minimize(loss=loss['Dz_loss'], var_list=train_params['Dz_loss'])
        train_op_D = train_op_Dz

        train_op_AE = opt_AE.minimize(loss=loss['AE_loss'], var_list=train_params['AE_loss'])

    # =====================================
    # TF Graph Handler
    asset_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "asset"))
    img_gen_dir = make_dir_if_not_exist(os.path.join(asset_dir, "img_gen"))
    img_rec_dir = make_dir_if_not_exist(os.path.join(asset_dir, "img_rec"))
    img_itpl_dir = make_dir_if_not_exist(os.path.join(asset_dir, "img_itpl"))

    log_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "log"))
    train_log_file = os.path.join(log_dir, "train.log")

    summary_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "summary_tf"))
    model_dir = make_dir_if_not_exist(os.path.join(args.output_dir, "model_tf"))

    train_helper = SimpleTrainHelper(
        log_dir=summary_dir,
        save_dir=model_dir,
        max_to_keep=3,
        max_to_keep_best=1,
    )
    # =====================================

    # =====================================
    # Training Loop
    # =====================================
    config_proto = tf.ConfigProto(allow_soft_placement=True)
    config_proto.gpu_options.allow_growth = True
    config_proto.gpu_options.per_process_gpu_memory_fraction = 0.9
    sess = tf.Session(config=config_proto)
    train_helper.initialize(sess, init_variables=True, create_summary_writer=True)

    Dz_fetch_keys = ['Dz_loss', 'D_loss_z0', 'D_loss_z1_gen',
                     'D_avg_prob_z0', 'D_avg_prob_z1_gen', 'gp0_z']
    D_fetch_keys = Dz_fetch_keys
    AE_fetch_keys = ['AE_loss', 'rec_x', 'G_loss_z1_gen']

    train_sampler = ContinuousIndexSampler(num_train, args.batch_size, shuffle=True)

    import math
    num_batch_per_epochs = int(math.ceil(num_train / args.batch_size))

    global_step = 0
    for epoch in range(args.epochs):
        for _ in range(num_batch_per_epochs):
            global_step += 1

            batch_ids = train_sampler.sample_ids()
            x = celebA_loader.sample_images_from_dataset(sess, 'train', batch_ids)

            z = np.random.randn(len(x), args.z_dim)

            for i in range(args.Dz_steps):
                _, Dm = sess.run([train_op_D, model.get_output(D_fetch_keys, as_dict=True)],
                    feed_dict={model.is_train: True, model.x_ph: x, model.z_ph: z})

            for i in range(args.AE_steps):
                _, AEm = sess.run([train_op_AE, model.get_output(AE_fetch_keys, as_dict=True)],
                    feed_dict={model.is_train: True, model.x_ph: x, model.z_ph: z})

            if global_step % args.save_freq == 0:
                train_helper.save(sess, global_step)

            if global_step % args.log_freq == 0:
                log_str = "\n[AAE/{} (MNIST), {}] Epoch {}/{}, Step {}".format(
                                args.enc_dec_model, args.run, epoch, args.epochs, global_step) + \
                          "\nDz_loss: {:.4f}, AE_loss: {:.4f}".format(
                              Dm['Dz_loss'], AEm['AE_loss']) + \
                          "\nrec_x: {:.4f}".format(AEm['rec_x']) + \
                          "\nD_loss_z0: {:.4f}, D_loss_z1_gen: {:.4f}, G_loss_z1_gen: {:.4f}".format(
                              Dm['D_loss_z0'], Dm['D_loss_z1_gen'], AEm['G_loss_z1_gen']) + \
                          "\nD_avg_prob_z0: {:.4f}, D_avg_prob_z1_gen: {:.4f}".format(
                              Dm['D_avg_prob_z0'], Dm['D_avg_prob_z1_gen']) + \
                          "\ngp0_z_coeff: {:.4f}, gp0_z: {:.4f}".format(args.gp0_z_coeff, Dm['gp0_z'])

                print(log_str)
                with open(train_log_file, "a") as f:
                    f.write(log_str)
                    f.write("\n")
                f.close()

                train_helper.add_summary(custom_tf_scalar_summary(
                    'AE_loss', AEm['AE_loss'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'rec_x', AEm['rec_x'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'G_loss_z1_gen', AEm['G_loss_z1_gen'], prefix='train'), global_step)

                train_helper.add_summary(custom_tf_scalar_summary(
                    'D_loss_z0', Dm['D_loss_z0'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'D_loss_z1_gen', Dm['D_loss_z1_gen'], prefix='train'), global_step)

                train_helper.add_summary(custom_tf_scalar_summary(
                    'D_prob_z0', Dm['D_avg_prob_z0'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'D_prob_z1_gen', Dm['D_avg_prob_z1_gen'], prefix='train'), global_step)

                train_helper.add_summary(custom_tf_scalar_summary(
                    'gp0_z', Dm['gp0_z'], prefix='train'), global_step)

            if global_step % args.viz_gen_freq == 0:
                # Generate images
                # ------------------------- #
                z = np.random.randn(64, args.z_dim)
                img_file = os.path.join(img_gen_dir, 'step[%d]_gen_test.png' % global_step)

                model.generate_images(img_file, sess, z, block_shape=[8, 8],
                                      batch_size=args.batch_size,
                                      dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

            if global_step % args.viz_rec_freq == 0:
                # Reconstruct images
                # ------------------------- #
                x = celebA_loader.sample_images_from_dataset(
                    sess, 'test', np.random.choice(num_test, size=64, replace=False))

                img_file = os.path.join(img_rec_dir, 'step[%d]_rec_test.png' % global_step)

                model.reconstruct_images(img_file, sess, x, block_shape=[8, 8],
                                         batch_size=args.batch_size,
                                         dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

            if global_step % args.viz_itpl_freq == 0:
                # Interpolate images
                # ------------------------- #
                x1 = celebA_loader.sample_images_from_dataset(
                    sess, 'test', np.random.choice(num_test, size=12, replace=False))
                x2 = celebA_loader.sample_images_from_dataset(
                    sess, 'test', np.random.choice(num_test, size=12, replace=False))

                img_file = os.path.join(img_itpl_dir, 'step[%d]_itpl_test.png' % global_step)

                model.interpolate_images(img_file, sess, x1, x2, num_itpl_points=12,
                                         batch_on_row=True, batch_size=args.batch_size,
                                         dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

        if epoch % 100 == 0:
            train_helper.save_separately(sess, model_name="model_epoch{}".format(epoch),
                                         global_step=global_step)

    # Last save
    train_helper.save(sess, global_step)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
