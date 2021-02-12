import argparse
import os

import pprint
pp = pprint.PrettyPrinter(indent=4)

import numpy as np
import tensorflow as tf

from models.enc_dec.dSprites import Encoder_1Konny, Decoder_1Konny, DiscriminatorZ_1Konny
from models.generative.factor_vae import FactorVAE

from my_utils.python_utils.image import binary_float_to_uint8
from my_utils.python_utils.training import iterate_data
from my_utils.tensorflow_utils.training.helper import SimpleTrainHelper, SimpleParamPrinter
from my_utils.tensorflow_utils.training.summary import custom_tf_scalar_summary

from my_utils.python_utils.general import make_dir_if_not_exist
from my_utils.python_utils.arg_parsing import save_args, str2bool

from global_settings import RAW_DATA_DIR

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--z_dim', type=int, default=10)

parser.add_argument('--lr_vae', type=float, default=1e-4)
parser.add_argument('--beta1_vae', type=float, default=0.9)
parser.add_argument('--beta2_vae', type=float, default=0.99)

parser.add_argument('--lr_Dz', type=float, default=1e-4)
parser.add_argument('--beta1_Dz', type=float, default=0.5)
parser.add_argument('--beta2_Dz', type=float, default=0.99)

parser.add_argument('--rec_x_mode', type=str, default='bce')
parser.add_argument('--rec_x_coeff', type=float, default=1)
parser.add_argument('--kld_loss_coeff', type=float, default=1)
parser.add_argument('--tc_loss_coeff', type=float, default=10)
parser.add_argument('--Dz_tc_loss_coeff', type=float, default=1)

parser.add_argument('--gp0_z_tc_mode', type=str, default="interpolation")
parser.add_argument('--gp0_z_tc_coeff', type=float, default=0.0)

parser.add_argument('--D_steps', type=int, default=1)
parser.add_argument('--vae_steps', type=int, default=1)

parser.add_argument('--log_freq', type=int, default=1000)
parser.add_argument('--save_freq', type=int, default=5000)

parser.add_argument('--viz_gen_freq', type=int, default=10000)
parser.add_argument('--viz_rec_freq', type=int, default=10000)
parser.add_argument('--viz_itpl_freq', type=int, default=10000)

parser.add_argument('--output_dir', type=str, required=True)
parser.add_argument('--run', type=str, required=True)
parser.add_argument('--enc_dec_model', type=str, required=True)
parser.add_argument('--force_rm_dir', type=str2bool, default="False")


def main(args):
    np.set_printoptions(threshold=np.nan, linewidth=1000, precision=3)

    # =====================================
    # Preparation
    # =====================================
    data_file = os.path.join(RAW_DATA_DIR, "ComputerVision", "dSprites",
                             "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz")

    # It is already in the range [0, 1]
    with np.load(data_file, encoding="latin1") as f:
        x_train = f['imgs']

    x_train = np.expand_dims(x_train.astype(np.float32), axis=-1)
    num_train = len(x_train)
    print("x_train: {}".format(num_train))

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
    SimpleParamPrinter.print_all_params_list()
    # SimpleParamPrinter.print_all_params_tf_slim()

    loss = model.get_loss()
    train_params = model.get_train_params()

    opt_Dz = tf.train.AdamOptimizer(learning_rate=args.lr_Dz, beta1=args.beta1_Dz, beta2=args.beta2_Dz)
    opt_vae = tf.train.AdamOptimizer(learning_rate=args.lr_vae, beta1=args.beta1_vae, beta2=args.beta2_vae)

    with tf.control_dependencies(model.get_all_update_ops()):
        train_op_Dz = opt_Dz.minimize(loss=loss['Dz_loss'], var_list=train_params['Dz_loss'])
        train_op_D = train_op_Dz

        train_op_vae = opt_vae.minimize(loss=loss['vae_loss'], var_list=train_params['vae_loss'])

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

    Dz_fetch_keys = ['Dz_loss', 'Dz_tc_loss', 'Dz_loss_normal', 'Dz_loss_factor',
                     'Dz_avg_prob_normal', 'Dz_avg_prob_factor', 'gp0_z_tc']
    D_fetch_keys = Dz_fetch_keys

    vae_fetch_keys = ['vae_loss', 'rec_x', 'kld_loss', 'tc_loss']

    global_step = 0
    for epoch in range(args.epochs):
        for batch_ids in iterate_data(num_train, args.batch_size, shuffle=True):
            global_step += 1

            x = x_train[batch_ids]
            z = np.random.normal(size=[len(x), args.z_dim])

            batch_ids_2 = np.random.choice(num_train, size=len(batch_ids)).tolist()
            xa = x_train[batch_ids_2]

            for i in range(args.D_steps):
                _, Dm = sess.run([train_op_D, model.get_output(D_fetch_keys, as_dict=True)],
                    feed_dict={model.is_train: True, model.x_ph: x, model.z_ph: z, model.xa_ph: xa})

            for i in range(args.vae_steps):
                _, VAEm = sess.run([train_op_vae, model.get_output(vae_fetch_keys, as_dict=True)],
                    feed_dict={model.is_train: True, model.x_ph: x, model.z_ph: z, model.xa_ph: xa})

            if global_step % args.save_freq == 0:
                train_helper.save(sess, global_step)

            if global_step % args.log_freq == 0:
                log_str = "\n[FactorVAE/{}/{} (dSprites)], Epoch[{}/{}], Step {}".format(
                                args.enc_dec_model, args.run, epoch, args.epochs, global_step) + \
                          "\nvae_loss: {:.4f}, Dz_loss: {:.4f}, Dz_tc_loss: {:.4f}".format(
                              VAEm['vae_loss'], Dm['Dz_loss'], Dm['Dz_tc_loss']) + \
                          "\nrec_x: {:.4f}, kld_loss: {:.4f}, tc_loss: {:.4f}".format(
                              VAEm['rec_x'], VAEm['kld_loss'], VAEm['tc_loss']) + \
                          "\nDz_loss_normal: {:.4f}, Dz_loss_factor: {:.4f}".format(
                              Dm['Dz_loss_normal'], Dm['Dz_loss_factor']) + \
                          "\nDz_avg_prob_normal: {:.4f}, Dz_avg_prob_factor: {:.4f}".format(
                              Dm['Dz_avg_prob_normal'], Dm['Dz_avg_prob_factor']) + \
                          "\ngp0_z_tc_coeff: {:.4f}, gp0_z_tc: {:.4f}".format(args.gp0_z_tc_coeff, Dm['gp0_z_tc'])

                print(log_str)
                with open(train_log_file, "a") as f:
                    f.write(log_str)
                    f.write("\n")
                f.close()

                train_helper.add_summary(custom_tf_scalar_summary(
                    'vae_loss', VAEm['vae_loss'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'rec_x', VAEm['rec_x'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'kld_loss', VAEm['kld_loss'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'tc_loss', VAEm['tc_loss'], prefix='train'), global_step)

                train_helper.add_summary(custom_tf_scalar_summary(
                    'Dz_tc_loss', Dm['Dz_tc_loss'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'Dz_loss_normal', Dm['Dz_loss_normal'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'Dz_loss_factor', Dm['Dz_loss_factor'], prefix='train'), global_step)

                train_helper.add_summary(custom_tf_scalar_summary(
                    'Dz_prob_normal', Dm['Dz_avg_prob_normal'], prefix='train'), global_step)
                train_helper.add_summary(custom_tf_scalar_summary(
                    'Dz_prob_factor', Dm['Dz_avg_prob_factor'], prefix='train'), global_step)

            if global_step % args.viz_gen_freq == 0:
                # Generate images
                # ------------------------- #
                z = np.random.normal(size=[64, args.z_dim])
                img_file = os.path.join(img_gen_dir, 'step[%d]_gen_test.png' % global_step)

                model.generate_images(img_file, sess, z, block_shape=[8, 8],
                                      batch_size=args.batch_size,
                                      dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

            if global_step % args.viz_rec_freq == 0:
                # Reconstruct images
                # ------------------------- #
                x = x_train[np.random.choice(num_train, size=64, replace=False)]
                img_file = os.path.join(img_rec_dir, 'step[%d]_rec_test.png' % global_step)

                model.reconstruct_images(img_file, sess, x, block_shape=[8, 8],
                                         batch_size=args.batch_size,
                                         dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

            if global_step % args.viz_itpl_freq == 0:
                # Interpolate images
                # ------------------------- #
                x1 = x_train[np.random.choice(num_train, size=12, replace=False)]
                x2 = x_train[np.random.choice(num_train, size=12, replace=False)]

                img_file = os.path.join(img_itpl_dir, 'step[%d]_itpl_test.png' % global_step)

                model.interpolate_images(img_file, sess, x1, x2, num_itpl_points=12,
                                         batch_on_row=True, batch_size=args.batch_size,
                                         dec_output_2_img_func=binary_float_to_uint8)
                # ------------------------- #

    # Last save
    train_helper.save(sess, global_step)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
