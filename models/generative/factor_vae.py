from six import iteritems
import tensorflow as tf

import pprint
pp = pprint.PrettyPrinter(indent=4)

from my_utils.tensorflow_utils.shaping import flatten_right_from
from my_utils.tensorflow_utils.slicing import shuffle_batch_4_each_feature
from my_utils.tensorflow_utils.distributions import KLD_DiagN_N01

from .base import BaseLatentModel


# Work very well
# IMPORTANT: Like 2 functions 'get_loss' and 'get_train_params',
# 'get_update_ops' also separate updates based on loss instead of
# including every update
class FactorVAE(BaseLatentModel):
    def __init__(self, x_shape, z_shape, encoder, decoder,
                 discriminator_z, rec_x_mode="mse",
                 use_gp0_z_tc=False, gp0_z_tc_mode="interpolation"):
        super(FactorVAE, self).__init__(x_shape, z_shape)

        assert hasattr(encoder, 'stochastic') and encoder.stochastic, \
            "'encoder' must have 'stochastic' attribute set to True!"
        self.encoder_fn = tf.make_template('encoder', encoder, is_train=self.is_train)

        self.decoder_fn = tf.make_template('decoder', decoder, is_train=self.is_train)

        assert hasattr(discriminator_z, 'num_outputs') and discriminator_z.num_outputs == 2, \
            "'discriminator_z' must have 'num_outputs' attribute set to 2!"
        self.disc_z_fn = tf.make_template('discriminator_z', discriminator_z, is_train=self.is_train)

        self.rec_x_mode = rec_x_mode

        self.use_gp0_z_tc = use_gp0_z_tc
        self.gp0_z_tc_mode = gp0_z_tc_mode

        self.xa_ph = tf.placeholder(tf.float32, [None] + self.x_shape, name="xa")

    def build(self, loss_coeff_dict):
        lc = loss_coeff_dict
        coeff_fn = self.one_if_not_exist

        vae_loss = 0
        Dz_loss = 0

        # Encoder / Decoder
        # ================================== #
        x0 = self.x_ph
        z1_gen, z1_gen_dist = self.encoder_fn(x0, return_distribution=True)
        x1 = self.decoder_fn(z1_gen)

        z0 = self.z_ph
        x1_gen = self.decoder_fn(z0)

        self.set_output('z1_gen', z1_gen)
        self.set_output('z_mean', z1_gen_dist['mean'])
        self.set_output('z_stddev', z1_gen_dist['stddev'])

        self.set_output('x1_gen', x1_gen)
        self.set_output('x1', x1)
        # ================================== #

        # Reconstruction loss
        # ================================== #
        print("[FactorVAE] rec_x_mode: {}".format(self.rec_x_mode))
        if self.rec_x_mode == 'mse':
            rec_x = tf.reduce_sum(tf.square(flatten_right_from(x0, 1) -
                                            flatten_right_from(x1, 1)), axis=1)
        elif self.rec_x_mode == 'l1':
            rec_x = tf.reduce_sum(tf.abs(flatten_right_from(x0, 1) -
                                         flatten_right_from(x1, 1)), axis=1)
        elif self.rec_x_mode == 'bce':
            _, x1_dist = self.decoder_fn(z1_gen, return_distribution=True)
            rec_x = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
                labels=flatten_right_from(x0, 1),
                logits=flatten_right_from(x1_dist['logit'], 1)), axis=1)
        else:
            raise ValueError("Do not support '{}'!".format(self.rec_x_mode))

        rec_x = tf.reduce_mean(rec_x, axis=0)

        assert rec_x.shape.ndims == 0, "rec_x.shape: {}".format(rec_x.shape)
        self.set_output('rec_x', rec_x)

        vae_loss += coeff_fn(lc, 'rec_x') * rec_x
        # ================================== #

        # KL divergence loss
        # ================================== #
        kld_loss = KLD_DiagN_N01(z1_gen_dist['mean'], z1_gen_dist['log_stddev'], from_axis=1)
        kld_loss = tf.reduce_mean(kld_loss, axis=0)

        assert kld_loss.shape.ndims == 0, "kld_loss.shape: {}".format(kld_loss.shape)
        self.set_output('kld_loss', kld_loss)

        vae_loss += coeff_fn(lc, 'kld_loss') * kld_loss
        # ================================== #

        # Discriminator loss to estimate total correlation
        # ================================== #
        xa = self.xa_ph
        z1a_gen = self.encoder_fn(xa, return_distribution=False)
        z1a_gen_perm = tf.stop_gradient(shuffle_batch_4_each_feature(z1a_gen))

        D_logit2_z1_gen = self.disc_z_fn(z1_gen)
        D_logit2_z1a_gen_perm = self.disc_z_fn(z1a_gen_perm)

        # Probability of 'being as usual' and probability of 'not being as usual'
        D_prob2_z1_gen = tf.nn.softmax(D_logit2_z1_gen)
        D_prob2_z1a_gen_perm = tf.nn.softmax(D_logit2_z1a_gen_perm)

        D_loss_z1_gen_tc = -tf.reduce_mean(tf.nn.log_softmax(D_logit2_z1_gen)[:, 1], axis=0)
        D_loss_z1a_gen_perm = -tf.reduce_mean(tf.nn.log_softmax(D_logit2_z1a_gen_perm)[:, 0], axis=0)

        self.set_output('Dz_prob2_normal', D_prob2_z1_gen)
        self.set_output('Dz_prob2_factor', D_prob2_z1a_gen_perm)
        self.set_output('Dz_avg_prob_normal', tf.reduce_mean(D_prob2_z1_gen[:, 0], axis=0))
        self.set_output('Dz_avg_prob_factor', tf.reduce_mean(D_prob2_z1a_gen_perm[:, 0], axis=0))
        self.set_output('Dz_loss_normal', D_loss_z1_gen_tc)
        self.set_output('Dz_loss_factor', D_loss_z1a_gen_perm)

        Dz_tc_loss = 0.5 * (D_loss_z1_gen_tc + D_loss_z1a_gen_perm)
        assert Dz_tc_loss.shape.ndims == 0, "Dz_tc_loss.shape: {}".format(Dz_tc_loss.shape)
        self.set_output('Dz_tc_loss', Dz_tc_loss)

        tc_loss = -tf.reduce_mean(D_logit2_z1_gen[:, 0] - D_logit2_z1_gen[:, 1], axis=0)
        assert tc_loss.shape.ndims == 0, "tc_loss.shape: {}".format(tc_loss.shape)
        self.set_output('tc_loss', tc_loss)

        Dz_loss += coeff_fn(lc, 'Dz_tc_loss') * Dz_tc_loss
        vae_loss += coeff_fn(lc, 'tc_loss') * tc_loss
        # ================================== #

        # Gradient penalty term for z
        # ================================== #
        if self.use_gp0_z_tc:
            print("[FactorVAE] use_gp0_z_tc: {}".format(self.use_gp0_z_tc))
            print("[FactorVAE] gp0_z_tc_mode: {}".format(self.gp0_z_tc_mode))

            gp0_z_tc = self.gp0(self.gp0_z_tc_mode, self.disc_z_fn,
                                z1_gen, z1a_gen_perm, at_D_comp=0)

            self.set_output('gp0_z_tc', gp0_z_tc)

            Dz_loss += coeff_fn(lc, 'gp0_z_tc') * gp0_z_tc
        # ================================== #

        print("All loss coefficients:")
        pp.pprint(self.loss_coeff_dict)

        self.set_output('Dz_loss', Dz_loss)
        self.set_output('vae_loss', vae_loss)

    def get_loss(self):
        return {
            'vae_loss': self.output_dict['vae_loss'],
            'Dz_loss': self.output_dict['Dz_loss'],
        }

    def get_train_params(self):
        enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        disc_z_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_z")

        train_params = {
            'vae_loss': enc_params + dec_params,
            'Dz_loss': disc_z_params,
        }

        for key, val in iteritems(train_params):
            assert len(val) > 0, "Loss '{}' has params {}".format(key, val)
            print("{}: {}".format(key, val))

        return train_params

    def get_update_ops(self):
        enc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="encoder")
        dec_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="decoder")
        disc_z_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope="discriminator_z")

        update_ops = {
            'vae_loss': enc_update_ops + dec_update_ops,
            'Dz_loss': disc_z_update_ops,
        }

        for key, val in iteritems(update_ops):
            print("{}: {}".format(key, val))

        return update_ops

    def encode(self, sess, x, deterministic=True):
        if deterministic:
            z = self.get_output('z_mean')
        else:
            z = self.get_output('z1_gen')

        return sess.run(z, feed_dict={self.is_train: False, self.x_ph: x})