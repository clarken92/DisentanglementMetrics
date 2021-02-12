from six import iteritems
import tensorflow as tf
import pprint
pp = pprint.PrettyPrinter(indent=4)

from my_utils.tensorflow_utils.shaping import mixed_shape, flatten_right_from

from .base import BaseLatentModel


# AAE
# =========================================== #
# Reconstruction loss over x
# Dz loss between z0 and z1_gen

# Dz: Dz0, Dz1_gen, gpz
# AE: rec_x (shared) | Gz1_gen (Enc)
class AAE(BaseLatentModel):
    def __init__(self, x_shape, z_shape, encoder, decoder, discriminator_z,
                 rec_x_mode='l1', stochastic_z=True,
                 use_gp0_z=True, gp0_z_mode="interpolation"):

        super(AAE, self).__init__(x_shape, z_shape)

        print("[AAE] stochastic_z: {}".format(stochastic_z))
        if stochastic_z:
            assert hasattr(encoder, 'stochastic') and encoder.stochastic, \
                "'encoder' must have 'stochastic' attribute set to True!"
        else:
            assert hasattr(encoder, 'stochastic') and (not encoder.stochastic), \
                "'encoder' must have 'stochastic' attribute set to False!"
        self.stochastic_z = stochastic_z

        self.encoder_fn = tf.make_template('encoder', encoder, is_train=self.is_train)
        self.decoder_fn = tf.make_template('decoder', decoder, is_train=self.is_train)
        self.disc_z_fn = tf.make_template('discriminator_z', discriminator_z, is_train=self.is_train)
        self.rec_x_mode = rec_x_mode

        self.use_gp0_z = use_gp0_z
        self.gp0_z_mode = gp0_z_mode

    def build(self, loss_coeff_dict):
        lc = loss_coeff_dict
        coeff_fn = self.one_if_not_exist

        batch_size = mixed_shape(self.x_ph)[0]
        ones = tf.ones([batch_size], dtype=tf.float32)
        zeros = tf.zeros([batch_size], dtype=tf.float32)

        Dz_loss = 0
        AE_loss = 0

        # Encoder / Decoder
        # ================================== #
        x0 = self.x_ph
        z1_gen, z1_gen_dist = self.encoder_fn(x0, return_distribution=True)
        x1 = self.decoder_fn(z1_gen)

        z0 = self.z_ph
        x1_gen = self.decoder_fn(z0)

        self.set_output('z1_gen', z1_gen)

        if self.stochastic_z:
            assert z1_gen_dist is not None, "'z1_gen_dist' must be not None!"
            self.set_output('z_mean', z1_gen_dist['mean'])
            self.set_output('z_stddev', z1_gen_dist['stddev'])
        else:
            assert z1_gen_dist is None, "'z1_gen_dist' must be None!"
            self.set_output('z_mean', tf.zeros_like(z1_gen))
            self.set_output('z_stddev', tf.ones_like(z1_gen))

        self.set_output('x1_gen', x1_gen)
        self.set_output('x1', x1)
        # ================================== #

        # Reconstruct x
        # ================================== #
        print("[AAE] rec_x_mode: {}".format(self.rec_x_mode))
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

        AE_loss += coeff_fn(lc, 'rec_x') * rec_x
        # ================================== #

        # Discriminate z
        # ================================== #
        # E_p(z)[log D(z0)] +  E_p(x)p(z|x)[log(1 - D(z1_gen))]
        D_logit_z0 = self.disc_z_fn(z0)
        D_logit_z1_gen = self.disc_z_fn(z1_gen)

        D_prob_z0 = tf.nn.sigmoid(D_logit_z0)
        D_prob_z1_gen = tf.nn.sigmoid(D_logit_z1_gen)

        D_loss_z0 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_z0, labels=ones), axis=0)
        D_loss_z1_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_z1_gen, labels=zeros), axis=0)
        G_loss_z1_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=D_logit_z1_gen, labels=ones), axis=0)

        self.set_output('D_logit_z0', D_logit_z0)
        self.set_output('D_prob_z0', D_prob_z0)
        self.set_output('D_prob_z1_gen', D_prob_z1_gen)
        self.set_output('D_avg_prob_z0', tf.reduce_mean(D_prob_z0, axis=0))
        self.set_output('D_avg_prob_z1_gen', tf.reduce_mean(D_prob_z1_gen, axis=0))

        self.set_output('D_loss_z0', D_loss_z0)
        self.set_output('D_loss_z1_gen', D_loss_z1_gen)
        self.set_output('G_loss_z1_gen', G_loss_z1_gen)

        Dz_loss += coeff_fn(lc, "D_loss_z1_gen") * (D_loss_z0 + D_loss_z1_gen)
        AE_loss += coeff_fn(lc, "G_loss_z1_gen") * G_loss_z1_gen
        # ================================== #

        # Gradient penalty term for z
        # ================================== #
        if self.use_gp0_z:
            print("[AAE] use_gp0_z: {}".format(self.use_gp0_z))
            print("[AAE] gp0_z_mode: {}".format(self.gp0_z_mode))

            gp0_z = self.gp0("interpolation", self.disc_z_fn, z1_gen, z0)
            self.set_output('gp0_z', gp0_z)

            Dz_loss += coeff_fn(lc, 'gp0_z') * gp0_z
        # ================================== #

        self.set_output('Dz_loss', Dz_loss)
        self.set_output('AE_loss', AE_loss)

    def get_loss(self):
        return {
            'Dz_loss': self.output_dict['Dz_loss'],
            'AE_loss': self.output_dict['AE_loss']
        }

    def get_train_params(self):
        enc_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="encoder")
        dec_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="decoder")
        disc_z_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="discriminator_z")

        train_params = {
            'AE_loss': enc_params + dec_params,
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
            'AE_loss': enc_update_ops + dec_update_ops,
            'Dz_loss': disc_z_update_ops,
        }

        for key, val in iteritems(update_ops):
            print("{}: {}".format(key, val))

        return update_ops
# =========================================== #