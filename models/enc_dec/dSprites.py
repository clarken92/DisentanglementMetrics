import functools
import tensorflow as tf

from my_utils.tensorflow_utils.shaping import mixed_shape, flatten_right_from


# From FactorVAE code by 1Konny
# =================================================== #
class Encoder_1Konny(object):
    def __init__(self, z_dim, stochastic=False, activation=tf.nn.relu):
        self.z_dim = z_dim
        self.stochastic = stochastic
        self.activation = activation

    def __call__(self, x, is_train, return_distribution=False, return_top_hid=False, scope=None, reuse=None):
        activation = self.activation

        weight_init = tf.truncated_normal_initializer(stddev=0.02)
        conv2d = functools.partial(tf.layers.conv2d, kernel_initializer=weight_init)
        dense = functools.partial(tf.layers.dense, kernel_initializer=weight_init)

        with tf.variable_scope(scope or self.__class__.__name__, reuse=reuse):
            x_shape = mixed_shape(x)
            assert len(x_shape) == 4 and x_shape[1] == x_shape[2] == 64

            h = x

            # with tf.variable_scope("block_1"):
            with tf.variable_scope("conv_1"):
                # (32, 32, 32)
                h = conv2d(h, filters=32, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            # with tf.variable_scope("block_2"):
            with tf.variable_scope("conv_2"):
                # (16, 16, 32)
                h = conv2d(h, filters=32, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            # with tf.variable_scope("block_3"):
            with tf.variable_scope("conv_3"):
                # (8, 8, 64)
                h = conv2d(h, filters=64, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            # with tf.variable_scope("block_4"):
            with tf.variable_scope("conv_4"):
                # (4, 4, 64)
                h = conv2d(h, filters=64, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            # with tf.variable_scope("block_5"):
            with tf.variable_scope("block_5"):
                # (4 * 4 * 64,)
                h = flatten_right_from(h, 1)
                h = dense(h, 128, use_bias=True)
                h = activation(h)

            with tf.variable_scope("top"):
                if self.stochastic:
                    mu = dense(h, self.z_dim, use_bias=True)
                    log_sigma = dense(h, self.z_dim, use_bias=True)
                    sigma = tf.exp(log_sigma)

                    eps = tf.random_normal(shape=mixed_shape(mu), mean=0.0, stddev=1.0)
                    z = mu + eps * sigma
                else:
                    z = dense(h, self.z_dim, use_bias=True)
                    mu = log_sigma = sigma = None

                dist = {'mean': mu, 'log_stddev': log_sigma, 'stddev': sigma}

            outputs = [z]
            if return_distribution:
                outputs.append(dist)
            if return_top_hid:
                outputs.append(h)

            return outputs[0] if (len(outputs) == 1) else tuple(outputs)


class Decoder_1Konny(object):
    def __init__(self, activation=tf.nn.relu):
        self.x_dim = (64, 64, 1)
        self.activation = activation

    def __call__(self, z, is_train, return_distribution=False, return_top_hid=False, scope=None, reuse=None):
        activation = self.activation

        weight_init = tf.truncated_normal_initializer(stddev=0.02)
        deconv2d = functools.partial(tf.layers.conv2d_transpose, kernel_initializer=weight_init)
        dense = functools.partial(tf.layers.dense, kernel_initializer=weight_init)

        with tf.variable_scope(scope or self.__class__.__name__, reuse=reuse):
            z_shape = mixed_shape(z)
            if len(z_shape) == 4:
                assert z_shape[1] == z_shape[2] == 1
                z = flatten_right_from(z, axis=1)
            batch_size = z_shape[0]

            # (z_dim,)
            h = z

            with tf.variable_scope("block_1"):
                # (128,)
                h = dense(h, 128, use_bias=True)
                h = activation(h)

            with tf.variable_scope("block_2"):
                h = dense(h, 4 * 4 * 64, use_bias=True)
                h = activation(h)

            with tf.variable_scope("block_3"):
                h = tf.reshape(h, [batch_size, 4, 4, 64])
                h = deconv2d(h, filters=64, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            with tf.variable_scope("block_4"):
                # (16, 16, 32)
                h = deconv2d(h, filters=32, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            with tf.variable_scope("block_5"):
                # (32, 32, 32)
                h = deconv2d(h, filters=32, kernel_size=4, strides=2, padding="same")
                h = activation(h)

            with tf.variable_scope("top"):
                # (64, 64, 1)
                x_logit = deconv2d(h, filters=self.x_dim[-1], kernel_size=4, strides=2, padding="same")
                x = tf.nn.sigmoid(x_logit)

            outputs = [x]
            if return_distribution:
                outputs.append({'prob': x, 'logit': x_logit})
            if return_top_hid:
                outputs.append(h)

            return outputs[0] if (len(outputs) == 1) else tuple(outputs)


class DiscriminatorZ_1Konny(object):
    def __init__(self, num_outputs=1):
        self.num_outputs = num_outputs

    def __call__(self, z, is_train, return_top_hid=False, scope=None, reuse=None):
        weight_init = tf.truncated_normal_initializer(stddev=0.02)
        dense = functools.partial(tf.layers.dense, kernel_initializer=weight_init)

        with tf.variable_scope(scope or self.__class__.__name__, reuse=reuse):
            z_shape = mixed_shape(z)
            if len(z_shape) == 4:
                assert z_shape[1] == z_shape[2] == 1
                z = flatten_right_from(z, 1)

            h = z
            for i in range(5):
                with tf.variable_scope("layer_{}".format(i)):
                    h = dense(h, 1000, use_bias=True)
                    h = tf.nn.leaky_relu(h, 0.2)

            with tf.variable_scope("top"):
                dZ = dense(h, self.num_outputs, use_bias=True)

            if self.num_outputs == 1:
                dZ = tf.reshape(dZ, mixed_shape(dZ)[:-1])

            outputs = [dZ]
            if return_top_hid:
                outputs.append(h)

            return outputs[0] if (len(outputs) == 1) else tuple(outputs)
# =================================================== #

