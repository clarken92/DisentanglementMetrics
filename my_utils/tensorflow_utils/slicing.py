import tensorflow as tf

from .shaping import mixed_shape
from ..python_utils.general import prod


def shuffle_batch_4_each_feature(x, name=None):
    # Used in FactorVAE
    # For each feature, shuffle along batch axis
    # Use 'map_fn' for parallel computation
    with tf.name_scope(name or "shuffle_batch_4_each_feature"):
        x_batch_shape = mixed_shape(x)
        batch_size, x_shape = x_batch_shape[0], x_batch_shape[1:]
        x_dim = prod(x_shape)

        x_flat = tf.reshape(x, [batch_size, x_dim])
        col_ids_dup = tf.tile(tf.expand_dims(tf.range(x_dim), axis=0), [batch_size, 1])
        batch_ids_dup_transposed = tf.tile(tf.expand_dims(tf.range(batch_size), axis=0), [x_dim, 1])

        batch_ids_shuffled = tf.transpose(tf.map_fn(tf.random_shuffle, batch_ids_dup_transposed,
                                          parallel_iterations=100, back_prop=True, swap_memory=True), [1, 0])

        rand_col_ids_nd = tf.stack([batch_ids_shuffled, col_ids_dup], axis=-1)
        rand_col_ids_nd = tf.stop_gradient(rand_col_ids_nd)

        x_shuffled_flat = tf.gather_nd(x_flat, rand_col_ids_nd)
        x_shuffled = tf.reshape(x_shuffled_flat, [batch_size] + x_shape)
        return x_shuffled
