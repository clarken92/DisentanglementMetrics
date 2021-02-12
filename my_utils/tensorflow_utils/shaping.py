from functools import reduce
from operator import mul
import tensorflow as tf
import numpy as np


def ndims(x):
    return x.get_shape().ndims


def mixed_shape(x):
    # Get the shape of x as a list
    shape = x.get_shape().as_list()
    tf_shape = tf.shape(x)
    return [shape[i] or tf_shape[i] for i in range(len(shape))]


def reshape_4_batch(x, new_shape, num_batch_axes=1):
    # A reshape function that takes into account of batch axes

    # new_shape: New shape of x that does not count batch axes
    # num_batch_axes: The number of left most dimensions of x considered to be batch axes

    if isinstance(new_shape, int):
        new_shape = [new_shape]
    elif isinstance(new_shape, (tuple, list)):
        new_shape = list(new_shape)
    elif isinstance(new_shape, np.ndarray):
        new_shape = new_shape.tolist()
    elif isinstance(new_shape, tf.TensorShape):
        new_shape = new_shape.as_list()
    else:
        raise ValueError("Do not support 'new_shape' with type '{}'!".format(type(new_shape)))

    x_shape = mixed_shape(x)
    new_shape_full = x_shape[:num_batch_axes] + new_shape
    return tf.reshape(x, new_shape_full)


def to_standard_shape(shape):
    if isinstance(shape, tf.TensorShape):
        shape = shape.as_list()
    elif isinstance(shape, int):
        shape = [shape]
    elif isinstance(shape, np.ndarray):
        assert shape.dtype == np.int32, "If 'x' is a Numpy array, it must have dtype=int32!"
        shape = shape.tolist()
    elif isinstance(shape, tuple):
        shape = list(shape)
    elif isinstance(shape, list):
        shape = shape
    else:
        raise ValueError("Do not support shape input of type {}!".format(type(shape)))
    return shape


def flatten(x):
    shape = mixed_shape(x)
    return tf.reshape(x, [reduce(mul, shape)])


def expand_last_dims(x, n, name=None):
    with tf.name_scope(name or "expand_last_dims"):
        x_shape = mixed_shape(x)
        new_x_shape = x_shape + [1] * n
        return tf.reshape(x, new_x_shape)


def flatten_left_to(x, axis, name=None):
    """
    Flat the tensor x to the left until we reach the axis (included)
    :param x:
    :param axis:
    :param name:
    :return: A flattened tensor
    """
    with tf.name_scope(name or "flatten_left_to"):
        shape = mixed_shape(x)
        assert -len(shape) < axis < len(shape), "'axis' must be between ({}, {})!".format(-len(shape), len(shape))
        if axis < 0:
            axis = len(shape) + axis
        flatten_dim = reduce(mul, shape[:axis+1])
        remaining_dims = shape[axis+1:]
        new_shape = [flatten_dim] + remaining_dims
        return tf.reshape(x, new_shape)


def flatten_right_from(x, axis, name=None):
    with tf.name_scope(name or "flatten_left_to"):
        shape = mixed_shape(x)
        assert -len(shape) < axis < len(shape), "'axis' must be between ({}, {}). " \
            "Found {}!".format(-len(shape), len(shape), axis)
        if axis < 0:
            axis = len(shape) + axis
        remaining_dims = shape[:axis]
        flatten_dim = reduce(mul, shape[axis:])
        new_shape = remaining_dims + [flatten_dim]
        return tf.reshape(x, new_shape)


def reconstruct_left(x, ref, num_kept, name=None):
    """
    Reconstruct the shape of the tensor 'x' to be the same as the tensor 'ref' with
    the remaining 'num_kept' axes after flattening
    :param x: A tensor whose shape to be reconstructed
    :param ref: A reference tensor
    :param num_kept: Number of remaining axes
    :param name: Name of this operator
    :return:
    """
    shape = x.get_shape().as_list()
    tf_shape = tf.shape(x)

    ref_shape = ref.get_shape().as_list()
    tf_ref_shape = tf.shape(ref)

    assert num_kept < len(shape) and num_kept < len(ref_shape)
    ax = len(shape) - num_kept
    ref_ax = len(ref_shape) - num_kept

    left_shape = [ref_shape[i] or tf_ref_shape[i] for i in range(ref_ax)]
    kept_shape = [shape[i] or tf_shape[i] for i in range(ax, len(shape))]

    out_shape = left_shape + kept_shape
    out = tf.reshape(x, out_shape, name=(name or "reconstruct_left"))
    return out


def repeat_1d(x, n, name=None):
    """
    Repeat an 1D Tensor x
    :param x: A 1D Tensor
    :param n: Number of repeats
    :return:
    """
    # Repeat (1, 2, 3, 4) with 3 => (1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4)
    with tf.name_scope(name or "repeat_1d"):
        x_shape = mixed_shape(x)
        assert len(x_shape) == 1

        out = tf.expand_dims(x, axis=-1)  # (d, 1)
        out = tf.tile(out, [1, n])  # (d, n)
        out = tf.reshape(out, [n * x_shape[0]])

        return out


def repeat_last(x, n, name=None):
    """
    Repeat the last axis of a tensor x n times
    :param x: A tensor of shape (b1, b2..., bk, d)
    :param n: Number of repeats
    :return:
    """
    with tf.name_scope(name or "repeat_last"):
        x = tf.convert_to_tensor(x)
        x_shape = mixed_shape(x)

        out = tf.expand_dims(x, axis=-1)  # (b1, b2,..., bk, d, 1)
        out = tf.tile(out, [1 for _ in range(len(x_shape))] + [n])  # (d, n)
        out = tf.reshape(out, x_shape[:-1] + [n * x_shape[-1]])

        return out