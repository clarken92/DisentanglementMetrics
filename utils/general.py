import math
import numpy as np


def sample_mask_with_rate(shape, one_rate):
    assert len(shape) == 2, "'shape' must be a list/tuple of format (batch, dim)!"
    assert 0.0 < one_rate < 1.0, "'one_rate' must be in (0, 1)"

    num_ones = int(round(one_rate * shape[1], 0))
    num_zeros = shape[1] - num_ones

    if num_ones == 0:
        return np.zeros(shape, dtype=np.bool)
    elif num_zeros == 0:
        return np.ones(shape, dtype=np.bool)
    else:
        mask = np.concatenate([np.ones([shape[0], num_ones], dtype=np.bool),
                               np.zeros([shape[0], num_zeros], dtype=np.bool)], axis=1)
        mask = np.apply_along_axis(np.random.permutation, axis=1, arr=mask)
        return mask


def normal_density(x, mean, stddev, from_axis=None, eps=1e-8):
    variance = np.maximum(stddev ** 2, eps)
    stddev = np.maximum(stddev, eps)

    density = np.exp(-np.square(x - mean) / (2 * variance)) / (stddev * math.sqrt(2 * math.pi))

    if (from_axis is not None) and (from_axis >= 0):
        shape = tuple(density.shape[:from_axis]) + (np.prod(density.shape[from_axis:]),)
        density = np.reshape(density, shape)
        density = np.prod(density, axis=from_axis)

    return density


def normal_density_cupy(x, mean, stddev, from_axis=None, eps=1e-8, gpu=0):
    import cupy as cp

    with cp.cuda.Device(gpu):
        variance = cp.maximum(stddev ** 2, eps)
        stddev = cp.maximum(stddev, eps)

        density = cp.exp(-cp.square(x - mean) / (2 * variance)) / (stddev * math.sqrt(2 * math.pi))

        if (from_axis is not None) and (from_axis >= 0):
            shape = tuple(density.shape[:from_axis]) + (cp.prod(density.shape[from_axis:]),)
            density = cp.reshape(density, shape)
            density = cp.prod(density, axis=from_axis)

        return density


def normal_log_density(x, mean, stddev, from_axis=None, eps=1e-8):
    variance = np.maximum(stddev ** 2, eps)
    log_stddev = np.log(np.maximum(stddev, eps))

    log_density = -0.5 * (math.log(2 * math.pi) + 2 * log_stddev + ((x - mean)**2 / variance))

    if (from_axis is not None) and (from_axis >= 0):
        shape = tuple(log_density.shape[:from_axis]) + (np.prod(log_density.shape[from_axis:]),)
        log_density = np.reshape(log_density, shape)
        log_density = np.sum(log_density, axis=from_axis)

    return log_density


def normal_log_density_cupy(x, mean, stddev, from_axis=None, eps=1e-8, gpu=0):
    import cupy as cp

    with cp.cuda.Device(gpu):
        variance = cp.maximum(stddev ** 2, eps)
        log_stddev = cp.log(cp.maximum(stddev, eps))

        log_density = -0.5 * (math.log(2 * math.pi) + 2 * log_stddev + ((x - mean)**2 / variance))

        if (from_axis is not None) and (from_axis >= 0):
            shape = tuple(log_density.shape[:from_axis]) + (cp.prod(log_density.shape[from_axis:]),)
            log_density = cp.reshape(log_density, shape)
            log_density = cp.sum(log_density, axis=from_axis)

        return log_density


def at_bin(x, bin_edges, one_hot=True):
    # x: (batch, )
    # bin_edges: num_bins + 1

    x = np.asarray(x)
    bin_edges = np.asarray(bin_edges)

    assert len(x.shape) == 1 and len(bin_edges.shape) == 1, \
        "x.shape: {}, bin_edges.shape: {}".format(x.shape, bin_edges.shape)

    low_edges = bin_edges[:-1]  # num_bins
    high_edges = bin_edges[1:]  # num_bins

    # (batch, num_bins)
    x_at_bin = np.logical_and(np.greater_equal(np.expand_dims(x, axis=1), np.expand_dims(low_edges, axis=0)),
                              np.less(np.expand_dims(x, axis=1), np.expand_dims(high_edges, axis=0)))

    min_edge = bin_edges[0]
    max_edge = bin_edges[-1]

    for n in range(len(x_at_bin)):
        if x[n] < min_edge:
            assert np.sum(x_at_bin[n].astype(np.int32)) == 0
            x_at_bin[n, 0] = True

        if x[n] >= max_edge:
            assert np.sum(x_at_bin[n].astype(np.int32)) == 0, x_at_bin[n]
            x_at_bin[n, -1] = True

    if not one_hot:
        x_at_bin = np.argmax(x_at_bin, axis=-1).astype(np.int32)

    # (batch, num_bins)
    return x_at_bin


def logsumexp(value, axis=None, keepdims=False, gpu_support=None):
    if gpu_support == "jax":
        print("Use jax to compute 'normal_log_density'!")
        import jax.numpy as np
    elif gpu_support == "cupy":
        print("Use cupy to compute 'normal_log_density'!")
        import cupy as np
    else:
        import numpy as np

    if axis is not None:
        max_val = np.max(value, axis=axis, keepdims=True)
        value0 = value - max_val
        if not keepdims:
            max_val = np.squeeze(max_val, axis)
        return max_val + np.log(np.sum(np.exp(value0), axis=axis, keepdims=keepdims))
    else:
        max_val = np.max(value)
        sum_exp = np.sum(np.exp(value - max_val))
        return max_val + math.log(sum_exp)


def logsumexp_cupy(value, axis=None, keepdims=False, gpu=0):
    import cupy as cp
    with cp.cuda.Device(gpu):
        if axis is not None:
            max_val = cp.max(value, axis=axis, keepdims=True)
            value0 = value - max_val
            if not keepdims:
                max_val = cp.squeeze(max_val, axis)
            return max_val + cp.log(cp.sum(cp.exp(value0), axis=axis, keepdims=keepdims))
        else:
            max_val = cp.max(value)
            sum_exp = cp.sum(cp.exp(value - max_val))
            return max_val + math.log(sum_exp)


def variance_uniform(low, high):
    # Variance of uniform distribution in the range [low, high]
    return (high - low) ** 2 / 12.0


def variance_categorical(prob):
    # Variance of categorical distribution with K classes, each has probability p1, p2,..., pK
    # prob: (batch, K)
    # var: (batch, K)

    # assert np.sum(prob) == 1.0, "'prob' must sum to 1. sum(prob) = {}!".format(np.sum(prob))
    # If prob = [0.6, 0.1, 0.3], variance (for each class) is = [0.6 * 0.4, 0.1 * 0.9, 0.3 * 0.7]
    return prob * (1.0 - prob)