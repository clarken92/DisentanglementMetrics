from six import iteritems
from collections import Counter

import math
import numpy as np


def discrete_mutual_info(a, b, eps=1e-8):
    # a: an array of ints, shape (batch,)
    # b: an array of ints, shape (batch,)
    assert isinstance(a, np.ndarray) and ('int' in a.dtype.name) and len(a.shape) == 1, \
        "'a' must be a 1D Numpy array of ints!"
    assert isinstance(b, np.ndarray) and ('int' in b.dtype.name) and len(b.shape) == 1, \
        "'b' must be a 1D Numpy array of ints!"

    assert len(a) == len(b), "'a' and 'b' must have the same length!"

    n = len(a)
    a_counter = Counter(a.tolist())
    b_counter = Counter(b.tolist())
    ab_counter = Counter([(int(ai), int(bi)) for ai, bi in zip(a, b)])

    MI = 0
    for ai_bi, num_ai_bi in iteritems(ab_counter):
        ai, bi = ai_bi

        num_ai = a_counter[ai]
        num_bi = b_counter[bi]

        # MI += (1.0 * num_ai_bi) / (1.0 * n) * math.log((1.0 * n * num_ai_bi) / (1.0 * num_ai * num_bi))
        MI += (1.0 * num_ai_bi) / (1.0 * n) * (math.log(max(1.0 * n * num_ai_bi, eps))
                                               - math.log(max(1.0 * num_ai * num_bi, eps)))

    return MI


def discrete_entropy(a):
    # a: an array of ints, shape (batch,)

    assert isinstance(a, np.ndarray) and ('int' in a.dtype.name) and len(a.shape) == 1, \
        "'a' must be a 1D Numpy array of ints!"

    n = len(a)
    a_counter = Counter(a.tolist())

    ent = 0
    for ai, num_ai in iteritems(a_counter):
        ai_prob = (1.0 * num_ai) / (1.0 * n)

        ent -= ai_prob * math.log(ai_prob)

    return ent


def histogram_discretize(x, num_bins):
    # x: an array of floats, shape (batch,)

    assert isinstance(x, np.ndarray) and ('float' in x.dtype.name) and len(x.shape) == 1, \
        "'x' must be a 1D Numpy array of floats!"

    hist, bin_edges = np.histogram(x, num_bins)
    x_disc = np.digitize(x, bin_edges[:-1])
    return x_disc


def compute_mig(z, y, is_discrete_z, is_discrete_y, num_bins=40, eps=1e-8):
    """
    :param z: (num_latents, num_samples)
    :param y: (num_factors, num_samples)
    :param num_bins:
    :return:
    """

    num_latents = len(z)
    num_factors = len(y)

    if isinstance(is_discrete_z, bool):
        is_discrete_z = [is_discrete_z] * num_latents
    assert isinstance(is_discrete_z, (list, tuple)) and len(is_discrete_z) == num_latents

    if isinstance(is_discrete_y, bool):
        is_discrete_y = [is_discrete_y] * num_factors
    assert isinstance(is_discrete_y, (list, tuple)) and len(is_discrete_y) == num_factors

    MI_z_y = np.zeros([num_latents, num_factors], dtype=np.float32)
    H_y = np.zeros([num_factors], dtype=np.float32)

    for k in range(num_factors):
        yk = y[k]
        if not is_discrete_y[k]:
            yk = histogram_discretize(yk, num_bins)
        H_y[k] = discrete_entropy(yk)

        for i in range(num_latents):
            zi = z[i]
            if not is_discrete_z[i]:
                zi = histogram_discretize(zi, num_bins)

            MI_z_y[i, k] = discrete_mutual_info(zi, yk, eps=eps)
            print("Compute MI for z{} and y{}: {}".format(i, k, MI_z_y[i, k]))

    # Sorted in decreasing order
    MI_ids_sorted = np.argsort(MI_z_y, axis=0)[::-1]
    MI_sorted = np.take_along_axis(MI_z_y, MI_ids_sorted, axis=0)

    MI_gap_y = np.divide(MI_sorted[0, :] - MI_sorted[1, :], H_y)
    MIG = np.mean(MI_gap_y)

    results = {
        'MI_z_y': MI_z_y,
        'H_y': H_y,
        'MI_sorted': MI_sorted,
        'MI_ids_sorted': MI_ids_sorted,
        'MI_gap_y': MI_gap_y,
        'MIG': MIG,
    }

    return results
