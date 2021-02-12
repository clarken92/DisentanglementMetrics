import math
import numpy as np

from tqdm import tqdm

from utils.general import normal_density, normal_log_density, logsumexp
from utils.general import normal_density_cupy, normal_log_density_cupy, logsumexp_cupy


# Numpy support (Very slow)
# =============================================== #
# Use density function
def estimate_entropies_v1(z_samples, z_mean, z_stddev, num_samples=10000, weights=None,
                          batch_size=10, eps=1e-8):
    """
    Compute H(q(z)) = E_p(x) E_q(z|x) [-log q(z)]

    E_p(x) we use num sample
    To compute log q(z), we use all samples to avoid bias

    :param z_samples: (N, z_dim), sample from q(z|x^(n))
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param num_samples: Only take 'num_samples' from 'z_samples'
    :param weights: A vector of length N
    :param batch_size:
    :param eps:
    :return:
    """
    assert len(z_samples) == len(z_mean) == len(z_stddev), "'z_samples', 'z_mean' and 'z_stddev' must " \
        "have the same length. Found {}, {} and {}, respectively!".format(len(z_samples), len(z_mean), len(z_stddev))
    data_size = len(z_mean)

    if weights is None:
        repeat = num_samples // data_size + 1
        z_samples_temp = []
        for _ in range(repeat):
            rand_ids = np.random.permutation(data_size)
            z_samples_temp.append(z_samples[rand_ids])

        z_samples_temp = np.concatenate(z_samples_temp, axis=0)
        assert len(z_samples_temp) > num_samples, "'len(z_samples_temp)': {}, 'len(num_samples): {}!".\
            format(len(z_samples_temp), num_samples)

        z_samples = z_samples_temp[:num_samples]
    else:
        assert len(weights) == data_size, "'weights' and 'z_samples' must have the same length. " \
            "Found {} and {}, respectively!".format(len(weights), len(z_samples))
        assert np.sum(weights) == 1.0, "'weights' should sum to 1. Found {:.3f}!".format(np.sum(weights))

        rand_ids = np.random.choice(data_size, size=num_samples, replace=True, p=weights)
        z_samples = z_samples[rand_ids]

    # (1, S, z_dim)
    z_samples = np.expand_dims(z_samples, axis=0)
    # (N, 1, z_dim)
    z_mean = np.expand_dims(z_mean, axis=1)
    # (N, 1, z_dim)
    z_stddev = np.expand_dims(z_stddev, axis=1)

    if weights is None:
        weights = [1.0 / data_size for _ in range(data_size)]

    # (N, 1, 1)
    weights = np.expand_dims(np.expand_dims(weights, axis=-1), axis=-1)

    # Entropies of all q_zi
    entropies = np.zeros(z_samples.shape[-1], dtype=np.float32)

    progress_bar = tqdm(total=num_samples)
    count = 0

    # num_samples for the outside expectation
    while count < num_samples:
        b = min(batch_size, num_samples - count)

        # (1, S_batch, z_dim)
        z_samples_batch = z_samples[:, count: count + b, :]

        # p(x^(n)) * q(z|x^(n))
        # (N, S_batch, z_dim)
        qz_samples_batch = normal_density(z_samples_batch, z_mean, z_stddev, eps=eps)
        # Sum (p(x^(n)) * q(z|x^(n)))
        # (S_batch, z_dim)
        qz_samples_batch = np.sum(weights * qz_samples_batch, axis=0)
        # (z_dim,)
        entropies += -np.sum(np.log(np.maximum(qz_samples_batch, eps)), axis=0)

        count += b
        progress_bar.update(b)

    progress_bar.close()

    entropies = entropies / (1.0 * num_samples)
    return entropies


# Use log density function and logsumexp
def estimate_entropies_v2(z_samples, z_mean, z_stddev, num_samples=10000, weights=None,
                          batch_size=10, eps=1e-8):
    """
    Compute H(q(z)) = E_p(x) E_q(z|x) [-log q(z)]

    :param z_samples: (N, z_dim), sample from q(z|x^(n))
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param num_samples: Only take 'num_samples' from 'z_samples'
    :param weights: A vector of length N
    :param batch_size:
    :param eps:
    :return:
    """
    assert len(z_samples) == len(z_mean) == len(z_stddev), "'z_samples', 'z_mean' and 'z_stddev' must " \
        "have the same length. Found {}, {} and {}, respectively!".format(
            len(z_samples), len(z_mean), len(z_stddev))
    data_size = len(z_mean)

    if weights is None:
        repeat = num_samples // data_size + 1
        z_samples_temp = []
        for _ in range(repeat):
            rand_ids = np.random.permutation(data_size)
            z_samples_temp.append(z_samples[rand_ids])

        z_samples_temp = np.concatenate(z_samples_temp, axis=0)
        assert len(z_samples_temp) > num_samples, "'len(z_samples_temp)': {}, 'len(num_samples): {}!". \
            format(len(z_samples_temp), num_samples)

        z_samples = z_samples_temp[:num_samples]
    else:
        assert len(weights) == data_size, "'weights' and 'z_samples' must have the same length. " \
                                          "Found {} and {}, respectively!".format(len(weights), len(z_samples))
        assert np.sum(weights) == 1.0, "'weights' should sum to 1. Found {:.3f}!".format(np.sum(weights))

        rand_ids = np.random.choice(data_size, size=num_samples, replace=True, p=weights)
        z_samples = z_samples[rand_ids]

    # (1, S, z_dim)
    z_samples = np.expand_dims(z_samples, axis=0)
    # (N, 1, z_dim)
    z_mean = np.expand_dims(z_mean, axis=1)
    # (N, S, z_dim)
    z_stddev = np.expand_dims(z_stddev, axis=1)

    if weights is None:
        # (1,)
        weights = -math.log(data_size)
    else:
        # (N, 1, 1)
        weights = np.expand_dims(np.expand_dims(np.log(weights), -1), -1)

    # Entropies of all q_zi
    entropies = np.zeros(z_samples.shape[-1], dtype=np.float32)

    progress_bar = tqdm(total=num_samples)
    count = 0
    while count < num_samples:
        b = min(batch_size, num_samples - count)

        # (1, S_batch, z_dim)
        z_samples_batch = z_samples[:, :, count: count + b]

        # (N, S_batch, z_dim)
        log_qz_samples_batch = normal_log_density(z_samples_batch, z_mean, z_stddev, eps=eps)

        entropies += - np.sum(logsumexp(log_qz_samples_batch + weights, axis=0), axis=0)

        count += b
        progress_bar.update(b)

    progress_bar.close()

    entropies = entropies / (1.0 * num_samples)
    return entropies


def MIG_4_dSprites(z_samples, z_mean, z_stddev, num_samples=10000, batch_size=10, version=1):
    """
    :param z_samples: [3, 6, 40, 32, 32, z_dim]
    :param z_mean: [3, 6, 40, 32, 32, z_dim]
    :param z_stddev: [3, 6, 40, 32, 32, z_dim]
    :param num_samples:
    :param batch_size:
    :param version: 1 or 2
    :return:
    """
    assert version == 1 or version == 2, "'version' can only be 1 or 2!"
    if version == 1:
        estimate_entropies = estimate_entropies_v1
    else:
        estimate_entropies = estimate_entropies_v2

    assert z_samples.shape == z_mean.shape == z_stddev.shape, "z_samples.shape: {}, " \
        "z_mean.shape: {}, z_stddev.shape: {}".format(z_samples.shape, z_mean.shape, z_stddev.shape)
    assert len(z_samples.shape) == 6 and z_samples.shape[:-1] == (3, 6, 40, 32, 32), \
        "z_samples.shape: {}".format(z_samples.shape)

    print("Estimate marginal entropy")
    # H(q(z)) estimated with stratified sampling
    # (z_dim, )
    marginal_entropies = estimate_entropies(
        np.reshape(z_samples, [3 * 6 * 40 * 32 * 32, -1]),
        np.reshape(z_mean, [3 * 6 * 40 * 32 * 32, -1]),
        np.reshape(z_stddev, [3 * 6 * 40 * 32 * 32, -1]),
        batch_size=batch_size)
    # (1, z_dim)
    marginal_entropies = np.expand_dims(marginal_entropies, axis=0)

    # (5, z_dim)
    cond_entropies = np.zeros([5, z_samples.shape[-1]], dtype=np.float32)

    print("Estimate conditional entropy for shape")
    for i in range(3):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            np.reshape(z_mean[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            np.reshape(z_stddev[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[0] += cond_entropies_i / 3.0

    print("Estimate conditional entropy for scale")
    for i in range(6):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            np.reshape(z_mean[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            np.reshape(z_stddev[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[1] += cond_entropies_i / 6.0

    print("Estimate conditional entropy for rotation")
    for i in range(40):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            np.reshape(z_mean[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            np.reshape(z_stddev[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[2] += cond_entropies_i / 40.0

    print("Estimate conditional entropy for pos x")
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_mean[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_stddev[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[3] += cond_entropies_i / 32.0

    print("Estimate conditional entropy for pos y")
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_mean[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_stddev[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[4] += cond_entropies_i / 32.0

    # (5, z_dim)
    MIs = marginal_entropies - cond_entropies
    # (5, z_dim)
    ids_sorted = np.argsort(MIs, axis=1)[:, ::-1]
    MIs_sorted = np.take_along_axis(MIs, ids_sorted, axis=1)

    factor_entropies = np.log([3, 6, 40, 32, 32])

    # Normalize MI by the entropy of factors
    MIs_sorted_normed = MIs_sorted / np.expand_dims(factor_entropies, axis=-1)
    MIG = MIs_sorted_normed[:, 0] - MIs_sorted_normed[:, 1]

    results = {
        'Hz': marginal_entropies,
        'Hy': factor_entropies,
        'Hz_cond_y': cond_entropies,
        'MIs': MIs,
        'MI_sorted': MIs_sorted,
        'MIs_sorted_normed': MIs_sorted_normed,
        'MIG': MIG,
    }

    return results
# =============================================== #


# Cupy support (Very fast)
# =============================================== #
# Use density function
def estimate_entropies_v1_cupy(z_samples, z_mean, z_stddev, weights=None,
                               num_samples=10000, batch_size=10, eps=1e-8, gpu=0):
    """
    Compute H(q(z)) = E_p(x) E_q(z|x) [-log q(z)]

    :param z_samples: (N, z_dim), sample from q(z|x^(n))
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param weights: A vector of length N
    :param num_samples: Only take 'num_samples' from 'z_samples'
    :param batch_size:
    :param eps:
    :param gpu: ID of the gpu
    :return:
    """
    assert len(z_samples) == len(z_mean) == len(z_stddev), "'z_samples', 'z_mean' and 'z_stddev' must " \
        "have the same length. Found {}, {} and {}, respectively!".format(len(z_samples), len(z_mean), len(z_stddev))
    data_size = len(z_mean)

    if weights is None:
        repeat = num_samples // data_size + 1
        z_samples_temp = []
        for _ in range(repeat):
            rand_ids = np.random.permutation(data_size)
            z_samples_temp.append(z_samples[rand_ids])

        z_samples_temp = np.concatenate(z_samples_temp, axis=0)
        assert len(z_samples_temp) > num_samples, "'len(z_samples_temp)': {}, 'len(num_samples): {}!".\
            format(len(z_samples_temp), num_samples)

        z_samples = z_samples_temp[:num_samples]
    else:
        assert len(weights) == data_size, "'weights' and 'z_samples' must have the same length. " \
            "Found {} and {}, respectively!".format(len(weights), len(z_samples))
        assert np.sum(weights) == 1.0, "'weights' should sum to 1. Found {:.3f}!".format(np.sum(weights))

        rand_ids = np.random.choice(data_size, size=num_samples, replace=True, p=weights)
        z_samples = z_samples[rand_ids]

    import cupy as cp
    with cp.cuda.Device(gpu):
        # (1, S, z_dim)
        z_samples = cp.expand_dims(cp.asarray(z_samples), axis=0)
        # (N, 1, z_dim)
        z_mean = cp.expand_dims(cp.asarray(z_mean), axis=1)
        # (N, 1, z_dim)
        z_stddev = cp.expand_dims(cp.asarray(z_stddev), axis=1)

        if weights is None:
            weights = cp.asarray([1.0 / data_size for _ in range(data_size)])
        else:
            weights = cp.asarray(weights)
        # (N, 1, 1)
        weights = cp.expand_dims(cp.expand_dims(weights, axis=-1), axis=-1)

        # Entropies of all q_zi
        entropies = cp.asarray(np.zeros(z_samples.shape[-1], dtype=np.float32))

        progress_bar = tqdm(total=num_samples)
        count = 0
        while count < num_samples:
            b = min(batch_size, num_samples - count)

            # (1, S_batch, z_dim)
            z_samples_batch = z_samples[:, count: count + b, :]

            # p(x^(n)) * q(z|x^(n))
            # (N, S_batch, z_dim)
            qz_samples_batch = normal_density_cupy(z_samples_batch, z_mean, z_stddev, eps=eps, gpu=gpu)
            # Sum (p(x^(n)) * q(z|x^(n)))
            # (S_batch, z_dim)
            qz_samples_batch = cp.sum(weights * qz_samples_batch, axis=0)

            # (z_dim,)
            entropies += -cp.sum(cp.log(cp.maximum(qz_samples_batch, eps)), axis=0)

            count += b
            progress_bar.update(b)

        progress_bar.close()

        # Convert entropies back to numpy
        entropies = cp.asnumpy(entropies)

    entropies = entropies / (1.0 * num_samples)
    return entropies


# Use log density function and logsumexp
def estimate_entropies_v2_cupy(z_samples, z_mean, z_stddev, weights=None,
                               num_samples=10000, batch_size=10, eps=1e-8, gpu=0):
    """
    Compute H(q(z)) = E_p(x) E_q(z|x) [-log q(z)]

    :param z_samples: (N, z_dim), sample from q(z|x^(n))
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param num_samples: Only take 'num_samples' from 'z_samples'
    :param weights: A vector of length N
    :param batch_size:
    :param eps:
    :param gpu: ID of the gpu
    :return:
    """
    assert len(z_samples) == len(z_mean) == len(z_stddev), "'z_samples', 'z_mean' and 'z_stddev' must " \
        "have the same length. Found {}, {} and {}, respectively!".format(
            len(z_samples), len(z_mean), len(z_stddev))
    data_size = len(z_mean)

    if weights is None:
        repeat = num_samples // data_size + 1
        z_samples_temp = []
        for _ in range(repeat):
            rand_ids = np.random.permutation(data_size)
            z_samples_temp.append(z_samples[rand_ids])

        z_samples_temp = np.concatenate(z_samples_temp, axis=0)
        assert len(z_samples_temp) > num_samples, "'len(z_samples_temp)': {}, 'len(num_samples): {}!". \
            format(len(z_samples_temp), num_samples)

        z_samples = z_samples_temp[:num_samples]
    else:
        assert len(weights) == data_size, "'weights' and 'z_samples' must have the same length. " \
                                          "Found {} and {}, respectively!".format(len(weights), len(z_samples))
        assert np.sum(weights) == 1.0, "'weights' should sum to 1. Found {:.3f}!".format(np.sum(weights))

        rand_ids = np.random.choice(data_size, size=num_samples, replace=True, p=weights)
        z_samples = z_samples[rand_ids]

    import cupy as cp
    with cp.cuda.Device(gpu):
        # (1, S, z_dim)
        z_samples = cp.expand_dims(cp.asarray(z_samples), axis=0)
        # (N, 1, z_dim)
        z_mean = cp.expand_dims(cp.asarray(z_mean), axis=1)
        # (N, 1, z_dim)
        z_stddev = cp.expand_dims(cp.asarray(z_stddev), axis=1)

        if weights is None:
            weights = cp.asarray([1.0 / data_size for _ in range(data_size)])
        else:
            weights = cp.asarray(weights)
        # (N, 1, 1)
        weights = cp.expand_dims(cp.expand_dims(weights, axis=-1), axis=-1)

        # Entropies of all q_zi
        entropies = cp.asarray(np.zeros(z_samples.shape[-1], dtype=np.float32))

        progress_bar = tqdm(total=num_samples)
        count = 0
        while count < num_samples:
            b = min(batch_size, num_samples - count)

            # (1, S_batch, z_dim)
            z_samples_batch = z_samples[:, :, count: count + b]

            # (N, S_batch, z_dim)
            log_qz_samples_batch = normal_log_density_cupy(z_samples_batch, z_mean, z_stddev, eps=eps, gpu=gpu)

            entropies += - cp.sum(logsumexp_cupy(log_qz_samples_batch + weights, axis=0, gpu=gpu), axis=0)

            count += b
            progress_bar.update(b)

        progress_bar.close()

    entropies = entropies / (1.0 * num_samples)
    return entropies


def MIG_4_dSprites_cupy(z_samples, z_mean, z_stddev, num_samples=10000,
                        batch_size=10, version=1, gpu=0):
    """
    :param z_samples: [3, 6, 40, 32, 32, z_dim]
    :param z_mean: [3, 6, 40, 32, 32, z_dim]
    :param z_stddev: [3, 6, 40, 32, 32, z_dim]
    :param batch_size:
    :param version: 1 or 2
    :return:
    """

    assert version == 1 or version == 2, "'version' can only be 1 or 2!"
    if version == 1:
        estimate_entropies = estimate_entropies_v1_cupy
    else:
        estimate_entropies = estimate_entropies_v2

    assert z_samples.shape == z_mean.shape == z_stddev.shape, "z_samples.shape: {}, " \
        "z_mean.shape: {}, z_stddev.shape: {}".format(z_samples.shape, z_mean.shape, z_stddev.shape)
    assert len(z_samples.shape) == 6 and z_samples.shape[:-1] == (3, 6, 40, 32, 32), \
        "z_samples.shape: {}".format(z_samples.shape)

    print("Estimate marginal entropy")
    # H(q(z)) estimated with stratified sampling
    # (z_dim, )
    marginal_entropies = estimate_entropies(
        np.reshape(z_samples, [3 * 6 * 40 * 32 * 32, -1]),
        np.reshape(z_mean, [3 * 6 * 40 * 32 * 32, -1]),
        np.reshape(z_stddev, [3 * 6 * 40 * 32 * 32, -1]),
        num_samples=num_samples, batch_size=batch_size, gpu=gpu)
    # (1, z_dim)
    marginal_entropies = np.expand_dims(marginal_entropies, axis=0)

    # (5, z_dim)
    cond_entropies = np.zeros([5, z_samples.shape[-1]], dtype=np.float32)

    print("Estimate conditional entropy for shape")
    for i in range(3):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            np.reshape(z_mean[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            np.reshape(z_stddev[i, :, :, :, :, :], [6 * 40 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size, gpu=gpu)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[0] += cond_entropies_i / 3.0

    print("Estimate conditional entropy for scale")
    for i in range(6):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            np.reshape(z_mean[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            np.reshape(z_stddev[:, i, :, :, :, :], [3 * 40 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size, gpu=gpu)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[1] += cond_entropies_i / 6.0

    print("Estimate conditional entropy for rotation")
    for i in range(40):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            np.reshape(z_mean[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            np.reshape(z_stddev[:, :, i, :, :, :], [3 * 6 * 32 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size, gpu=gpu)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[2] += cond_entropies_i / 40.0

    print("Estimate conditional entropy for pos x")
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_mean[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_stddev[:, :, :, i, :, :], [3 * 6 * 40 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size, gpu=gpu)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[3] += cond_entropies_i / 32.0

    print("Estimate conditional entropy for pos y")
    for i in range(32):
        cond_entropies_i = estimate_entropies(
            np.reshape(z_samples[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_mean[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            np.reshape(z_stddev[:, :, :, :, i, :], [3 * 6 * 40 * 32, -1]),
            num_samples=num_samples, batch_size=batch_size, gpu=gpu)

        # Compute the sum of conditional entropy for each scale value, then take the mean
        cond_entropies[4] += cond_entropies_i / 32.0

    # (5, z_dim)
    MIs = marginal_entropies - cond_entropies
    # (5, z_dim)
    ids_sorted = np.argsort(MIs, axis=1)[:, ::-1]
    MIs_sorted = np.take_along_axis(MIs, ids_sorted, axis=1)

    factor_entropies = np.log([3, 6, 40, 32, 32])

    # Normalize MI by the entropy of factors
    # (5, z_dim)
    MIs_sorted_normed = MIs_sorted / np.expand_dims(factor_entropies, axis=-1)
    # (5,)
    MIG = MIs_sorted_normed[:, 0] - MIs_sorted_normed[:, 1]

    results = {
        'H_z': marginal_entropies,
        'H_y': factor_entropies,
        'H_z_cond_y': cond_entropies,
        'MI': MIs,
        'MI_sorted': MIs_sorted,
        'MI_sorted_normed': MIs_sorted_normed,
        'MIG': MIG,
    }

    return results
# =============================================== #
