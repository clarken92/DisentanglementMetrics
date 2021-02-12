import numpy as np
import cupy as cp
from tqdm import tqdm

from utils.general import normal_density_cupy


def robust_prod_cupy(x, axis, eps=1e-8):
    return cp.exp(cp.sum(cp.log(cp.maximum(x, eps)), axis=axis))


def estimate_SEP_cupy(z_mean, z_stddev, x_weights=None,
                      num_samples=10000, seed=None, batch_size=10, eps=1e-8, gpu=0):
    """
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param x_weights: A vector of length N
    :param num_samples: Number of samples
    :param seed: Numpy random state used for sampling
    :param batch_size: Batch size during computation
    :param eps: Batch size during computation
    :param gpu: ID of the gpu
    :return:
    """
    print("num_samples: {}".format(num_samples))
    print("batch_size: {}".format(batch_size))

    assert z_mean.shape == z_stddev.shape and len(z_mean.shape) == 2, \
        "z_mean.shape={} and z_stddev={}".format(z_mean.shape, z_stddev.shape)

    num_x, z_dim = z_mean.shape[0], z_mean.shape[1]
    print("num_x: {}".format(num_x))
    print("z_dim: {}".format(z_dim))

    if x_weights is None:
        x_weights = np.full([num_x], fill_value=1.0 / num_x, dtype=np.float32)

    assert 1.0 - 1e-5 <= np.sum(x_weights) <= 1.0 + 1e-5, "'weights' should sum to 1. Found {:.5f}!".format(np.sum(x_weights))

    rs = np.random.RandomState(seed)
    rand_ids = rs.choice(num_x, size=num_samples, replace=True, p=x_weights)

    # (S, z_dim)
    noise = rs.randn(num_samples, z_dim)
    z_samples = z_mean[rand_ids] + noise * z_stddev[rand_ids]

    import cupy as cp
    with cp.cuda.Device(gpu):
        # (S, z_dim)
        z_samples = cp.asarray(z_samples)
        # (N, 1, z_dim)
        z_mean = cp.expand_dims(cp.asarray(z_mean), axis=1)
        # (N, 1, z_dim)
        z_stddev = cp.expand_dims(cp.asarray(z_stddev), axis=1)

        print("z_samples.shape: {}".format(z_samples.shape))
        print("z_mean.shape: {}".format(z_mean.shape))
        print("z_stddev.shape: {}".format(z_stddev.shape))

        x_weights = cp.asarray(x_weights)
        print("x_weights.shape: {}".format(x_weights.shape))

        # (z_dim,)
        H_zi_cond_x = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_zi = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_z_not_i = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim, z_dim)
        H_zi_zj = cp.asarray(np.zeros([z_dim, z_dim], dtype=np.float32))
        # scalar
        H_z = cp.asarray(np.zeros(1, dtype=np.float32))

        progress_bar = tqdm(total=num_samples)
        count = 0
        while count < num_samples:
            b = min(batch_size, num_samples - count)

            # (S_batch, z_dim)
            q_zi_cond_x_batch = normal_density_cupy(z_samples[count: count + b],
                                                    z_mean[rand_ids[count: count + b], 0, :],
                                                    z_stddev[rand_ids[count: count + b], 0, :],
                                                    eps=eps, gpu=gpu)

            # (1, S_batch, z_dim)
            z_batch = cp.expand_dims(z_samples[count: count + b], axis=0)

            # (N, S_batch, z_dim)
            q_zi_cond_x_all_batch = normal_density_cupy(z_batch, z_mean, z_stddev, eps=eps, gpu=gpu)

            # Computing H_zi_cond_x
            # --------------------------------- #
            H_zi_cond_x += -cp.sum(cp.log(cp.maximum(q_zi_cond_x_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_zi
            # --------------------------------- #
            # Sum (p(x^(n)) * q(zi|x^(n)))
            # (N, 1, 1)  * (N, S_batch, z_dim) then sum axis 0 => (S_batch, z_dim)
            q_zi_batch = cp.sum(cp.expand_dims(cp.expand_dims(x_weights, axis=-1), axis=-1) *
                                q_zi_cond_x_all_batch, axis=0)

            # Sum (S_batch, z_dim) over S_batch => (z_dim,)
            H_zi += -cp.sum(cp.log(cp.maximum(q_zi_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_z_not_i
            # --------------------------------- #
            for i in range(z_dim):
                not_i_axes = [j for j in range(z_dim) if j != i]
                assert len(not_i_axes) == z_dim - 1, "len(not_i_axes) = {}".format(len(not_i_axes))
                # Prod (N, S_batch, z_dim-1) over axis=-1 => (N, S_batch)
                q_z_not_i_cond_x_all_batch = cp.prod(q_zi_cond_x_all_batch[:, :, not_i_axes], axis=-1)
                # (N, 1) * (N, S_batch) then sum over axis=0 => (S_batch, )
                qz_not_i_batch = cp.sum(cp.expand_dims(x_weights, axis=-1) * q_z_not_i_cond_x_all_batch, axis=0)
                H_z_not_i[i] += -cp.sum(cp.log(cp.maximum(qz_not_i_batch, eps)), axis=0)
            # --------------------------------- #

            #'''
            # Computing H_zi_zj
            # --------------------------------- #
            for i in range(z_dim):
                for j in range(i+1, z_dim):
                    # Prod (N, S_batch, 2) over axis=-1 => (N, S_batch)
                    q_zi_zj_cond_x_all_batch = cp.prod(q_zi_cond_x_all_batch[:, :, [i, j]], axis=-1)
                    # (N, 1) * (N, S_batch) then sum over axis=0 => (S_batch,)
                    q_zi_zj_batch = cp.sum(cp.expand_dims(x_weights, axis=-1) * q_zi_zj_cond_x_all_batch, axis=0)
                    H_zi_zj[i, j] += -cp.sum(cp.log(cp.maximum(q_zi_zj_batch, eps)), axis=0)
                    H_zi_zj[j, i] = H_zi_zj[i, j]
            # --------------------------------- #
            # '''

            # Computing H_z
            # --------------------------------- #
            # Prod (N, S_batch, z_dim) over axis=-1 => (N, S_batch)
            q_z_cond_x_all_batch = cp.prod(q_zi_cond_x_all_batch, axis=-1)

            # (N, 1) * (N, S_batch) then sum axis 0 => (S_batch, )
            q_z_batch = cp.sum(cp.expand_dims(x_weights, axis=-1) * q_z_cond_x_all_batch, axis=0)

            H_z += -cp.sum(cp.log(cp.maximum(q_z_batch, eps)), axis=0)
            # --------------------------------- #

            count += b
            progress_bar.update(b)

        progress_bar.close()

        # Convert entropies back to numpy
        H_zi_cond_x = cp.asnumpy(H_zi_cond_x)
        H_zi = cp.asnumpy(H_zi)
        H_z_not_i = cp.asnumpy(H_z_not_i)
        H_zi_zj = cp.asnumpy(H_zi_zj)
        H_z = cp.asnumpy(H_z)

    # (z_dim, )
    H_zi_cond_x = H_zi_cond_x / (1.0 * num_samples)
    H_zi = H_zi / (1.0 * num_samples)
    H_z_not_i = H_z_not_i / (1.0 * num_samples)
    H_zi_zj = H_zi_zj / (1.0 * num_samples)
    H_z = H_z / (1.0 * num_samples)

    MI_zi_x = H_zi - H_zi_cond_x
    MI_zi_zj = np.expand_dims(H_zi, axis=-1) + np.expand_dims(H_zi, axis=0) - H_zi_zj
    SEP_zi = H_zi + H_z_not_i - H_z
    SEP = np.mean(SEP_zi)

    results = {
        "H_zi_cond_x": H_zi_cond_x,
        "H_zi": H_zi,
        "H_z_not_i": H_z_not_i,
        "H_z": H_z,
        "MI_zi_x": MI_zi_x,
        "MI_zi_zj": MI_zi_zj,
        "SEP_zi": SEP_zi,
        "SEP": SEP,
    }

    return results


def estimate_SEPIN_cupy(z_mean, z_stddev, x_weights=None,
                        num_samples=10000, seed=None, batch_size=10, eps=1e-8, gpu=0):
    """
    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param x_weights: A vector of length N
    :param num_samples: Number of samples
    :param seed: Numpy random state used for sampling
    :param batch_size: Batch size during computation
    :param eps: Batch size during computation
    :param gpu: ID of the gpu
    :return:
    """
    print("num_samples: {}".format(num_samples))
    print("batch_size: {}".format(batch_size))

    assert z_mean.shape == z_stddev.shape and len(z_mean.shape) == 2, \
        "z_mean.shape={} and z_stddev={}".format(z_mean.shape, z_stddev.shape)

    num_x, z_dim = z_mean.shape[0], z_mean.shape[1]
    print("num_x: {}".format(num_x))
    print("z_dim: {}".format(z_dim))

    if x_weights is None:
        x_weights = np.full([num_x], fill_value=1.0 / num_x, dtype=np.float32)

    assert 1.0 - 1e-5 <= np.sum(x_weights) <= 1.0 + 1e-5, "'weights' should sum to 1. Found {:.5f}!".format(np.sum(x_weights))

    rs = np.random.RandomState(seed)
    rand_ids = rs.choice(num_x, size=num_samples, replace=True, p=x_weights)

    # (S, z_dim)
    noise = rs.randn(num_samples, z_dim)
    z_samples = z_mean[rand_ids] + noise * z_stddev[rand_ids]

    import cupy as cp
    with cp.cuda.Device(gpu):
        # (S, z_dim)
        z_samples = cp.asarray(z_samples)
        # (N, 1, z_dim)
        z_mean = cp.expand_dims(cp.asarray(z_mean), axis=1)
        # (N, 1, z_dim)
        z_stddev = cp.expand_dims(cp.asarray(z_stddev), axis=1)

        print("z_samples.shape: {}".format(z_samples.shape))
        print("z_mean.shape: {}".format(z_mean.shape))
        print("z_stddev.shape: {}".format(z_stddev.shape))

        x_weights = cp.asarray(x_weights)
        print("x_weights.shape: {}".format(x_weights.shape))

        # (z_dim,)
        H_zi_cond_x = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_zi = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_z_not_i_cond_x = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_z_not_i = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # scalar
        H_z = cp.asarray(np.zeros(1, dtype=np.float32))
        # scalar
        H_z_cond_x = cp.asarray(np.zeros(1, dtype=np.float32))

        progress_bar = tqdm(total=num_samples)
        count = 0
        while count < num_samples:
            b = min(batch_size, num_samples - count)

            # Computing H_zi_cond_x
            # --------------------------------- #
            # (S_batch, z_dim)
            q_zi_cond_x_batch = normal_density_cupy(z_samples[count: count + b],
                                                    z_mean[rand_ids[count: count + b], 0, :],
                                                    z_stddev[rand_ids[count: count + b], 0, :],
                                                    eps=eps, gpu=gpu)

            # Sum over axis=0 of (S_batch, z_dim) => z_dim
            H_zi_cond_x += -cp.sum(cp.log(cp.maximum(q_zi_cond_x_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_zi
            # --------------------------------- #
            # (1, S_batch, z_dim) and (N, S_batch, z_dim) => (N, S_batch, z_dim)
            q_zi_cond_x_all_batch = normal_density_cupy(
                cp.expand_dims(z_samples[count: count + b], axis=0),
                z_mean, z_stddev, eps=eps, gpu=gpu)

            # Sum (p(x^(n)) * q(zi|x^(n)))
            # (N, 1, 1)  * (N, S_batch, z_dim) then sum axis=0 => (S_batch, z_dim)
            q_zi_batch = cp.sum(cp.expand_dims(cp.expand_dims(x_weights, axis=-1), axis=-1) *
                                q_zi_cond_x_all_batch, axis=0)

            # Sum (S_batch, z_dim) over S_batch => (z_dim,)
            H_zi += -cp.sum(cp.log(cp.maximum(q_zi_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_z_not_i and H_z_not_i_cond_x
            # --------------------------------- #
            for i in range(z_dim):
                not_i_axes = [j for j in range(z_dim) if j != i]
                assert len(not_i_axes) == z_dim - 1, "len(not_i_axes)={}".format(len(not_i_axes))

                # (S_batch, z_dim)
                # q_z_not_i_cond_x_batch = cp.prod(q_zi_cond_x_batch[:, not_i_axes], axis=-1)
                q_z_not_i_cond_x_batch = robust_prod_cupy(q_zi_cond_x_batch[:, not_i_axes], axis=-1)

                # (S_batch, z_dim) then sum over axis=0 => (z_dim,)
                H_z_not_i_cond_x[i] += -cp.sum(cp.log(cp.maximum(q_z_not_i_cond_x_batch, eps)), axis=0)

                # Prod (N, S_batch, z_dim-1) over axis=-1 => (N, S_batch)
                # q_z_not_i_cond_x_all_batch = cp.prod(q_zi_cond_x_all_batch[:, :, not_i_axes], axis=-1)
                q_z_not_i_cond_x_all_batch = robust_prod_cupy(q_zi_cond_x_all_batch[:, :, not_i_axes], axis=-1)

                # (N, 1) * (N, S_batch) then sum over axis=0 => (S_batch, )
                qz_not_i_batch = cp.sum(cp.expand_dims(x_weights, axis=-1) * q_z_not_i_cond_x_all_batch, axis=0)
                H_z_not_i[i] += -cp.sum(cp.log(cp.maximum(qz_not_i_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_z and H_z_cond_x
            # --------------------------------- #
            # (S_batch, )
            # q_z_cond_x_batch = cp.prod(q_zi_cond_x_batch, axis=-1)
            q_z_cond_x_batch = robust_prod_cupy(q_zi_cond_x_batch, axis=-1)

            # (S_batch, ) then sum over axis=0 => scalar
            H_z_cond_x += -cp.sum(cp.log(cp.maximum(q_z_cond_x_batch, eps)), axis=0)

            # Prod (N, S_batch, z_dim) over axis=-1 => (N, S_batch)
            # q_z_cond_x_all_batch = cp.prod(q_zi_cond_x_all_batch, axis=-1)
            q_z_cond_x_all_batch = robust_prod_cupy(q_zi_cond_x_all_batch, axis=-1)
            # print("q_z_cond_x_all_batch[:10, :10]:\n{}".format(cp.asnumpy(q_z_cond_x_all_batch[:10, :10])))

            # (N, 1) * (N, S_batch) then sum axis 0 => (S_batch, )
            q_z_batch = cp.sum(cp.expand_dims(x_weights, axis=-1) * q_z_cond_x_all_batch, axis=0)
            # print("q_z_batch[:10]: {}".format(cp.asnumpy(q_z_batch[:10])))

            # scalar
            H_z += -cp.sum(cp.log(cp.maximum(q_z_batch, eps)), axis=0)
            # --------------------------------- #

            count += b
            progress_bar.update(b)

        progress_bar.close()

        # Convert entropies back to numpy
        H_zi_cond_x = cp.asnumpy(H_zi_cond_x)
        H_zi = cp.asnumpy(H_zi)
        H_z_not_i_cond_x = cp.asnumpy(H_z_not_i_cond_x)
        H_z_not_i = cp.asnumpy(H_z_not_i)
        H_z_cond_x = cp.asnumpy(H_z_cond_x)
        H_z = cp.asnumpy(H_z)

    # (z_dim, )
    H_zi_cond_x = H_zi_cond_x / (1.0 * num_samples)
    H_zi = H_zi / (1.0 * num_samples)
    H_z_not_i_cond_x = H_z_not_i_cond_x / (1.0 * num_samples)
    H_z_not_i = H_z_not_i / (1.0 * num_samples)
    H_z_cond_x = H_z_cond_x / (1.0 * num_samples)
    H_z = H_z / (1.0 * num_samples)

    MI_zi_x = np.maximum(H_zi - H_zi_cond_x, 0)
    coeff_zi = MI_zi_x / np.sum(MI_zi_x)
    assert 1 - 1e-4 < np.sum(coeff_zi) < 1 + 1e-4, "np.sum(coeff_zi)={}".format(np.sum(coeff_zi))

    MI_z_not_i_x = np.maximum(H_z_not_i - H_z_not_i_cond_x, 0)
    MI_z_x = np.maximum(H_z - H_z_cond_x, 0)

    SEP_zi = np.maximum(H_zi + H_z_not_i - H_z, 0)
    WSEP = np.sum(coeff_zi * SEP_zi)

    SEPIN_zi = MI_z_x - MI_z_not_i_x
    WSEPIN = np.sum(coeff_zi * SEPIN_zi)

    INDIN_zi = MI_zi_x - SEP_zi
    WINDIN = np.sum(coeff_zi * INDIN_zi)

    print("MI_zi_x: {}".format(MI_zi_x))
    print("MI_z_x: {}".format(MI_z_x))
    print("MI_z_not_i_x: {}".format(MI_z_not_i_x))
    print("SEP_zi: {}".format(SEP_zi))
    print("\nSEPIN_zi: {}".format(SEPIN_zi))
    print("\nINDIN_zi: {}".format(INDIN_zi))
    print("\ncoeff_zi: {}".format(coeff_zi))
    print("\nWSEPIN: {}".format(WSEPIN))
    print("\nSEPIN: {}".format(np.mean(SEPIN_zi)))
    print("\nWINDIN: {}".format(WINDIN))
    print("\nINDIN: {}".format(np.mean(INDIN_zi)))

    results = {
        "H_zi_cond_x": H_zi_cond_x,
        "H_zi": H_zi,
        "H_z_not_i_cond_x": H_z_not_i_cond_x,
        "H_z_not_i": H_z_not_i,
        "H_z_cond_x": H_z_cond_x,
        "H_z": H_z,
        "MI_zi_x": MI_zi_x,
        "MI_z_not_i_x": MI_z_not_i_x,
        "MI_z_x": MI_z_x,
        "SEP_zi": SEP_zi,
        "WSEP": WSEP,
        "SEPIN_zi": SEPIN_zi,
        "WSEPIN": WSEPIN,
        "INDIN_zi": INDIN_zi,
        "WINDIN": WINDIN,
    }
    from six import iteritems
    for key, val in iteritems(results):
        print("{}: {}".format(key, val))

    return results


def estimate_JEMMIG_cupy(z_mean, z_stddev, y, x_weights=None, num_samples=10000,
                         seed=None, batch_size=10, eps=1e-8, gpu=0):
    """
    We simply need to care about log q(zi,yk) - log q(zi) - log q(yk) =
        log sum_x^(n) (q(zi | x^(n)) q(yk | x^(n)))
    Note that, yk^(m) for a particular x^(n) can be 0.

    :param z_mean: (N, z_dim), mean of q(z|x^(n))
    :param z_stddev: (N, z_dim), stddev of q(z|x^(n))
    :param y: (N, num_factors), an 2D int array contains values of factors
    :param x_weights: A vector of length N
    :param num_samples: Number of samples
    :param seed: Numpy random state used for sampling
    :param batch_size: Batch size during computation
    :param eps: Batch size during computation
    :param gpu: ID of the gpu
    :return:
    """
    print("num_samples: {}".format(num_samples))
    print("batch_size: {}".format(batch_size))

    # Processing z
    # ------------------------------------ #
    assert z_mean.shape == z_stddev.shape and len(z_mean.shape) == 2, \
        "z_mean.shape={} and z_stddev={}".format(z_mean.shape, z_stddev.shape)

    num_x, z_dim = z_mean.shape[0], z_mean.shape[1]
    print("num_x: {}".format(num_x))
    print("z_dim: {}".format(z_dim))

    if x_weights is None:
        x_weights = np.full([num_x], fill_value=1.0 / num_x, dtype=np.float32)

    assert 1.0 - 1e-5 <= np.sum(x_weights) <= 1.0 + 1e-5, "'weights' should sum to 1. Found {:.5f}!".format(np.sum(x_weights))

    rs = np.random.RandomState(seed)
    rand_ids = rs.choice(num_x, size=num_samples, replace=True, p=x_weights)

    # (S, z_dim)
    noise = rs.randn(num_samples, z_dim)
    z_samples = z_mean[rand_ids] + noise * z_stddev[rand_ids]
    # ------------------------------------ #

    # Processing y
    # ------------------------------------ #
    print("Processing ground truth factors!")

    assert len(y.shape) == 2 and ('int' in y.dtype.name) and len(y) == num_x, \
        "y.shape={} and y.dtype={}!".format(y.shape, y.dtype.name)

    y_dim = y.shape[1]

    y_unique = []
    y_prob = []
    H_yk = []

    for k in range(y_dim):
        yk_unique, yk_count = np.unique(y[:, k], return_counts=True)
        y_unique.append(yk_unique)

        yk_prob = yk_count / (1.0 * np.sum(yk_count))
        y_prob.append(yk_prob)

        H_yk.append(-np.sum(yk_prob * np.log(np.maximum(yk_prob, eps))))

    H_yk = np.asarray(H_yk, dtype=np.float32)

    print("Done!")
    # ------------------------------------ #

    import cupy as cp
    with cp.cuda.Device(gpu):
        # (S, z_dim)
        z_samples = cp.asarray(z_samples)
        # (N, 1, z_dim)
        z_mean = cp.expand_dims(cp.asarray(z_mean), axis=1)
        # (N, 1, z_dim)
        z_stddev = cp.expand_dims(cp.asarray(z_stddev), axis=1)
        # (N, y_dim)
        y = cp.asarray(y)

        print("z_samples.shape: {}".format(z_samples.shape))
        print("z_mean.shape: {}".format(z_mean.shape))
        print("z_stddev.shape: {}".format(z_stddev.shape))

        x_weights = cp.asarray(x_weights)
        print("x_weights.shape: {}".format(x_weights.shape))

        # (z_dim,)
        H_zi_cond_x = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim,)
        H_zi = cp.asarray(np.zeros(z_dim, dtype=np.float32))
        # (z_dim, y_dim)
        H_zi_yk = cp.asarray(np.zeros([z_dim, y_dim], dtype=np.float32))

        progress_bar = tqdm(total=num_samples)
        count = 0

        while count < num_samples:
            b = min(batch_size, num_samples - count)

            # (S_batch, z_dim)
            q_zi_cond_x_batch = normal_density_cupy(z_samples[count: count + b],
                                                    z_mean[rand_ids[count: count + b], 0, :],
                                                    z_stddev[rand_ids[count: count + b], 0, :],
                                                    eps=eps, gpu=gpu)

            # (1, S_batch, z_dim)
            z_batch = cp.expand_dims(z_samples[count: count + b], axis=0)

            # (N, S_batch, z_dim)
            q_zi_cond_x_all_batch = normal_density_cupy(z_batch, z_mean, z_stddev, eps=eps, gpu=gpu)

            # Computing H_zi_cond_x
            # --------------------------------- #
            H_zi_cond_x += -cp.sum(cp.log(cp.maximum(q_zi_cond_x_batch, eps)), axis=0)
            # --------------------------------- #

            # Computing H_zi
            # --------------------------------- #
            # Sum (p(x^(n)) * q(zi|x^(n)))
            # (N, 1, 1)  * (N, S_batch, z_dim) then sum axis 0 => (S_batch, z_dim)
            q_zi_batch = cp.sum(cp.expand_dims(cp.expand_dims(x_weights, axis=-1), axis=-1) *
                                q_zi_cond_x_all_batch, axis=0)

            # Sum (S_batch, z_dim) over S_batch => (z_dim,)
            H_zi += -cp.sum(cp.log(cp.maximum(q_zi_batch, eps)), axis=0)
            # --------------------------------- #

            for k in range(y_dim):
                # (N, 1) == (1, S_batch) => (N, S_batch)
                q_yk_cond_x_all_batch = cp.equal(cp.expand_dims(y[:, k], axis=-1),
                                                 cp.expand_dims(y[rand_ids[count: count + b], k], axis=0))

                # print("q_yk_cond_x_all_batch[:10, :10]: {}".format(q_yk_cond_x_all_batch[:10, :10]))

                # (N, S_batch, 1)
                q_yk_cond_x_all_batch = cp.expand_dims(
                    cp.asarray(q_yk_cond_x_all_batch, dtype=cp.float32), axis=-1)

                # (N, S_batch, z_dim) * (N, S_batch, 1) then sum over axis=0 => (S_batch, z_dim)
                q_zi_yk_batch = cp.sum(cp.expand_dims(cp.expand_dims(x_weights, axis=-1), axis=-1) *
                                       q_zi_cond_x_all_batch * q_yk_cond_x_all_batch, axis=0)

                # Sum (S_batch, z_dim) over S_batch => (z_dim, )
                H_zi_yk[:, k] += -cp.sum(cp.log(cp.maximum(q_zi_yk_batch, eps)), axis=0)

            count += b
            progress_bar.update(b)

        progress_bar.close()

        # Convert entropies back to numpy
        H_zi_cond_x = cp.asnumpy(H_zi_cond_x)
        H_zi = cp.asnumpy(H_zi)
        H_zi_yk = cp.asnumpy(H_zi_yk)

    # (z_dim, )
    H_zi_cond_x = H_zi_cond_x / (1.0 * num_samples)
    H_zi = H_zi / (1.0 * num_samples)
    H_zi_yk = H_zi_yk / (1.0 * num_samples)

    print("H_yk: {}".format(H_yk))
    print("\nH_zi: {}".format(H_zi))
    print("\nH_zi_yk:\n{}".format(H_zi_yk))

    MI_zi_yk = np.expand_dims(H_zi, axis=-1) + np.expand_dims(H_yk, axis=0) - H_zi_yk

    ids_sorted = []
    MI_zi_yk_sorted = []
    H_zi_yk_sorted = []

    RMIG_yk = []
    RMIG_norm_yk = []
    JEMMIG_yk = []

    for k in range(y_dim):
        # Compute RMIG and JEMMI
        ids_sorted_at_k = np.argsort(MI_zi_yk[:, k], axis=0)[::-1]
        MI_zi_yk_sorted_at_k = np.take_along_axis(MI_zi_yk[:, k], ids_sorted_at_k, axis=0)
        H_zi_yk_sorted_at_k = np.take_along_axis(H_zi_yk[:, k], ids_sorted_at_k, axis=0)

        RMIG_yk_at_k = MI_zi_yk_sorted_at_k[0] - MI_zi_yk_sorted_at_k[1]
        RMIG_norm_yk_at_k = np.divide(RMIG_yk_at_k, H_yk[k])
        JEMMIG_yk_at_k = H_zi_yk_sorted_at_k[0] - MI_zi_yk_sorted_at_k[0] + MI_zi_yk_sorted_at_k[1]

        ids_sorted.append(ids_sorted_at_k)
        MI_zi_yk_sorted.append(MI_zi_yk_sorted_at_k)
        H_zi_yk_sorted.append(H_zi_yk_sorted_at_k)

        RMIG_yk.append(RMIG_yk_at_k)
        RMIG_norm_yk.append(RMIG_norm_yk_at_k)
        JEMMIG_yk.append(JEMMIG_yk_at_k)

    ids_sorted = np.stack(ids_sorted, axis=-1)
    MI_zi_yk_sorted = np.stack(MI_zi_yk_sorted, axis=-1)
    H_zi_yk_sorted = np.stack(H_zi_yk_sorted, axis=-1)

    RMIG_yk = np.asarray(RMIG_yk, dtype=np.float32)
    RMIG_norm_yk = np.asarray(RMIG_norm_yk, dtype=np.float32)
    RMIG_norm = np.mean(RMIG_norm_yk, axis=0)

    JEMMIG_yk = np.stack(JEMMIG_yk, axis=-1)

    results = {
        "H_yk": H_yk,
        "H_zi_cond_x": H_zi_cond_x,
        "H_zi": H_zi,
        "H_zi_yk": H_zi_yk,
        "H_zi_yk_sorted": H_zi_yk_sorted,
        "MI_zi_yk": MI_zi_yk,
        "MI_zi_yk_sorted": MI_zi_yk_sorted,
        "id_sorted": ids_sorted,

        "RMIG_yk": RMIG_yk,
        "RMIG_norm_yk": RMIG_norm_yk,
        "RMIG_norm": RMIG_norm,
        "JEMMIG_yk": JEMMIG_yk,
    }

    return results