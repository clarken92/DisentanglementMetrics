from six import iteritems
from collections import Counter

import math
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


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

        MI += (1.0 * num_ai_bi) / (1.0 * n) * (math.log(max(1.0 * n * num_ai_bi, eps))
                                               - math.log(max(1.0 * num_ai * num_bi, eps)))

    return MI


def discrete_entropy(a, eps=1e-8):
    # a: an array of ints, shape (batch,)

    assert isinstance(a, np.ndarray) and ('int' in a.dtype.name) and len(a.shape) == 1, \
        "'a' must be a 1D Numpy array of ints!"

    n = len(a)
    a_counter = Counter(a.tolist())

    ent = 0
    for ai, num_ai in iteritems(a_counter):
        ai_prob = (1.0 * num_ai) / (1.0 * n)

        ent -= ai_prob * math.log(max(ai_prob, eps))

    return ent


def histogram_discretize(x, num_bins):
    # x: an array of floats, shape (batch,)

    assert isinstance(x, np.ndarray) and ('float' in x.dtype.name) and len(x.shape) == 1, \
        "'x' must be a 1D Numpy array of floats!"

    hist, bin_edges = np.histogram(x, num_bins)
    x_disc = np.digitize(x, bin_edges[:-1])
    return x_disc


def compute_modularity(z, y, is_discrete_z, is_discrete_y, num_bins, eps=1e-8):
    """
    :param z: (num_samples, num_latents)
    :param y: (num_samples, num_factors)
    :param num_bins:
    :param eps:
    :return:
    """
    assert (len(z.shape) == len(y.shape) == 2) and (z.shape[0] == y.shape[0]), \
        "z.shape={} and y.shape={}".format(z.shape, y.shape)

    num_latents = z.shape[1]
    num_factors = y.shape[1]

    if isinstance(is_discrete_z, bool):
        is_discrete_z = [is_discrete_z] * num_latents
    assert isinstance(is_discrete_z, (list, tuple)) and len(is_discrete_z) == num_latents

    if isinstance(is_discrete_y, bool):
        is_discrete_y = [is_discrete_y] * num_factors
    assert isinstance(is_discrete_y, (list, tuple)) and len(is_discrete_y) == num_factors

    MI_zi_yk = np.zeros([num_latents, num_factors], dtype=np.float32)
    H_yk = np.zeros([num_factors], dtype=np.float32)

    for k in range(num_factors):
        yk = y[:, k]
        if not is_discrete_y[k]:
            yk = histogram_discretize(yk, num_bins)
        H_yk[k] = discrete_entropy(yk)

        for i in range(num_latents):
            zi = z[:, i]
            if not is_discrete_z[i]:
                zi = histogram_discretize(zi, num_bins)

            MI_zi_yk[i, k] = discrete_mutual_info(zi, yk, eps=eps)
            # print("MI(z{}, y{}) = {:.3f}".format(i, k, MI_zi_yk[i, k]))

    # Sorted in decreasing order
    zi_ids_sorted = np.argsort(MI_zi_yk, axis=0)[::-1]
    MI_zi_yk_sorted = np.take_along_axis(MI_zi_yk, zi_ids_sorted, axis=0)

    MIG_yk = np.divide(MI_zi_yk_sorted[0, :] - MI_zi_yk_sorted[1, :], np.maximum(H_yk, eps))
    MIG = np.mean(MIG_yk)

    # (num_latents,)
    yk_idx_max = np.argmax(MI_zi_yk, axis=1)
    MI_zi_yk_top_over_k = MI_zi_yk[np.arange(0, num_latents), yk_idx_max]

    T = np.zeros([num_latents, num_factors], dtype=np.float32)
    for i in range(num_latents):
        T[i, yk_idx_max[i]] = MI_zi_yk[i, yk_idx_max[i]]

    modularity_scores = 1 - (np.sum((MI_zi_yk - T) ** 2, axis=1) / (MI_zi_yk_top_over_k**2 * (num_factors-1)))
    modularity = np.mean(modularity_scores, axis=0)

    np.set_printoptions(suppress=True, precision=4, threshold=np.nan, linewidth=1000)
    print("\nMI_zi_yk:\n{}".format(MI_zi_yk))
    print("\nyk_idx_max:\n{}".format(yk_idx_max))
    print("\nMI_zi_yk_top_over_k:\n{}".format(MI_zi_yk_top_over_k))
    print("\nT:\n{}".format(T))
    print("modularity_scores: {}".format(modularity_scores))

    results = {
        'MI_zi_yk': MI_zi_yk,
        'H_yk': H_yk,
        'MI_zi_yk_sorted': MI_zi_yk_sorted,
        'zi_ids_sorted': zi_ids_sorted,
        'MIG_yk': MIG_yk,
        'MIG': MIG,
        'modularity_scores': modularity_scores,
        'modularity': modularity
    }

    return results


def compute_explicitness(z, y):
    assert isinstance(y, np.ndarray) and ('int' in y.dtype.name) and len(y.shape) == 2, \
        "'y' must be a 2D Numpy array of ints!"

    assert (len(z.shape) == len(y.shape) == 2) and (z.shape[0] == y.shape[0]), \
        "z.shape={} and y.shape={}".format(z.shape, y.shape)

    num_factors = y.shape[1]

    AUCs_mean_yk = np.zeros(num_factors, dtype=np.float32)

    all_AUCs = []
    all_AUCs_factors = []
    all_AUCs_factor_vals = []

    for k in range(num_factors):
        model = LogisticRegression(C=1e10, multi_class='ovr', solver='liblinear',
                                   max_iter=1000, tol=1e-5)

        model.fit(z, y[:, k])
        preds = model.predict_proba(z)

        aucs = []

        print("Factor {} has {} classes".format(k, len(model.classes_)))
        for idx, val in enumerate(model.classes_):
            print("\rFinish computing {} values!".format(idx + 1), end='')
            y_true = y[:, k] == val
            y_pred = preds[:, idx]

            auc = roc_auc_score(y_true, y_pred)
            aucs.append(auc)

            all_AUCs_factor_vals.append(val)

        AUCs_mean_yk[k] = np.mean(aucs)

        all_AUCs.extend(aucs)
        all_AUCs_factors.extend([k] * len(aucs))

    explicitness_scores = AUCs_mean_yk
    explicitness = np.mean(explicitness_scores, axis=0)

    results = {
        'AUCs_mean_yk': AUCs_mean_yk,
        'all_AUCs': all_AUCs,
        'all_AUCs_factors': all_AUCs_factors,
        'all_AUCs_factor_vals': all_AUCs_factor_vals,
        'explicitness_scores': explicitness_scores,
        'explicitness': explicitness,
    }
    return results
