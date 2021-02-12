import numpy as np
from tqdm import tqdm
from sklearn.linear_model import Lasso
from sklearn.ensemble.forest import RandomForestRegressor


def mse(predicted, target):
    # mean square error
    predicted = predicted[:, None] if len(predicted.shape) == 1 else predicted  # (n,)->(n,1)
    target = target[:, None] if len(target.shape) == 1 else target  # (n,)->(n,1)
    err = predicted - target
    err = err.T.dot(err) / len(err)
    return err[0, 0]


def rmse(predicted, target):
    # root mean square error
    return np.sqrt(mse(predicted, target))


def nmse(predicted, target, eps=1e-8):
    # normalized mean square error
    return mse(predicted, target) / np.maximum(np.var(target), eps)


def nrmse(predicted, target, eps=1e-8):
    # normalized root mean square error
    return rmse(predicted, target) / np.maximum(np.std(target), eps)


def entropic_scores(R, eps=1e-8):
    # R: importance matrix: (num_latents, num_factors)
    R = np.abs(R)

    # ps: distribution over latents: (num_latents, num_factors)
    P = R / np.maximum(np.sum(R, axis=0), eps)
    # print("ps:\n{}".format(P))
    # print("sum(ps, axis=0): {}".format(np.sum(P, axis=0)))

    # H_norm: (num_factors,)
    H_norm = -np.sum(P * np.log(np.maximum(P, eps)), axis=0)
    if P.shape[0] > 1:
        H_norm = H_norm / np.log(P.shape[0])

    # print("H_norm: {}".format(H_norm))
    return 1 - H_norm


# Run with different parameters
def compute_metrics_with_LASSO(latents, factors, err_fn=nrmse,
                               params={"alpha": 0.02}, cont_mask=None):
    """
    :param latents: (N, z_dim). They use E_q(z|x)[z]
    :param factors: (N, K)
    :param err_fn: Error function
    :param params: Parameters of LASSO
    :param cont_mask: Continuous mask
    :return:
    """

    assert len(latents.shape) == len(factors.shape) == 2, \
        "'latents' and 'factors' must be 2D arrays!"
    assert len(latents) == len(factors), "'latents' and 'factors' must have the same length!"

    num_factors = factors.shape[1]

    if not cont_mask:
        cont_mask = [True] * num_factors
    else:
        assert len(cont_mask) == num_factors, "len(cont_mask)={}".format(len(cont_mask))

    R = []
    train_errors = []

    print("Training LASSO regressor for {} factors!".format(num_factors))
    for k in tqdm(range(num_factors)):
        if cont_mask[k]:
            print("Factor {} is continuous. Process it!".format(k))
            # (N, )
            factors_k = factors[:, k]

            model = Lasso(**params)
            model.fit(latents, factors_k)

            # (N, )
            factors_k_pred = model.predict(latents)
            print("\nAfter training factor {}".format(k))
            print("factor_k[:20]: {}".format(factors_k[:20]))
            print("factor_k_pred[:20]: {}".format(factors_k_pred[:20]))
            print("factor_k[-20:]: {}".format(factors_k[-20:]))
            print("factor_k_pred[-20:]: {}".format(factors_k_pred[-20:]))
            # Scalar
            train_errors.append(err_fn(factors_k_pred, factors_k))
            print("train_error_k: {}".format(train_errors[-1]))

            # Get the weight of the linear regressor, whose shape is (num_latents, 1)
            R.append(np.abs(model.coef_[:, None]))
        else:
            print("Factor {} is not continuous. Do not process it!".format(k))

    # (num_latents, num_factors)
    R = np.concatenate(R, axis=1)
    assert R.shape[1] == np.sum(np.asarray(cont_mask, dtype=np.int32)), \
        "R.shape={} while #cont={}".format(
        R.shape[1], np.sum(np.asarray(cont_mask, dtype=np.int32)))

    # Disentanglement: (num_latents,)
    disentanglement_scores = entropic_scores(R.T)
    print("disentanglement_scores: {}".format(disentanglement_scores))

    c_rel_importance = np.sum(R, axis=1) / np.sum(R)  # relative importance of each code variable
    assert 1 - 1e-4 < np.sum(c_rel_importance) < 1 + 1e-4, \
        "c_rel_importance: {}".format(c_rel_importance)
    disentanglement = np.sum(disentanglement_scores * c_rel_importance)

    # Completeness
    completeness_scores = entropic_scores(R)
    print("completeness_scores: {}".format(completeness_scores))
    completeness = np.mean(completeness_scores)

    # Informativeness
    train_avg_error = np.mean(train_errors)
    print("train_avg_error: {}".format(train_avg_error))

    results = {
        'cont_mask': cont_mask,
        'importance_matrix': R,
        'disentanglement_scores': disentanglement_scores,
        'disentanglement': disentanglement,
        'completeness_scores': completeness_scores,
        'completeness': completeness,
        'train_errors': train_errors,
        'train_avg_error': train_avg_error,
    }

    return results


def compute_metrics_with_RandomForest(latents, factors, err_fn=nrmse,
                                      params={"n_estimators": 10, "max_depth": 8},
                                      cont_mask=None):
    """
    :param latents: (N, z_dim). They use E_q(z|x)[z]
    :param factors: (N, K)
    :param err_fn: Error function
    :param params: Parameters of LASSO
    :return:
    """

    assert len(latents.shape) == len(factors.shape) == 2, \
        "'latents' and 'factors' must be 2D arrays!"
    assert len(latents) == len(factors), "'latents' and 'factors' must have the same length!"

    num_factors = factors.shape[1]

    R = []
    train_errors = []

    if not cont_mask:
        cont_mask = [True] * num_factors
    else:
        assert len(cont_mask) == num_factors, "len(cont_mask)={}".format(len(cont_mask))

    print("Training Random Forest regressor for {} factors!".format(num_factors))
    for k in tqdm(range(num_factors)):
        if cont_mask:
            print("Factor {} is continuous. Process it!".format(k))

            # (N, )
            factors_k = factors[:, k]
            model = RandomForestRegressor(**params)
            model.fit(latents, factors_k)

            # (N, )
            factors_k_pred = model.predict(latents)

            # Scalar
            train_errors.append(err_fn(factors_k_pred, factors_k))

            # Get the weight of the linear regressor, whose shape is (num_latents, 1)
            R.append(np.abs(model.feature_importances_[:, None]))
        else:
            print("Factor {} is not continuous. Do not process it!".format(k))

    # (num_latents, num_factors)
    R = np.concatenate(R, axis=1)
    assert R.shape[1] == np.sum(np.cast(cont_mask, dtype=np.int32)), \
        "R.shape={} while #cont={}".format(
            R.shape[1], np.sum(np.cast(cont_mask, dtype=np.int32)))

    # Disentanglement: (num_latents,)
    disentanglement_scores = entropic_scores(R.T)
    c_rel_importance = np.sum(R, axis=1) / np.sum(R)  # relative importance of each code variable
    assert 1 - 1e-4 < np.sum(c_rel_importance) < 1 + 1e-4, \
        "c_rel_importance: {}".format(c_rel_importance)
    disentanglement = np.sum(disentanglement_scores * c_rel_importance)

    # Completeness
    completeness_scores = entropic_scores(R)
    completeness = np.mean(completeness_scores)

    # Informativeness
    train_avg_error = np.mean(train_errors)

    results = {
        'importance_matrix': R,
        'disentanglement_scores': disentanglement_scores,
        'disentanglement': disentanglement,
        'completeness_scores': completeness_scores,
        'completeness': completeness,
        'train_errors': train_errors,
        'train_avg_error': train_avg_error,
    }

    return results
