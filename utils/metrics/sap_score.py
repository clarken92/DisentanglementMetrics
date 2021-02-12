import numpy as np
from sklearn import svm


def _compute_score_matrix(z_train, y_train, z_test, y_test, cont_ys):
    """
    :param z_train: (num_latents, num_samples)
    :param y_train: (num_factors, num_samples)
    :param z_test: (num_latents, num_samples)
    :param y_test: (num_factors, num_samples)
    :param cont_ys: (num_factors)
    :return:
    """

    num_latents = z_train.shape[0]  # Number of latent components
    num_factors = y_train.shape[0]  # Number of factor components

    # Initialize a score matrix
    score_matrix = np.zeros([num_latents, num_factors])

    for i in range(num_latents):
        for j in range(num_factors):
            zi = z_train[i]  # The i-th latent: (num_train, )
            yj = y_train[j]  # The j-th factor: (num_train, )
            if cont_ys[j]:
                # Attribute is considered continuous.
                # [[zi zi, zi yj]
                #  [zi yj, yj yj]]
                cov_mat_zi_yj = np.cov(zi, yj, ddof=1)
                assert cov_mat_zi_yj.shape == (2, 2), "'cov_mat_zi_yj.shape': {}".\
                    format(cov_mat_zi_yj.shape)

                cov_zi_yj = cov_mat_zi_yj[0, 1]
                var_zi = cov_mat_zi_yj[0, 0]
                var_yj = cov_mat_zi_yj[1, 1]

                if var_zi * var_yj > 1e-12:
                    score_matrix[i, j] = (cov_zi_yj ** 2) / (var_zi * var_yj)
                else:
                    score_matrix[i, j] = 0.

            else:
                # If attributes are discrete, use a linear classifier.
                # Attribute is considered discrete.
                zi_test = z_test[i]
                yj_test = y_test[j]

                classifier = svm.LinearSVC(C=0.01, class_weight="balanced")
                classifier.fit(np.expand_dims(zi, axis=-1), yj)
                yj_pred = classifier.predict(np.expand_dims(zi_test, axis=-1))

                score_matrix[i, j] = np.mean(np.equal(yj_pred, yj_test).astype(np.int32))

    return score_matrix


def _compute_avg_diff_top_two(matrix):
    sorted_matrix = np.sort(matrix, axis=0)
    return np.mean(sorted_matrix[-1, :] - sorted_matrix[-2, :])


def compute_sap(z_train, y_train, z_test, y_test, cont_ys):
    # Compute score matrix based on training/testing
    score_matrix = _compute_score_matrix(z_train, y_train, z_test, y_test, cont_ys)

    # Score matrix should have shape [num_latents, num_factors].
    assert score_matrix.shape[0] == z_train.shape[0]
    assert score_matrix.shape[1] == y_train.shape[0]

    sap_score = _compute_avg_diff_top_two(score_matrix)
    return sap_score

