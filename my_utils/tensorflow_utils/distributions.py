import tensorflow as tf

from .shaping import flatten_right_from


def KLD_2DiagNs(mu1, log_std1, mu2, log_std2, from_axis=1, clip_log_std=False, eps=1e-15, name=None):
    """
    KL(p1||p2) where p1 is a Diagonal Gaussian N(mu1, log_std1) and
                     p2 is a Diagonal Gaussian N(mu2, log_std2)

    If log_std1 > 50, var1 > exp(100) = inf.
    Thus, we must limit the value of log_std1 to be smaller than 50.
    """

    with tf.name_scope(name or "KLD_2DiagN"):
        if clip_log_std:
            log_std1 = tf.clip_by_value(log_std1, -40.0, 40.0)
            log_std2 = tf.clip_by_value(log_std2, -40.0, 40.0)

        var1 = tf.exp(2 * log_std1)
        var2 = tf.exp(2 * log_std2)

        kl = (log_std2 - log_std1) - 0.5 + 0.5 * (var1 + (mu1 - mu2)**2) / (var2 + eps)

        if from_axis is not None:
            kl = flatten_right_from(kl, axis=from_axis)
            kl = tf.reduce_sum(kl, axis=from_axis)

        return kl


def KLD_2DiagNs_v2(mu1, std1, mu2, std2, from_axis=1, eps=1e-15, big_pos=1e30, name=None):
    with tf.name_scope(name or "KLD_2DiagN_v2"):
        log_std1 = tf.log(tf.clip_by_value(std1, eps, big_pos))
        log_std2 = tf.log(tf.clip_by_value(std2, eps, big_pos))

        var1 = std1 ** 2
        var2 = std2 ** 2

        kl = (log_std2 - log_std1) - 0.5 + 0.5 * (var1 + (mu1 - mu2)**2) / (var2 + eps)

        if from_axis is not None:
            kl = flatten_right_from(kl, axis=from_axis)
            kl = tf.reduce_sum(kl, axis=from_axis)

        return kl


def KLD_DiagN_N01(mu, log_std, from_axis=1, clip_log_std=False, name=None):
    """
    KL(p||N(0,1)) where p is a Diagonal Gaussian N(mu, log_std)
    Also support individual KLD of each components if from_axis is set to None
    """

    with tf.name_scope(name or "KLD_DiagN_N01"):
        if clip_log_std:
            log_std = tf.clip_by_value(log_std, -40.0, 40.0)

        var = tf.exp(2 * log_std)

        kl = -log_std - 0.5 + 0.5 * (mu**2 + var)

        if from_axis is not None:
            if from_axis < 0:
                from_axis = kl.shape.ndims - from_axis

            kl = flatten_right_from(kl, axis=from_axis)
            kl = tf.reduce_sum(kl, axis=from_axis)

        return kl


def KLD_DiagN_N01_v2(mu, std, from_axis=1, eps=1e-15, big_pos=1e15, name=None):
    with tf.name_scope(name or "KLD_DiagN_N01_v2"):
        log_std = tf.log(tf.clip_by_value(std, eps, big_pos))
        var = std ** 2

        kl = -log_std - 0.5 + 0.5 * (mu**2 + var)

        if from_axis is not None:
            if from_axis < 0:
                from_axis = kl.shape.ndims - from_axis

            kl = flatten_right_from(kl, axis=from_axis)
            kl = tf.reduce_sum(kl, axis=from_axis)

        return kl
