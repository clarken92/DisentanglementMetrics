from six import iteritems
import tensorflow as tf


def custom_tf_scalar_summary(key, value, prefix=None):
    tag = key if prefix is None else "{}/{}".format(prefix, key)
    return tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])


def custom_tf_scalar_summaries(keys_values, prefix=None):
    summaries = []
    for key, value in iteritems(keys_values):
        summaries.append(custom_tf_scalar_summary(key, value, prefix=prefix))
    return summaries