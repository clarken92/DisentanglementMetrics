import tensorflow as tf


ALL_MODELS = dict()


def default_model_name(model):
    global ALL_MODELS
    name_id = ALL_MODELS.get(model.__class__.__name__)
    name_id = 1 if name_id is None else name_id + 1
    ALL_MODELS[model.__class__.__name__] = name_id

    name = "{}_{}".format(model.__class__.__name__, name_id)
    return name


class LiteBaseModel:
    def __init__(self):
        self._built = False
        self.output_dict = dict()
        self.loss_coeff_dict = dict()
        self.train_grads_params_dict = None

        self.is_train = tf.placeholder(tf.bool, [], name="is_train")

    def one_if_not_exist(self, d, key, msn_if_not_1="Coefficient of '{}' is: {}",
                         update_loss_coeffs=True):
        if d is None:
            val = 1
        else:
            assert isinstance(d, dict), "'d' must be a dictionary"
            val = d.get(key, None)
            if val is None:
                val = 1

        if (val != 1) and msn_if_not_1 is not None:
            print(msn_if_not_1.format(key, val))

        if update_loss_coeffs:
            self.loss_coeff_dict[key] = val

        return val

    def build(self, *args, **kwargs):
        raise NotImplementedError

    def set_output(self, key, tensor):
        self.output_dict[key] = tensor

    def get_output(self, keys, as_dict=False):
        if isinstance(keys, list):
            if as_dict:
                return {key: self.output_dict[key] for key in keys}
            else:
                return [self.output_dict.get(key, None) for key in keys]

        if isinstance(keys, tuple):
            if as_dict:
                return {key: self.output_dict[key] for key in keys}
            else:
                return (self.output_dict.get(key, None) for key in keys)

        elif isinstance(keys, str):
            if as_dict:
                return {keys: self.output_dict.get(keys, None)}
            else:
                return self.output_dict.get(keys, None)

        else:
            raise ValueError("'keys' must be a string or a list/tuple")

    def get_loss(self):
        # return dict() for each loss term
        raise NotImplementedError

    def get_train_params(self):
        # return dict() for each loss term
        raise NotImplementedError

    def get_train_grads_params(self):
        # return dict() for each loss term
        raise NotImplementedError

    def get_update_ops(self):
        # return dict() for each loss term
        raise NotImplementedError

    def get_all_update_ops(self):
        # print(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
        return tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    def get_feed_dict(self, batch_data):
        # return dict() for each input
        raise NotImplementedError
