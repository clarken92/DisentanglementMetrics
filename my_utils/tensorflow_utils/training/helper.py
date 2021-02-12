from six import iteritems
from os.path import join, dirname, abspath
import json
from collections import OrderedDict

import numpy as np
import tensorflow as tf


# Different from the old version, this version support saving best model
class SimpleTrainHelper(object):
    def __init__(self, log_dir, save_dir, var_list=None, max_to_keep=1, max_to_keep_best=1):
        self.log_dir = log_dir
        self.summary_writer = None

        # Only save model at the last epoch
        self.saver = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep)
        # Only save best model
        self.saver_best = tf.train.Saver(var_list=var_list, max_to_keep=max_to_keep_best)

        # `save_path` only contains prefix.
        # suffix is added automatically by Tensorflow
        self.save_dir = save_dir
        self.save_path = join(abspath(save_dir), "model")
        self.save_path_best = join(abspath(save_dir), "best_model")

    def initialize(self, sess, init_variables=True, create_summary_writer=True,
                   do_load=False, load_path="", load_step=0, load_best=False):

        if init_variables:
            sess.run(tf.global_variables_initializer())

        if create_summary_writer:
            self.create_summary_writer()

        if do_load:
            if load_best:
                self.load_best(sess, load_path, load_step)
            else:
                self.load(sess, load_path, load_step)

    def save(self, sess, global_step=None):
        self.saver.save(sess, self.save_path, global_step=global_step)

    def save_best(self, sess, global_step=None):
        self.saver_best.save(sess, self.save_path_best, global_step=global_step)

    def save_separately(self, sess, model_name, global_step=None):
        # Save model separately, which do not use the current Saver
        # Suitable for taking snapshot of model at some fixed points
        saver = tf.train.Saver()
        saver.save(sess, join(self.save_dir, model_name), global_step=global_step)

    def load(self, sess, load_path="", load_step=0):
        if len(load_path) == 0:
            print("'load_path' is not set! Use the default 'save_path'=[{}]".format(self.save_path))
            load_path = self.save_path

        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint from [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        print("Load model from [{}]".format(load_path))
        self.saver.restore(sess, load_path)

    def load_best(self, sess, load_path, load_step=0):
        if len(load_path) == 0:
            print("'load_path' is not set! Use the default 'save_path'=[{}]-best".format(self.save_path_best))
            load_path = self.save_path_best

        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint from [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        print("Load model from [{}]".format(load_path))
        self.saver.restore(sess, load_path)

    def get_var_names_in_ckpt(self):
        save_dir = self.save_dir
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        assert checkpoint is not None, "Cannot load checkpoint from [{}]!".format(save_dir)
        ckpt_path = tf.train.latest_checkpoint(save_dir)

        objs = tf.train.list_variables(ckpt_path)
        var_names = [obj[0] for obj in objs]

        return var_names

    def create_summary_writer(self):
        self.summary_writer = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())

    def add_summary(self, summary, global_step):
        self.summary_writer.add_summary(summary, global_step)

    def add_summaries(self, summaries, global_step):
        for summary in summaries:
            self.add_summary(summary, global_step)


class SimpleModelLoader(object):
    @staticmethod
    def load_var_list(var_list, sess, load_path, load_step=-1):
        saver = tf.train.Saver(var_list)

        print("Load model from [{}] with step {}".format(load_path, load_step))
        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint at [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        print("Load model from path [{}]".format(load_path))
        saver.restore(sess, load_path)

    @staticmethod
    def load_var_scopes(var_scopes, sess, load_path, load_step=-1):
        """
        Load some of the saved model (according to var_scopes)
        to the model (graph) in `sess`
        :param sess: The current session
        :param load_path: Path to a saved model
        :param load_step: Step of a saved model
        :param trainable_only: Only trainable variables (default=False)
        :return:
        """
        print("Load variable scopes: {}".format(var_scopes))
        variables = []
        for scope in var_scopes:
            variables.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
        saver = tf.train.Saver(var_list=variables)

        print("Load model from [{}] with step {}".format(load_path, load_step))
        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint at [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        print("Load model from path [{}]".format(load_path))
        saver.restore(sess, load_path)

    @staticmethod
    def load_all_saved(sess, load_path, load_step=-1, trainable_only=True):
        """
        Load the entire model (graph) from `load_path` to (part of)
        the model (graph) in `sess`
        :param sess: The current session
        :param load_path: Path to a saved model
        :param load_step: Step of a saved model
        :param trainable_only: Only trainable variables (default=False)
        :return:
        """
        print("Load entire model from [{}] at step {}".format(load_path, load_step))

        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint at [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        key = tf.GraphKeys.TRAINABLE_VARIABLES if trainable_only else tf.GraphKeys.GLOBAL_VARIABLES

        # Create a new session and load the small graph within this session
        new_sess = tf.Session(graph=tf.Graph())
        with new_sess.as_default():
            with new_sess.graph.as_default():
                assert new_sess is tf.get_default_session()
                assert new_sess.graph is tf.get_default_graph()

                assert not (sess is tf.get_default_session())
                assert not (sess.graph is tf.get_default_graph())

                meta_file = load_path + ".meta"
                saver = tf.train.import_meta_graph(meta_file)

                saver.restore(new_sess, load_path)
                variables = tf.get_collection(key)
                print("Saved variables: {}".format([var.name for var in variables]))
                values = new_sess.run(variables)
                loaded_variables = [(var.name, val) for var, val in zip(variables, values)]

        new_sess.close()
        del new_sess

        variables = tf.get_collection(key)
        print("Current variables: {}".format([var.name for var in variables]))
        variable_dict = {var.name: var for var in variables}

        load_ops = []
        for name, val in loaded_variables:
            load_ops.append(variable_dict[name].assign(val))

        load_ops = tf.group(*load_ops)
        sess.run(load_ops)

    @staticmethod
    def load_normal(sess, load_path="", load_step=-1):
        saver = tf.train.Saver()

        print("Load model from [{}] with step {}".format(load_path, load_step))
        if load_step > 0:
            load_path += "-{}".format(load_step)
        else:
            save_dir = dirname(load_path)
            checkpoint = tf.train.get_checkpoint_state(save_dir)
            assert checkpoint is not None, "Cannot load checkpoint from [{}]!".format(save_dir)
            load_path = tf.train.latest_checkpoint(save_dir)

        print("Load model from [{}]".format(load_path))
        saver.restore(sess, load_path)

    @staticmethod
    def get_vars_in_ckpt(self, load_path=""):
        vars_in_ckpt = tf.train.list_variables(load_path)
        return vars_in_ckpt


class SimpleParamPrinter(object):
    @staticmethod
    def _variables2json(var_names, var_info):
        root = dict()
        var_names = [name.split('/') for name in var_names]

        for n, name in enumerate(var_names):
            node = root
            for i, key in enumerate(name):
                if not (key in node):
                    if i < len(name) - 1:
                        node[key] = dict()
                    else:
                        node[key] = var_info[n]
                node = node[key]
        json_str = json.dumps(root, ensure_ascii=True, indent=4)
        return json_str

    @staticmethod
    def print_all_params_json(trainable_only=True):
        """
        List all parameters of a `sess.graph`
        :param trainable_only: Only list trainable variable
        :return:
        """
        if trainable_only:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

        var_names = [var.name for var in variables]
        var_shapes = [var.get_shape().as_list() for var in variables]

        var_info = list()
        for shape in var_shapes:
            var_info.append({"shape": "{}".format(shape)})

        json_str = SimpleParamPrinter._variables2json(var_names, var_info)
        print(json_str)

    @staticmethod
    def print_all_params_list(trainable_only=True):
        """
        List all parameters of a `sess.graph`
        :param sess:
        :param trainable_only: Only list trainable variable
        :return:
        """
        if trainable_only:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        # Shape for every variable
        var_names = [var.name for var in variables]
        var_shapes = [var.get_shape().as_list() for var in variables]

        output_str = list()
        output_str.append("=" * 50)
        output_str.append("Shape of all variables:\n")
        for name, shape in zip(var_names, var_shapes):
            output_str.append("{} | shape: {}".format(name, shape))

        output_str.append("=" * 50)
        output_str.append("#params of all variable scopes:\n")

        scope_num_params = OrderedDict()
        total_num_params = 0
        for name, shape in zip(var_names, var_shapes):
            scope = name.split("/")[0]

            if scope in scope_num_params:
                scope_num_params[scope] += np.prod(shape)
            else:
                scope_num_params[scope] = np.prod(shape)

        for key, val in iteritems(scope_num_params):
            output_str.append("{}: {}".format(key, val))
            total_num_params += val

        output_str.append("=" * 50)
        output_str.append("Total #params: {}".format(total_num_params))

        output_str = "\n".join(output_str)
        print(output_str)

    @staticmethod
    def print_all_params_tf_slim(trainable_only=True):
        if trainable_only:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        else:
            variables = tf.get_collection(tf.GraphKeys.MODEL_VARIABLES)

        import tensorflow.contrib.slim as slim
        slim.model_analyzer.analyze_vars(variables, print_info=True)

