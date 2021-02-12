from six import moves, iteritems
from os import makedirs
from os.path import exists
import json
from functools import reduce
import operator
import shutil
from datetime import datetime

import numpy as np


def and_composite(*logic_funcs):
    def composite_func(x):
        result = True
        for func in logic_funcs:
            out = func(x)
            assert isinstance(out, bool)
            result = result and out
        return result
    return composite_func


def composite(*funcs):
    def composite_func(x):
        for func in funcs:
            x = func(x)
        return x
    return composite_func


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def append_and_return(l, val):
    assert isinstance(l, list), "'l' must be a list!"
    l.append(val)
    return val


def print_both(text="", file=None, end="\n"):
    print(text, end=end)

    if file is not None:
        file.write(text)
        file.write(end)


def load_json(file_path):
    with open(file_path, "r") as f:
        obj = json.load(f)
    return obj


def save_json(obj, file_path):
    with open(file_path, "w") as f:
        json.dump(obj, f, indent=4, sort_keys=True)


# One hot
def one_hot(indices, num_categories=None, dtype='float32'):
    if num_categories is None:
        num_categories = int(np.max(indices) + 1)

    indices = np.asarray(indices, dtype=np.int32)
    shape = [s for s in indices.shape]
    indices = np.reshape(indices, [prod(shape)]).astype(np.int32)
    return np.reshape(np.eye(num_categories, dtype=dtype)[indices], shape + [num_categories])


def one_hot_2_category(x, cat_slices=None):
    # x: a 2D array of shape (batch, dim)
    # cat_slices: A list of 2-tuples specify the start and end indices of categorical variables
    x = np.asarray(x, dtype=np.float32)
    assert len(x.shape) == 2, "'x' must be a 2D array!"

    if cat_slices is None:
        cat_slices = [(0, x.shape[1])]

    output = []

    old_end_idx = 0
    for i, (start_idx, end_idx) in enumerate(cat_slices):
        assert start_idx < end_idx, "Categorical variable {} lie in the slice [{}, {})!".format(
            i, start_idx, end_idx)
        assert old_end_idx <= start_idx, "Categorical variable {} starts from dimension {} but " \
            "categorical variable {} ends at dimension {}!".format(i, start_idx, i-1, old_end_idx)

        if old_end_idx < start_idx:
            x_noncat = x[:, old_end_idx: start_idx]
            output.append(x_noncat)

        assert np.equal(np.sum(x[:, start_idx: end_idx], axis=1), 1).all()
        x_cat = np.expand_dims(np.argmax(x[:, start_idx: end_idx], axis=1), axis=-1).astype(np.float32)
        output.append(x_cat)

        old_end_idx = end_idx

    if old_end_idx < x.shape[1]:
        x_noncat = x[:, old_end_idx:]
        output.append(x_noncat)

    return np.concatenate(output, axis=1)


def multi_one_hots(x, m, dtype='float32'):
    """
    :param x: a List of list of indices
    :param m: Number of possible values
    :param dtype: dtype of the one-hot vector
    :return: An multi-one-hot numpy array of 2D dimension
    """

    z = np.zeros([len(x), m], dtype=dtype)
    for i, _x in enumerate(x):
        assert len(np.asarray(_x).shape) == 1, "`x` must be a list of list of int!"
        z[i, _x] = 1
    z = np.vstack(z)
    assert len(z) == len(x)
    return z


def multi_one_hots2indices(x):
    assert isinstance(x, np.ndarray)
    assert len(x.shape) == 2

    z = []
    for i in moves.xrange(x.shape[0]):
        z.append([j for j in moves.xrange(x.shape[1]) if x[i, j] == 1])
    return z


# For preparing directories before training
def remove_dir_if_exist(dir_path, ask_4_permission=True):
    if exists(dir_path):
        if ask_4_permission:
            check = input("Is it OK to remove '{}'?\n(y/n)".format(dir_path))
            if check.lower() == 'y':
                print("Remove existed folder: '{}'".format(dir_path))
                shutil.rmtree(dir_path)
        else:
            print("Remove existed folder: '{}'".format(dir_path))
            shutil.rmtree(dir_path)

    return dir_path


def make_dir_if_not_exist(dir_path):
    if not exists(dir_path):
        makedirs(dir_path)
    return dir_path


def dict_w_valid_keys(d, keys):
    new_d = dict()

    for key in keys:
        val = d.get(key, None)
        if val is not None:
            new_d[key] = val

    return new_d


def dict_w_new_keys(d, prefix="", suffix=""):
    assert isinstance(d, dict), "'d' must be a dict. Found {}!".format(type(d))
    assert isinstance(suffix, str), "'suffix' must be a str. Found {}!".format(type(suffix))
    new_d = {prefix + key + suffix: value for key, value in iteritems(d)}
    return new_d


def to_list(x):
    if not hasattr(x, '__len__') or isinstance(x, str):
        return [x]
    elif isinstance(x, (list, tuple, set)):
        return list(x)
    elif isinstance(x, np.ndarray):
        return x.tolist()
    else:
        raise ValueError("Do not support object of type '{}'!".format(type(x)))


def int2binary(x, start_from_left=True, max_len=-1):
    assert isinstance(x, int), "'x' must be an integer!"
    b = bin(x)[2:]
    b = [int(s) for s in b]

    if max_len > 0:
        assert len(b) <= max_len, "The binary sequence length of " \
            "{} is {}, bigger than 'max_len'={}!".format(x, len(b), max_len)
        b = [0 for _ in range(max_len - len(b))] + b

    # Currently, it start from right so 2 will be convert to [1, 0] = 2^1 + 0 * 2^0
    if start_from_left:
        b = b[::-1]

    return b


# Parsing
# ------------------------ #
def get_arg_string(args, include_explicit_args=False):
    arg_str = ""
    for key, val in iteritems(args):
        if isinstance(val, dict):
            new_val = json.dumps(val)
            arg_str += "--{}='{}' ".format(key, new_val)
        elif isinstance(val, str):
            arg_str += '--{}="{}" '.format(key, val)
        elif isinstance(val, (list, tuple)):
            arg_str += '--{} '.format(key)
            for v in val:
                if isinstance(v, str):
                    arg_str += '"{}" '.format(v)
                else:
                    arg_str += '{} '.format(v)
        else:
            arg_str += '--{}={} '.format(key, val)

    if include_explicit_args:
        arg_str += "--_explicit_args='{}' ".format(json.dumps(args))
    arg_str = arg_str.strip()

    return arg_str
# ------------------------ #


def default_value_if_not_exist(dictionary, key, default_val, msn_if_not_default=None):
    # Check whether the 'key' is in 'dictionary'. Otherwise return default value
    if dictionary is None:
        val = default_val
    else:
        assert isinstance(dictionary, dict), "'d' must be a dictionary"
        val = dictionary.get(key, None)
        if val is None:
            val = default_val

    if (val != default_val) and (msn_if_not_default is not None):
        print(msn_if_not_default.format(key, val))
    return val


def reshape_axes(axes, n_row, n_col):
    # Reshape a 1D axes into a 2D axes of shape (n_row, n_col)
    assert n_row >= 1 and n_col >= 1
    assert isinstance(axes, list) and len(axes) == n_row * n_col

    if n_row == 1 and n_col == 1:
        return axes[0]

    elif (n_row == 1 and n_col > 1) or (n_row > 1 and n_col == 1):
        return axes
    else:
        new_axes = []

        n = 0
        for i in range(n_row):
            new_axes.append(axes[n: n+ n_col])
            n += n_col

        return new_axes


def get_meshgrid(center, spans_each_side, points_each_side, return_center_idx=False):
    """
    center: A 2-tuple describe the center value
    span_each_side: A 2-tuple describe the span value along EACH side
    points_each_side: A 2-tuple describe the number of points along EACH side
    """
    assert hasattr(center, '__len__') and len(center) == 2, "'center' must be an array of length 2!"
    assert hasattr(spans_each_side, '__len__') and len(spans_each_side) == 2, \
        "'spans_each_side' must be an array of length 2!"
    assert hasattr(points_each_side, '__len__') and len(points_each_side) == 2, \
        "'points_each_side' must be an array of length 2!"

    c1, c2 = center
    s1, s2 = spans_each_side
    p1, p2 = points_each_side

    assert s1 > 0 and s2 > 0, "'spans_each_side' must contain positive floats!"
    assert (isinstance(p1, int) and p1 > 0) and (isinstance(p2, int) and p2 > 0), \
        "'points_each_side' must contain positive integers!"

    r1 = [(c1 - s1) + 1.0 * i * s1 / p1 for i in range(p1)]
    r1 += [c1]
    r1 += [c1 + 1.0 * i * s1 / p1 for i in range(1, p1 + 1)]

    r2 = [(c2 - s2) + 1.0 * i * s2 / p2 for i in range(p2)]
    r2 += [c2]
    r2 += [c2 + 1.0 * i * s2 / p2 for i in range(1, p2 + 1)]

    y_val, x_val = np.meshgrid(r1, r2, indexing='ij')

    y_val = np.ravel(y_val)
    x_val = np.ravel(x_val)

    meshgrid = np.stack([y_val, x_val], axis=-1)
    center_idx = p1 * (2 * p2 + 1) + p2

    if return_center_idx:
        return meshgrid, center_idx
    else:
        return meshgrid


def current_datetime():
    now = datetime.now()
    return now.strftime("%Y_%m_%d_%H_%M_%S")