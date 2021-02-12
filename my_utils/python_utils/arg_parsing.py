import argparse
import json


def str2bool(s):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    if s.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif s.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def str2ints(s, sep=';'):
    try:
        return [int(i) for i in s.split(sep)]
    except Exception:
        raise argparse.ArgumentTypeError("A string representing a list of ints "
                                         "separated by {} is expected!".format(sep))


def str2floats(s, sep=';'):
    try:
        return [float(i) for i in s.split(sep)]
    except Exception:
        raise argparse.ArgumentTypeError("A string representing a list of floats "
                                         "separated by {} is expected!".format(sep))


def str2path(s):
    if len(s) > 0 and '/' not in s:
        s = './' + s
    return s


def str2dict(s):
    return json.loads(s)


def args2dict(args):
    if isinstance(args, argparse.Namespace):
        return vars(args)
    else:
        assert isinstance(args, dict), "If 'args' is not argparse.Namespace, it must be a dict!"
        return args


def save_args(save_file, args, excluded_keys=()):
    if len(excluded_keys) > 0:
        # Prevent removing existing keys from args
        import copy
        args = copy.deepcopy(args2dict(args))

        for key in excluded_keys:
            args.pop(key, None)
    else:
        args = args2dict(args)

    with open(save_file, "w") as f:
        json.dump(args, f, indent=4, sort_keys=True)


def update_args(args, new_args, excluded_keys=()):
    assert isinstance(args, argparse.Namespace), "'args' must be argparse.Namespace!"
    new_args = args2dict(new_args)

    if len(excluded_keys) > 0:
        # Prevent removing existing keys from new_args
        import copy
        args = copy.deepcopy(args2dict(args))

        for key in excluded_keys:
            new_args.pop(key, None)
    else:
        args = args2dict(args)

    from six import iteritems
    for key, val in iteritems(new_args):
        if key in args:
            args.__setattr__(key, val)
    return args


def assert_keys(args1, args2, keys):
    args1 = args2dict(args1)
    args2 = args2dict(args2)

    for key in keys:
        val1 = args1.get(key)
        val2 = args2.get(key)

        if (val1 is not None) or (val2 is not None):
            assert val1 == val2, "args1[{}]={} while args2[{}]={}!".format(key, val1, key, val2)


def extract_keys(args, pattern):
    if isinstance(args, argparse.Namespace):
        args = vars(args)
    assert isinstance(args, dict)

    import re
    p = re.compile(pattern)

    matched_keys = []
    for key in args.keys():
        if p.match(key):
            matched_keys.append(key)
    return matched_keys