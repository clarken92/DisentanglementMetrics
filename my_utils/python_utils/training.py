from six import moves, iteritems
from os.path import exists
import random
import shutil
import numpy as np


# ----------------------- Train Iteration ----------------------- #
# Use frequently
def iterate_data(data_size_or_ids, batch_size,
                 shuffle=False, seed=None, include_remaining=True):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    :param data_size_or_ids:
    :param batch_size:
    :param shuffle:
    :param seed: None for complete randomisation
    :param include_remaining:
    :return:
    """
    if isinstance(data_size_or_ids, int):
        data_size = data_size_or_ids
        ids = list(range(data_size_or_ids))
    else:
        assert hasattr(data_size_or_ids, '__len__')
        ids = data_size_or_ids.tolist() if isinstance(data_size_or_ids, np.ndarray) \
            else list(data_size_or_ids)
        data_size = len(data_size_or_ids)

    rs = np.random.RandomState(seed)
    if shuffle:
        rs.shuffle(ids)
    nb_batch = len(ids) // batch_size

    for batch in moves.xrange(nb_batch):
        yield ids[batch * batch_size: (batch + 1) * batch_size]

    if include_remaining and nb_batch * batch_size < data_size:
        yield ids[nb_batch * batch_size:]


# Sampler
# --------------------------------------- #
# Use frequently
# This sampler repeatedly iterates over a dataset
class ContinuousIndexSampler(object):
    """
    V1.0: Stable running
    V1.1: Add seed and local RandomState
    """
    def __init__(self, data_size_or_ids, sample_size, shuffle=False, seed=None):
        self.sample_size = sample_size
        self.shuffle = shuffle
        self.seed = seed
        # It is OK to have a RandomState like this
        self._rs = np.random.RandomState(self.seed)

        if isinstance(data_size_or_ids, int):
            self.ids = list(range(data_size_or_ids))
        else:
            assert hasattr(data_size_or_ids, '__len__')
            self.ids = list(data_size_or_ids)

        self.sids = []
        while len(self.sids) < self.sample_size:
            self.sids += self._renew_ids()
        self.pointer = 0

    def _renew_ids(self):
        rids = [idx for idx in self.ids]
        if self.shuffle:
            # If seed is None, a completely new RandomState is created each time
            # rs = np.random.RandomState(self.seed)
            # rs.shuffle(rids)

            # We do not need to create a new RandomState each time
            self._rs.shuffle(rids)
        return rids

    def sample_ids(self):
        if self.pointer + self.sample_size > len(self.sids):
            self.sids = self.sids[self.pointer:] + self._renew_ids()
            while len(self.sids) < self.sample_size:
                self.sids += self._renew_ids()
            self.pointer = 0

        return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
        self.pointer += self.sample_size
        return return_ids

    def sample_ids_continuous(self):
        while True:
            if self.pointer + self.sample_size > len(self.sids):
                self.sids = self.sids[self.pointer:] + self._renew_ids()
                while len(self.sids) < self.sample_size:
                    self.sids += self._renew_ids()
                self.pointer = 0

            return_ids = self.sids[self.pointer: self.pointer + self.sample_size]
            self.pointer += self.sample_size
            yield return_ids
# --------------------------------------- #


class StoppingCondition(object):
    def __init__(self, best_improvement_steps=0, next_improvement_steps=0, mode='min'):
        """
        :param next_improvement_steps: Number of steps for next result improvement.
               If 0, stop immediately after seeing no improvement
               If n, after seeing no improvement, wait n more steps
        :param best_improvement_steps: Number of steps for best result improvement.
               If 0, stop immediately after seeing no improvement
               If n, after seeing no improvement, wait n more steps
        :param mode: min/max
        """

        self.next_imp_steps = next_improvement_steps
        self.best_imp_steps = best_improvement_steps

        assert mode in ['min', 'max'], "`mode` can only be either 'min' or 'max'!"
        if mode == 'min':
            self.comp_fn = lambda a, b: a <= b
            self.best_result = np.inf
            self.prev_result = np.inf
        else:
            self.comp_fn = lambda a, b: a >= b
            self.best_result = -np.inf
            self.prev_result = -np.inf

        self.curr_next_imp_step = 0
        self.curr_best_imp_step = 0

    def continue_running(self, result):
        # Improve over the best result
        if self.comp_fn(result, self.best_result):
            self.best_result = result
            self.curr_best_imp_step = 0
        else:
            self.curr_best_imp_step += 1

        # Improve over the previous result
        if self.comp_fn(result, self.prev_result):
            self.prev_result = result
            self.curr_next_imp_step = 0
        else:
            self.curr_next_imp_step += 1

        # No improvement over best or previous results after a predefined number of steps
        if self.curr_best_imp_step > self.best_imp_steps or \
           self.curr_next_imp_step > self.next_imp_steps:
            return False
        else:
            return True
