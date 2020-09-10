from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class LRSchedule(LambdaLR):
    def __init__(self, optimizer, schedule, **kw):
        super(LRSchedule, self).__init__(optimizer, schedule, **kw)
        self._schedule = schedule


class HyperparamSchedule(object):
    def __init__(self, hyperparam, schedule, **kw):
        super(HyperparamSchedule, self).__init__(**kw)
        self.hyperparam = hyperparam
        self.schedule = schedule

    def step(self):
        value = self.schedule.get_lr()
        self.hyperparam._v = value
        return value


class SchedulePiece(object):
    def __init__(self, steps:int=None, **kw):
        super(SchedulePiece, self).__init__(**kw)
        assert(steps is not None)
        self.num_steps = steps

    def get_lr_sched(self, optimizer):
        return LRSchedule(optimizer, self)

    def bind(self, optimizer):
        pass        # TODO: implement this

    def __rshift__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other, steps=np.infty)
        ret = CatSchedule(self, other)
        return ret

    def __add__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other, steps=np.infty)
        ret = SumSchedule(self, other)
        return ret

    def __sub__(self, other):
        return self.__add__(-1 * other)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            other = Constant(other, steps=np.infty)
        return ProdSchedule(self, other)

    def get_lr(self, i):
        if i >= self.num_steps:
            return None
        ret = self._get_lr(i)
        return ret

    def _get_lr(self, i):
        pass

    def __call__(self, i):
        return self.get_lr(i)


class CatSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = sum([arg.num_steps for arg in args])
        super(CatSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def get_lr(self, i):
        lr = None
        for piece in self.pieces:
            if i >= piece.num_steps:     # piece must have been finalized
                i = i - piece.num_steps
            else:
                lr = piece.get_lr(i)
                break
        return lr


class SumSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = min([arg.num_steps for arg in args])
        super(SumSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def get_lr(self, i):
        lr = 0.
        for piece in self.pieces:
            _lr = piece.get_lr(i)
            if _lr is None:
                return None
            lr += _lr
        return lr


class ProdSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = min([arg.num_steps for arg in args])
        super(ProdSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def get_lr(self, i):
        lr = 1.
        for piece in self.pieces:
            _lr = piece.get_lr(i)
            if _lr is None:
                return None
            lr *= _lr
        return lr


class Constant(SchedulePiece):
    def __init__(self, val:float, **kw):
        super(Constant, self).__init__(**kw)
        self.val = val

    def _get_lr(self, _):
        return self.val


class Linear(SchedulePiece):
    def __init__(self, start:int=0, end:int=1, **kw):
        super(Linear, self).__init__(**kw)
        self.start, self.end = start, end
        if self.num_steps > 0:
            self._slope = (self.end - self.start) / self.num_steps
        else:
            self._slope = 0
        self._offset = start

    def _get_lr(self, i):
        i = i + .5
        return i * self._slope + self._offset


class Lambda(SchedulePiece):
    def __init__(self, f, **kw):
        super(Lambda, self).__init__(**kw)
        self.f = f

    def _get_lr(self, i):
        return self.f(i)


class Cosine(SchedulePiece):
    def __init__(self, cycles:float=0.5, phase:float=0.,
                 low:float=0, high:float=1, **kw):
        super(Cosine, self).__init__(**kw)
        self.cycles, self.phase = cycles, phase
        self.low, self.high = low, high
        _low, _high = low + 1, high - 1
        self._add, self._scale = (_low + _high) / 2, (high - low) / 2

    def _get_lr(self, i):
        perc = i / self.num_steps
        ret = (np.cos((perc * self.cycles + self.phase) * 2 * np.pi) * self._scale) + self._add
        return ret


def try_cosine():
    import matplotlib.pyplot as plt
    sched = Cosine(cycles=1, steps=100)
    x = list(range(100))
    y = [sched.get_lr(i) for i in x]
    plt.plot(x, y)
    plt.show()


def try_cat():
    import matplotlib.pyplot as plt
    sched = (Linear(steps=10) >> Cosine(steps=90) >> Cosine(steps=100)) * Linear(steps=200, start=1, end=0)
    # sched = Linear(steps=5)
    x = list(range(sched.num_steps))
    y = [sched.get_lr(i) for i in x]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # try_cosine()
    try_cat()
