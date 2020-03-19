from abc import ABC, abstractmethod

import torch
import numpy as np
from torch.optim.lr_scheduler import LambdaLR


class LRSchedule(LambdaLR):
    def __init__(self, optimizer, schedule, **kw):
        super(LRSchedule, self).__init__(optimizer, lambda epoch: schedule.step(), **kw)
        self._schedule = schedule


class SchedulePiece(object):
    def __init__(self, steps:int=None, **kw):
        super(SchedulePiece, self).__init__(**kw)
        assert(steps is not None)
        self.num_steps = steps
        self.i = 0

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

    def step(self):
        if self.i >= self.num_steps:
            return None
        ret = self._step()
        self.i += 1
        return ret

    def _step(self):
        pass


class CatSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = sum([arg.num_steps for arg in args])
        super(CatSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def step(self):
        lr = None
        i = 0
        while lr is None and i < len(self.pieces):
            lr = self.pieces[i].step()
            i += 1
        return lr


class SumSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = min([arg.num_steps for arg in args])
        super(SumSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def step(self):
        lr = 0.
        for piece in self.pieces:
            _lr = piece.step()
            if _lr is None:
                return None
            lr += _lr
        return lr


class ProdSchedule(SchedulePiece):
    def __init__(self, *args, **kw):
        steps = min([arg.num_steps for arg in args])
        super(ProdSchedule, self).__init__(steps=steps, **kw)
        self.pieces = args

    def step(self):
        lr = 1.
        for piece in self.pieces:
            _lr = piece.step()
            if _lr is None:
                return None
            lr *= _lr
        return lr


class Constant(SchedulePiece):
    def __init__(self, val:float, **kw):
        super(Constant, self).__init__(**kw)
        self.val = val

    def step(self):
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

    def _step(self):
        return self.i * self._slope + self._offset


class Lambda(SchedulePiece):
    def __init__(self, f, **kw):
        super(Lambda, self).__init__(**kw)
        self.f = f

    def _step(self):
        return self.f(self.i)


class Cosine(SchedulePiece):
    def __init__(self, cycles:float=0.5, phase:float=0.,
                 low:float=0, high:float=1, **kw):
        super(Cosine, self).__init__(**kw)
        self.cycles, self.phase = cycles, phase
        self.low, self.high = low, high
        _low, _high = low + 1, high - 1
        self._add, self._scale = (_low + _high) / 2, (high - low) / 2

    def _step(self):
        perc = self.i / self.num_steps
        ret = (np.cos((perc * self.cycles + self.phase) * 2 * np.pi) * self._scale) + self._add
        return ret


def try_cosine():
    import matplotlib.pyplot as plt
    sched = Cosine(cycles=1, steps=100)
    x = list(range(100))
    y = [sched.step() for _ in x]
    plt.plot(x, y)
    plt.show()


def try_cat():
    import matplotlib.pyplot as plt
    sched = (Linear(steps=10) >> Cosine(steps=90) >> Cosine(steps=100)) * Linear(steps=200, start=1, end=0)
    x = list(range(sched.num_steps))
    y = [sched.step() for _ in x]
    plt.plot(x, y)
    plt.show()


if __name__ == '__main__':
    # try_cosine()
    try_cat()
