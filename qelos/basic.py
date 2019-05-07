import torch
import math
from abc import ABC, abstractmethod


__all__ = ["GeLU", "Swish", "RecDropout", "Stateful"]


class Stateful(ABC):
    statevars = []

    @abstractmethod
    def batch_reset(self):
        pass

    @classmethod
    def get_state_(cls, x):
        state = {}
        if isinstance(x, Stateful):
            for statevar in x.statevars:
                if hasattr(x, statevar):
                    state_val = getattr(x, statevar)
                    if not isinstance(state_val, torch.nn.Module):
                        if isinstance(state_val, Stateful):
                            state_val_states = cls.get_state_(state_val)
                            state_val_states = {".".join([statevar, k]): v for k, v in state_val_states.items()}
                            state.update(state_val_states)
                        else:
                            state[statevar] = state_val
        if isinstance(x, torch.nn.Module):
            for childname, child in x.named_children():
                childrenstates = cls.get_state_(child)
                childrenstates = {".".join([childname, k]): v for k, v in childrenstates.items()}
                state.update(childrenstates)
        return state

    @classmethod
    def set_state_(cls, x, s):
        for k, v in s.items():
            xe = x
            kpieces = k.split(".")
            for kpiece in kpieces[:-1]:
                xe = getattr(xe, kpiece)
            setattr(xe, kpieces[-1], v)

    def get_state(self):
        return self.get_state_(self)

    def set_state(self, s):
        self.set_state_(self, s)


# region from huggingface github transformer
class GeLU(torch.nn.Module):
    def forward(self, x):
        # return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Swish(torch.nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class RecDropout(torch.nn.Module, Stateful):
    """ Variational Dropout for use in rec cells.
        Uses the same dropout masks until rec_reset() is called (then mask is resampled on next forward call) """
    statevars = ["mask"]
    def __init__(self, p=0, shareaxis=None):
        """
        :param p:   dropout probability
        :param shareaxis:   axis (int or tuple of int) for sharing the dropout mask across
        """
        super(RecDropout, self).__init__()
        self.dropout = torch.nn.Dropout(p=p)
        self.p = p
        self.mask = None
        self.shareaxis = (shareaxis,) if isinstance(shareaxis, int) else shareaxis

    def batch_reset(self):
        self.mask = None

    def forward(self, x):
        y = x
        if self.training:
            shareaxis = self.shareaxis
            if shareaxis is None:
                shareaxis = []
            if self.mask is None:
                mask_shape = [x.size(i) if i not in shareaxis else 1
                              for i in range(x.dim())]
                mask = torch.ones(*mask_shape).to(x.device)
                self.mask = self.dropout(mask)
            y = x * self.mask
        return y

    @staticmethod
    def convert_to_standard_in(m, names=None):
        """ Convert RecDropouts contained in given module m to normal nn.Dropouts.
            Use names=None as additional filter (only listed names will be converted). """
        for _name, child in m.named_children():
            if (names is None or _name in names) and isinstance(child, RecDropout):
                a = getattr(m, _name)
                b = torch.nn.Dropout(p=a.p)
                setattr(m, _name, b)
            RecDropout.convert_to_standard_in(child, names=names)