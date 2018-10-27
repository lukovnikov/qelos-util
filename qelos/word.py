import torch
import qelos as q
from collections import OrderedDict
import os
import numpy as np
import json
from copy import deepcopy


__all__ = ["WordEmb"]


class WordEmb(torch.nn.Embedding):
    masktoken = "<MASK>"
    """ is a VectorEmbed with a dictionary to map words to ids """
    def __init__(self, dim=None, worddic=None,
                 max_norm=None, norm_type=2, scale_grad_by_freq=False,
                 sparse=False, no_masking=False, word_dropout=0.,
                 _weight=None,
                 **kw):
        """
        Normal word embedder. Subclasses nn.Embedding.

        :param dim: embedding vector dimension
        :param worddic: worddic, must be provided
        :param value: (optional) value to set the weight of nn.Embedding to
        :param max_norm: see nn.Embedding
        :param norm_type: see nn.Embedding
        :param scale_grad_by_freq: see nn.Embedding
        :param sparse: see nn.Embedding
        :param fixed: fixed embeddings
        :param no_masking: ignore usual mask id (default "<MASK>") in this instance of WordEmb
            --> no masking (will return no mask), useful for using WordEmb in output vectors
        :param word_dropout: if >0, applies word-level embeddings (zeros complete word vectors).
                             The word dropout mask is shared across timesteps and examples in a batch.
                             Must call rec_reset() to sample new dropout mask for a new batch.
        :param kw:
        """
        assert(worddic is not None)     # always needs a dictionary
        self.D = OrderedDict() if worddic is None else worddic
        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive

        # extract maskid from worddic
        maskid = worddic[self.masktoken] if self.masktoken in worddic else None
        maskid = maskid if not no_masking else None
        indim = max(worddic.values())+1        # to init from worddic

        super(WordEmb, self).__init__(indim, dim, padding_idx=maskid,
                                      max_norm=max_norm, norm_type=norm_type,
                                      scale_grad_by_freq=scale_grad_by_freq,
                                      sparse=sparse, _weight=_weight)

        if _weight is None:
            self.reset_parameters()
        self.word_dropout = q.RecDropout(p=word_dropout) if word_dropout > 0 else None

    def reset_parameters(self):
        initrange = 0.1
        self.weight.data.uniform_(-initrange, initrange)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def forward(self, x):
        ret = super(WordEmb, self).forward(x)

        if self.word_dropout is not None:
            word_dropout_mask = torch.ones(self.weight.size(0), 1, device=self.weight.device)
            word_dropout_mask = self.word_dropout(word_dropout_mask).clamp(0, 1)
            x_drop = word_dropout_mask[x]
            ret = ret * x_drop
        mask = None
        if self.padding_idx is not None:
            mask = (x != self.padding_idx).int()
        return ret, mask

    def freeze(self, val:bool=True):
        self.weight.requires_grad = not val

    @classmethod
    def load_pretrained(cls, W, D, **kw): #weights, worddic=None):
        # check consistency between D and W
        assert(W.dim() == 2)
        vocsize, dim = W.size()
        vocsizeD = max(D.values())+1
        assert(vocsizeD == vocsize)
        # create
        ret = cls(dim=dim, worddic=D, _weight=W, **kw)
        return ret

    @classmethod
    def load_pretrained_path(cls, path, **kw):
        W, D = WordEmb._load_path(path)
        W = torch.tensor(W)
        ret = cls.load_pretrained(W, D, **kw)
        return ret

    @classmethod
    def transform_to_format(cls, path, outpath):
        pass    # TODO: implement transformation from normal format to numpy + worddic format

    @staticmethod
    def _load_path(path):   # TODO: optionally, only retain a given set of words in memory
        """ Loads a path. Returns a numpy array (vocsize, dim) and dictionary from words to ids"""
        tt = q.ticktock("wordvec loader")

        # load weights
        tt.tick()
        W = np.load(path+".npy")
        tt.tock("vectors loaded")

        # load words
        tt.tick()
        with open(path+".words") as f:
            words = json.load(f)
        tt.tock("words loaded")

        D = dict(zip(words, range(len(words))))
        return W, D


class OverriddenWordEmb(torch.nn.Module):
    def __init__(self, base:WordEmb, override:WordEmb, which=None, whichnot=None):
        super(OverriddenWordEmb, self).__init__()
        self.D = base.D
        self.base = base
        self.over = override.adapt(base.D)
        self.vecdim = self.base.vecdim
        assert(not (which is not None and whichnot is not None))
        numout = max(base.D.values()) + 1
        whichnot = set()

        overridemask_val = np.zeros((numout,), dtype="float32")
        if which is None:   # which: list of words to override
            for k, v in base.D.items():     # for all symbols in base dic
                if k in override.D and k not in whichnot:         # if also in override dic
                    overridemask_val[v] = 1
        else:
            for k in which:
                if k in override.D:     # TODO: if k from which is missing from base.D
                    overridemask_val[base.D[k]] = 1
        self.overridemask = q.val(overridemask_val).v

    def forward(self, x):
        x = x.contiguous()
        xshape = x.size()
        x = x.view(-1)
        base_emb, base_msk = self.base(x)
        over_emb, over_msk = self.over(x)
        over_msk_select = torch.gather(self.overridemask, 0, x)
        emb = base_emb * (1 - over_msk_select.unsqueeze(1)) + over_emb * over_msk_select.unsqueeze(1)
        emb = emb.view(*(xshape + (-1,)))
        msk = None
        if base_msk is not None:
            msk = base_msk.view(xshape)
        return emb, msk