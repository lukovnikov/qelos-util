import torch
import qelos as q
from collections import OrderedDict
import os
import numpy as np
import json
from copy import deepcopy


__all__ = ["WordEmb", "SwitchedWordEmb", "WordLinout"]


class VectorLoader(object):
    @classmethod
    def load_pretrained_path(cls, path, selectD=None, **kw):
        W, D = WordEmb._load_path(path)
        ret = cls.load_pretrained(W, D, selectD=selectD, **kw)
        return ret

    @classmethod
    def transform_to_format(cls, path, outpath):
        pass    # TODO: implement transformation from normal format to numpy + worddic format

    @staticmethod
    def _load_path(path):
        """ Loads a path. Returns a numpy array (vocsize, dim) and dictionary from words to ids
            :param path:    path where to load embeddings from. Must contain .npy and .words files.
        """
        tt = q.ticktock("wordvec loader")

        # load weights
        tt.tick()
        W = np.load(path+".npy")
        tt.tock("vectors loaded")

        # load words
        tt.tick()
        with open(path+".words") as f:
            words = json.load(f)
            D = dict(zip(words, range(len(words))))
        tt.tock("words loaded")
        return W, D


class WordEmb(torch.nn.Embedding, VectorLoader):
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

        # word dropout
        self.word_dropout = torch.nn.Dropout(p=word_dropout) if word_dropout > 0 else None
        self._word_dropout_mask = None

    def batch_reset(self):
        self._word_dropout_mask = None

    def reset_parameters(self):
        initrange = 0.1
        self.weight.data.uniform_(-initrange, initrange)
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)

    def do_word_dropout(self, x):   # (batsize, ..., dim)
        if self.training and self.word_dropout is not None:
            if self._word_dropout_mask is None:
                # sample new mask
                word_dropout_mask = torch.ones(self.weight.size(0), 1, device=self.weight.device)
                word_dropout_mask = self.word_dropout(word_dropout_mask).clamp(0, 1)
                self._word_dropout_mask = word_dropout_mask     # cache mask for this batch
            x_drop = self._word_dropout_mask[x]
            x = x * x_drop
        return x

    def forward(self, x):
        ret = super(WordEmb, self).forward(x)
        ret = self.do_word_dropout(ret)
        mask = None
        if self.padding_idx is not None:
            mask = (x != self.padding_idx).int()
        return ret, mask

    def freeze(self, val:bool=True):
        self.weight.requires_grad = not val

    @classmethod
    def load_pretrained(cls, W, D, selectD=None, **kw): #weights, worddic=None):
        """
        :param W:   numpy matrix of weights (numwords, dim)
        :param D:   associated dictionary mappings words (string) to ids (int)
        :param selectD: (optional) dictionary used to construct new matrix based on a selection from W
        :param kw:  any kwargs to pass into WordEmb constructor
        :return:    WordEmb
        """
        # check consistency between D and W
        assert(len(W.shape) == 2)
        vocsize, dim = W.shape
        vocsizeD = max(D.values())+1
        assert(vocsizeD == vocsize)

        # rearrange according to newD
        if selectD is not None:
            vocsize = max(selectD.values()) + 1
            new_weight = np.zeros((vocsize, W.shape[1]), dtype=W.dtype)
            new_dic = {}
            for k, v in selectD.items():
                if k in D:
                    new_weight[v, :] = W[D[k], :]
                    new_dic[k] = v
            W, D = new_weight, new_dic

        # create
        W = torch.tensor(W)
        ret = cls(dim=dim, worddic=D, _weight=W, **kw)
        return ret


class SwitchedWordEmb(torch.nn.Module):
    """
    WordEmb that contains multiple WordEmbs and switches between them based on settings.
    Uses word dropout and mask of base WordEmb !!!
    """
    def __init__(self, base, **kw):
        super(SwitchedWordEmb, self).__init__()
        self.base = base
        self.D = self.base.D
        self.other_embs = torch.nn.ModuleList()
        self.register_buffer("select_mask",
            torch.zeros(self.base.weight.size(0), 1, dtype=torch.int64, device=self.base.weight.device))

    def override(self, emb:WordEmb, selectwords=None):
        """
        :param emb:     WordEmb whose entries will override the base (and previous overrides)
        :param selectwords:   which words to override. If None, emb.D's keys are used.
        :return:
        """
        if hasattr(emb, "word_dropout") and emb.word_dropout is not None:
            print("WARNING: word dropout of base will be applied before output. "
                  "Word dropout on the emb provided here will be applied before that "
                  "and the combined effect may result in over-dropout.")
        if selectwords is None:
            selectwords = set(emb.D.keys())
        # ensure that emb.D maps words in self.base.D to the same id as self.base.D
        selid = len(self.other_embs) + 1
        for k, v in self.D.items():
            if v > emb.weight.size(0):
                raise q.SumTingWongException("the override must contain all positions of base.D but doesn't have ('{}':{})".format(k, v))
            if k in emb.D and k in selectwords:
                if emb.D[k] != v:
                    raise q.SumTingWongException("the override emb must map same words to same id "
                                                 "but {} maps to {} in emb.D and to {} in self.base.D"
                                                 .format(k, emb.D[k], v))
                # update select_mask
                self.select_mask[v] = selid
        self.other_embs.append(emb)
        return self

    def forward(self, x):   # (batsize, ...,) int ids
        baseemb, basemask = self.base(x)
        otherembs = [other(x)[0] for other in self.other_embs]
        catemb = torch.stack([baseemb]+otherembs, -1)   # (batsize, ..., dim, numembs)
        selmask = self.select_mask[x]   # (batsize, ..., 1) int ids of which embedder to use
        selmaskrep = [1] * (selmask.dim() - 1) + [catemb.size(-2)]
        selmask = selmask.repeat(*selmaskrep).unsqueeze(-1)
        ret = torch.gather(catemb, -1, selmask).squeeze(-1) # (batsize, ..., dim)
        ret = self.base.do_word_dropout(ret)
        return ret, basemask


class WordLinout(torch.nn.Linear):
    def __init__(self, dim=None, worddic=None, bias=True, _weight=None, _bias=None, **kw):
        assert(worddic is not None)     # always needs a dictionary
        self.D = OrderedDict() if worddic is None else worddic
        wdvals = list(worddic.values())
        assert(min(wdvals) >= 0)     # word ids must be positive
        outdim = max(worddic.values())+1        # to init from worddic

        super(WordLinout, self).__init__(dim, outdim, bias=bias)

        if _weight is None:
            assert(_bias is None)
            self.reset_parameters()
        else:
            assert(_bias is not None or bias is False)
            self.weight = torch.nn.Parameter(_weight)
            if _bias is not None:
                self.bias = torch.nn.Parameter(_bias)

    def reset_parameters(self):
        initrange = 0.1
        self.weight.data.uniform_(-initrange, initrange)
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        ret = super(WordLinout, self).forward(x)
        return ret

    def freeze(self, val:bool=True):
        self.weight.requires_grad = not val
        self.bias.requires_grad = not val

    @classmethod
    def load_pretrained(cls, W, b=None, D=None, selectD=None, **kw): #weights, worddic=None):
        """
        :param W:   numpy matrix of weights (numwords, dim)
        :param D:   associated dictionary mappings words (string) to ids (int)
        :param selectD: (optional) dictionary used to construct new matrix based on a selection from W
        :param kw:  any kwargs to pass into WordEmb constructor
        :return:    WordEmb
        """
        # check consistency between D and W
        assert(len(W.shape) == 2)
        vocsize, dim = W.shape
        vocsizeD = max(D.values())+1
        assert(vocsizeD == vocsize)

        # rearrange according to newD
        if selectD is not None:
            vocsize = max(selectD.values()) + 1
            new_weight = np.zeros((vocsize, W.shape[1]), dtype=W.dtype)
            new_bias = np.zeros((vocsize,), dtype=b.dtype) if b is not None else None
            new_dic = {}
            for k, v in selectD.items():
                if k in D:
                    new_weight[v, :] = W[D[k], :]
                    if new_bias is not None:
                        new_bias[v] = b[D[k]]
                    new_dic[k] = v
            W, D, b = new_weight, new_dic, new_bias

        # create
        W = torch.tensor(W)
        b = torch.tensor(b) if b is not None else None
        ret = cls(dim=dim, worddic=D, _weight=W, _bias=b, **kw)
        return ret