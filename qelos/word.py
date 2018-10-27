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
        weight = self.weight
        if self.word_dropout is not None:
            word_dropout_mask = torch.ones(weight.size(0), 1, device=weight.device)
            word_dropout_mask = self.word_dropout(word_dropout_mask)
            weight = weight * word_dropout_mask
        ret = torch.nn.functional.embedding(x, weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm,
            norm_type=self.norm_type,
            scale_grad_by_freq=self.scale_grad_by_freq,
            sparse=self.sparse)

        mask = None
        if self.padding_idx is not None:
            mask = (x != self.padding_idx).int()
        return ret, mask

    def freeze(self, val:bool=True):
        self.weight.requires_grad = not val

    def replace_vectors(self, W, wdic, keep=None):
        """
        :param W:       (Tensor) matrix containing new vectors
        :param wdic:    dictionary containing mappings
                            from words that need to be replaced in self.D
                            to ids of vectors in given W
                        (keys need not all be covered by self.D)
        :param keep:    set of words that must not be replaced even if they are in given wdic
        :return:
        """
        if keep is not None:
            wdic = {k: v for k, v in wdic.items() if k not in keep}
        for k, v in self.D.items():
            if k in wdic:
                self.weight[v, :] = W[wdic[k], :]
        self._set_grad_hook()

    def replace_vectors_path(self, path, keep=None):
        W, D = WordEmb._load_path(path)
        W = torch.tensor(W)
        self.replace_vectors(W, D, keep=keep)

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


class PartiallyPretrainedWordEmb(WordEmb):
    def __init__(self, dim=50, worddic=None, keepvanilla=None, path=None, gradfracs=(1., 1.), **kw):
        """
        :param dim:         embedding dimension
        :param worddic:     which words to create embeddings for, must map from strings to ids
        :param keepvanilla: set of words which will be kept in the vanilla set of vectors
                            even if they occur in pretrained embeddings
        :param path:        where to load pretrained word from
        :param gradfracs:   tuple (vanilla_frac, pretrained_frac)
        :param kw:
        """
        super(PartiallyPretrainedWordEmb, self).__init__(dim=dim, worddic=worddic, **kw)
        path = self._get_path(dim, path=path)
        value, wdic = self.loadvalue(path, dim, indim=None,
                                     worddic=None, maskid=None,
                                     rareid=None)
        value = torch.tensor(value)
        self.mixmask = q.val(np.zeros((len(self.D),), dtype="float32")).v

        for k, v in self.D.items():
            if k in wdic and (keepvanilla is None or k not in keepvanilla):
                self.weight[v, :] = value[wdic[k], :]
                self.mixmask[v] = 1

        # self.weight = torch.nn.Parameter(self.weight)

        self.gradfrac_vanilla, self.gradfrac_pretrained = gradfracs

        def apply_gradfrac(grad):
            if self.gradfrac_vanilla != 1.:
                grad = grad * ((1 - self.mixmask.unsqueeze(1)) * q.v(self.gradfrac_vanilla)
                               + self.mixmask.unsqueeze(1))
            if self.gradfrac_pretrained != 1.:
                grad = grad * (self.mixmask.unsqueeze(1) * q.v(self.gradfrac_pretrained)
                               + (1 - self.mixmask.unsqueeze(1)))
            return grad

        self.weight.register_hook(apply_gradfrac)