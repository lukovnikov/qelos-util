import torch
import qelos as q
from torch import nn
import numpy as np
import re
import math
from nltk.translate.bleu_score import sentence_bleu
import warnings

EPS = 1e-6


__all__ = ["Accuracy", "SeqAccuracy", "SeqElemAccuracy", "MacroBLEU"]


def logsumexp(x, axis=-1):
    xmax, _ = torch.max(x, axis, keepdim=True)
    _x = x - xmax
    _x = torch.exp(_x)
    _x = torch.sum(_x, axis, keepdim=True)
    lse = xmax + torch.log(_x)
    return lse.squeeze(-1)


def nan2zero(x):
    nanmask = torch.isnan(x)
    if torch.any(nanmask).item() == 1:
        _x_cpy = torch.zeros_like(x)
        _xv = x.masked_select(~nanmask)
        _x_cpy.masked_scatter_(~nanmask, _xv)
        return _x_cpy
    return x


def inf2zero(x):
    infmask = x == -np.inf
    if torch.any(infmask).item() == 1:
        _x_cpy = torch.zeros_like(x)
        _xv = x.masked_select(~infmask)
        _x_cpy.masked_scatter_(~infmask, _xv)
        return _x_cpy
    return x


class Loss(nn.Module):
    def __init__(self, size_average=True, **kw):
        super(Loss, self).__init__(**kw)
        self.size_average = size_average

    def forward(self, x, gold, mask=None, _noagg=False, **kw):
        y, ignoremask = self._forward(x, gold, mask=mask, **kw)
        y = y.float()
        if _noagg:
            return y, ignoremask

        if ignoremask is not None:
            y = y * ignoremask.float().clamp(0, 1)      # ensure ignoremask is not higher than 1
        else:
            total = y.size(0)

        loss = y.sum()
        if self.size_average:
            loss /= total
        return loss


class DiscreteLoss(Loss):
    """ Loss with ignore_index(es), provides default implementation of _get_ignore_mask """
    def __init__(self, size_average=True, ignore_index=None, **kw):
        super(DiscreteLoss, self).__init__(size_average=size_average, **kw)
        if ignore_index is not None:
            if not q.issequence(ignore_index):
                self.ignore_indices = [ignore_index]
        else:
            self.ignore_indices = None

    def _get_ignore_mask(self, gold):
        mask = None     # (batsize,)
        if self.ignore_indices is not None:
            for ignore in self.ignore_indices:
                mask_i = (gold != ignore)   # zero for ignored ones
                if mask is None:
                    mask = mask_i
                else:
                    mask = mask & mask_i
        return mask


class Accuracy(DiscreteLoss):
    def _forward(self, x, gold):
        ignoremask = self._get_ignore_mask(gold)
        maxes, best = torch.max(x, 1)
        same = best == gold
        if ignoremask is not None:
            same = same | ~ ignoremask
        return same.float(), ignoremask


class SeqAccuracy(DiscreteLoss):
    """ very basic explicit seqaccuracy implementation.
        does not support batchable sparse mask """
    def _forward(self, x, gold):
        """
        :param x: (batsize, seqlen, vocsize) - probabilities over output symbols for every time step
        :param gold: (batsize, seqlen) - ids of gold output symbols at every time step
        :return: loss value, ignormask
        """
        ignoremask = self._get_ignore_mask(gold)
        _, best = torch.max(x, 2)       # (batsize, seqlen) - most probable symbols at every time step
        same = best == gold
        outignoremask = None
        if ignoremask is not None:
            same = same | ~ ignoremask   # set ignored portions to be same[i,j]=True
            outignoremask = ignoremask.long().sum(1) > 0
        sameseqs = same.long().sum(1)
        sameseqs = sameseqs == int(same.size(1))
        return sameseqs, outignoremask


class SeqElemAccuracy(DiscreteLoss):    # TODO take end of sequence token into account
    def forward(self, x, gold):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
        diff = argmaxes == gold
        if ignoremask is not None:
            diff = diff * ignoremask
            total = torch.sum(ignoremask.long()).item()
        else:
            total = gold.size(0) * gold.size(1)
        acc = torch.sum(diff.float())
        if self.size_average:
            acc = acc / total
        return acc, total


class MacroBLEU(DiscreteLoss):      # TODO take end of sequence token into account
    """ macro-averaged BLEU over sequences """
    def __init__(self, order=4, predcut=None, ignore_index=None, **kw):
        """
        :param order:           n-gram order of BLEU
        :param predcut:         function to cut prediction. Gets the argmax over prediction and ignore_index kwarg.
                                Must fill all elements after end of sequence with provided ignore_index
        """
        super(MacroBLEU, self).__init__(ignore_index=ignore_index, **kw)
        self.order = order
        self.weights = tuple([1. / self.order for _ in range(self.order)])
        self.predcut = predcut
        warnings.filterwarnings("ignore", module="nltk")

    def forward(self, x, gold, mask=None):
        if x.size(1) > gold.size(1):
            x = x[:, :gold.size(1)]
        ignoremask = self._get_ignore_mask(gold)
        maxes, argmaxes = torch.max(x, dim=2)
        ignore_id = None
        if self.ignore_indices is not None:
            ignore_id = self.ignore_indices[0]
        argmaxes = argmaxes.cpu()
        if self.predcut is not None:
            argmaxes = self.predcut(argmaxes, ignore_index=ignore_id)
        gold = gold.cpu()
        bleus = 0.
        for i in range(gold.size(0)):
            predseq = [str(a) for a in list(argmaxes[i]) if a != ignore_id]
            goldseq = [str(a) for a in list(gold[i]) if a not in self.ignore_indices]
            bleu = sentence_bleu([goldseq], predseq, weights=self.weights)
            bleus += bleu

        total = gold.size(0)
        if self.size_average:
            bleus = bleus / total
        return bleus, total

# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

# region NEW LOSSES
class SeqKLLoss(DiscreteLoss):
    """ Straight implementation of cross-entropy loss for sequence prediction.
        Same as Sequence cross-entropy if no label smoothing.
        To be used after torch.nn.Softmax() """

    def __init__(self, time_average=True, time_agg=None, weight=None, size_average=True,
                 ignore_index=None, label_smoothing=0., smooth_mix=0., mode="probs", **kw):
        """

        :param time_agg:        aggregation over time: if "avg", then averages, "sum" sums. Takes priority over time_average
        :param time_average:    averages over time if True. Default False.
        :param weight:          ?
        :param size_average:    average over batch (True) or sum (False)
        :param ignore_index:    which tokens in gold to ignore (mask)
        :param label_smoothing: how much uniform label smoothing to perform (between 0 and 1) to get target distribution
        :param smooth_mix:      how much to mix predictive distribution with target distribution
        :param mode:            "probs" (probs must be normalized by Softmax()), "logits" (probs are logits), "logprobs" (probs are log probs, produced by LogSoftmax())
        :param kw:
        """
        super(SeqKLLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        if time_agg is None:
            time_agg = "avg" if time_average else "sum"
        assert (time_agg in "sum avg".split())
        self.time_agg = time_agg
        self.label_smoothing = label_smoothing
        self.smooth_mix = smooth_mix
        self.mode = mode

    def _forward(self, probs, gold, mask=None):
        if q.v(self.label_smoothing) > 0. or q.v(self.smooth_mix) > 0.:
            return self._forward_smooth(probs, gold, mask=mask)
        else:
            return self._forward_normal(probs, gold, mask=mask)

    def _forward_smooth(self, probs, gold, mask=None):
        if self.mode != "probs":
            raise NotImplemented("'logits' and 'logprobs' mode not implemented with softened targets (TODO)")

        if probs.size(1) > gold.size(1):
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()

        ignoremask = self._get_ignore_mask(gold)  # whether to ignore a certain time step of a certain example
        outignoremask = None

        if mask is not None:
            probs = probs * mask

        prob_mask = (probs > 0).float()  # (batsize, seqlen, vocsize)
        if isinstance(q.v(self.label_smoothing), float):
            lsv = q.v(self.label_smoothing)
            assert (lsv >= 0 and lsv <= 1)
            prob_mask_weights = lsv / prob_mask.sum(2)
            _gold = torch.ones_like(probs) * prob_mask_weights.unsqueeze(2) * prob_mask  # masked uniform
            _gold.scatter_(2, gold.unsqueeze(2), (1 - lsv) + prob_mask_weights.unsqueeze(2))
        else:
            _gold = self.label_smoothing(gold, prob_mask)

        if q.v(self.smooth_mix) > 0.:
            smv = q.v(self.smooth_mix)
            _gold = _gold * (1 - smv) + smv * probs.detach()

        assert (np.allclose(_gold.sum(2).cpu().detach().numpy(),
                            np.ones((_gold.size(0), _gold.size(1))), atol=1e-3))

        log_probs = - (torch.log(probs + (1 - prob_mask)) - torch.log(_gold + (1 - prob_mask)))
        # REMARK: (1 - prob_mask) is added before log() to ensure that no -inf's are there
        kls = log_probs * _gold
        kls = kls * prob_mask  # prob can be removed
        gold_log_probs = kls.sum(2)

        seqlens = torch.tensor(seqlen).float().to(gold.device)

        if ignoremask is not None:
            gold_log_probs = gold_log_probs * ignoremask.float()  # should work because normal softmax was used --> no infs
            seqlens = ignoremask.float().sum(1)
            outignoremask = ignoremask.long().sum(1) > 0

        gold_log_probs = gold_log_probs.sum(1)
        if self.time_agg == "avg":
            gold_log_probs = gold_log_probs / seqlens.clamp(min=EPS)

        return gold_log_probs, outignoremask

    def _forward_normal(self, probs, gold, mask=None):
        if probs.size(1) > gold.size(1):  # if probs is longer than gold seq
            probs = probs[:, :gold.size(1)]
        batsize, seqlen, vocsize = probs.size()

        ignoremask = self._get_ignore_mask(gold)
        outignoremask = None

        if mask is not None:
            if self.mode == "probs":
                probs = probs * mask
            elif self.mode == "logits":
                probs = probs + torch.log(mask.float())
            elif self.mode == "logprobs":
                raise NotImplemented("mask in logprobs not implemented")

        gold_probs = probs.gather(2, gold.unsqueeze(2))
        assert (gold_probs.size(2) == 1)
        gold_probs = gold_probs.squeeze(2)

        if self.mode == "probs":
            gold_log_probs = - torch.log(gold_probs.clamp(min=1e-9))
        elif self.mode == "logprobs":
            gold_log_probs = - gold_probs
        elif self.mode == "logits":
            gold_log_probs = - gold_probs + logsumexp(probs)

        seqlens = torch.tensor(seqlen).float().to(gold.device)

        if ignoremask is not None:
            gold_log_probs = gold_log_probs * ignoremask.float()  # should work because normal softmax was used --> no infs
            seqlens = ignoremask.float().sum(1)
            outignoremask = ignoremask.long().sum(1) > 0

        gold_log_probs = gold_log_probs.sum(1)
        if self.time_agg == "avg":
            gold_log_probs = gold_log_probs / seqlens.clamp(min=EPS)

        return gold_log_probs, outignoremask


class KLLoss(SeqKLLoss):
    def __init__(self, weight=None, size_average=True, ignore_index=None, label_smoothing=0., smooth_mix=0.,
                 mode="probs", **kw):
        super(KLLoss, self).__init__(weight=weight, size_average=size_average, ignore_index=ignore_index,
                                     label_smoothing=label_smoothing, smooth_mix=smooth_mix, mode=mode, **kw)

    def _forward(self, probs, gold, mask=None):  # (batsize, numsym), (batsize,) ints, (batsize, numsym) bool
        # insert a seq dimension
        probs = probs.unsqueeze(1)
        gold = gold.unsqueeze(1)
        if mask is not None:
            mask = mask.unsqueeze(1)
        logprobs, ignormask = super(KLLoss, self)._forward(probs, gold, mask=mask)
        return logprobs, ignormask


class SeqPPLLoss(SeqKLLoss):
    def post_agg_epoch(self, x):
        """ function applied after aggregation (avg/sum) over whole epoch (so far) """
        return math.exp(x)


class PPLLoss(KLLoss):
    def post_agg_epoch(self, x):
        return math.exp(x)


class SeqDistillLoss(DiscreteLoss):
    """ Distillation (KD) loss for sequences of categorical distributions """
    def __init__(self, time_average=True, size_average=True, ignore_index=None,
                 temperature=1., mixture=0.5, soft_gold_mode="logits", **kw):
        """
        :param time_average:    average over time
        :param size_average:    average over batches
        :param ignore_index:    gold ids whose time steps will be ignored
        :param temperature:     softmax temperature (!: will not be applied if soft_gold_mode is not "logits")
        :param mixture:         mixing portion of soft and hard gold
        :param soft_gold_mode:  "logits", "logprobs", "probs": how the softgold is normalized. If not "logits", temperatur will not be applied
        :param kw:
        """
        super(SeqDistillLoss, self).__init__(size_average=size_average, ignore_index=ignore_index, **kw)
        self.time_average = time_average
        self.temperature = temperature
        self.mixture = mixture
        self.soft_gold_mode = soft_gold_mode
        self.sm = torch.nn.Softmax(-1)
        self.logsm = torch.nn.LogSoftmax(-1)
        if soft_gold_mode != "logits":
            print("WARNING: temperature is not applied to soft gold in soft_gold_mode {}".format(soft_gold_mode))

    def _forward(self, probs, gold, mask=None):
        """
        :param probs:       (batsize, seqlen, numsym) prediction vector of logits
        :param gold:        tuple of (batsize, seqlen, numsym) soft gold and (batsize, seqlen) ints for hard gold
        """
        assert(mask is None)    # masks should not be supported here
        softgold, hardgold = gold
        t = q.v(self.temperature)
        mix = q.v(self.mixture)

        if probs.size(1) > softgold.size(1):
            probs = probs[:, :softgold.size(1)]
        batsize, seqlen, vocsize = probs.size()

        ignoremask = self._get_ignore_mask(hardgold)
        outignoremask = None

        # hard gold - normal CE, !!: probs are logits
        hard_gold_log_probs = 0
        if mix < 1:
            hard_gold_probs = probs.gather(2, hardgold.unsqueeze(2))
            assert(hard_gold_probs.size(2) == 1)
            hard_gold_probs = hard_gold_probs.squeeze(2)
            hard_gold_log_probs = - hard_gold_probs + logsumexp(probs)      # !!! no temperature necessary for hard gold

        # soft gold
        kl = 0
        if mix > 0:
            if self.soft_gold_mode in ("logits", "logprobs"):       # --> using LogSoftmax
                _log_probs = self.logsm(probs / t)
                if self.soft_gold_mode == "logits":
                    if torch.any(probs == -np.infty).item() == 1:
                        softgold = softgold + torch.log((probs != -np.infty).float())
                    _log_softgold = self.logsm(softgold / t)
                else:
                    _log_softgold = softgold    # provided softgold was logprobs
                _softgold = torch.exp(_log_softgold)
                # assert(torch.all((_softgold.sum(-1) > 0.9999) & (_softgold.sum(-1) < 1.0001)).item() == 1)
                kls = _softgold * (_log_softgold - _log_probs)        # KL
            else:
                _log_probs = self.logsm(probs / t)
                kls = softgold * (torch.log(softgold) - _log_probs)
            kls = nan2zero(kls)
            kls = inf2zero(kls)
            kl = kls.sum(2)

        loss = mix * kl + (1 - mix) * hard_gold_log_probs        # (batsize, seqlen)

        seqlens = torch.tensor(seqlen).float().to(softgold.device)

        if ignoremask is not None:
            loss = loss * ignoremask.float()
            loss = nan2zero(loss)
            seqlens = ignoremask.float().sum(1)
            outignoremask = ignoremask.long().sum(1) > 0

        _loss = loss.sum(1)
        if self.time_average is True:
            _loss = _loss / seqlens.clamp(min=EPS)

        return _loss, outignoremask

# endregion