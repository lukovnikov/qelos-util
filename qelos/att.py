import torch
import qelos as q
import numpy as np
import math


# region normal attention
class AttComp(torch.nn.Module):
    """ computes attention scores """
    def forward(self, qry, ctx, ctx_mask=None):
        raise NotImplemented()


class SummComp(torch.nn.Module):
    def forward(self, values, alphas):
        raise NotImplemented()


class Attention(torch.nn.Module):
    """ Computes phrase attention. For use with encoders and decoders from rnn.py """
    def __init__(self, attcomp:AttComp=None, summcomp:SummComp=None, score_norm=torch.nn.Softmax(-1)):
        """
        :param attcomp:     used to compute attention scores
        :param summcomp:    used to compute summary
        """
        super(Attention, self).__init__()
        # self.prevatts = None    # holds previous attention vectors
        # self.prevatt_ptr = None     # for every example, contains a list with pointers to indexes of prevatts
        self.attcomp = attcomp if attcomp is not None else DotAttComp()
        self.summcomp = summcomp if summcomp is not None else SumSummComp()
        self.score_norm = score_norm

    def forward(self, qry, ctx, ctx_mask=None, values=None):
        """
        :param qry:     (batsize, dim)
        :param ctx:     (batsize, seqlen, dim)
        :param ctx_mask: (batsize, seqlen)
        :param values:  (batsize, seqlen, dim)
        :return:
        """
        scores = self.attcomp(qry, ctx, ctx_mask=ctx_mask)
        scores = scores + (torch.log(ctx_mask.float()) if ctx_mask is not None else 0)
        alphas = self.score_norm(scores)
        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)
        return alphas, summary, scores


class DotAttComp(AttComp):
    def forward(self, qry, ctx, ctx_mask=None):
        """
        :param qry:         (batsize, dim) or (batsize, zeqlen, dim)
        :param ctx:         (batsize, seqlen, dim)
        :param ctx_mask:
        :return:
        """
        if qry.dim() == 2:
            ret = torch.einsum("bd,bsd->bs", [qry, ctx])
        elif qry.dim() == 3:
            ret = torch.einsum("bzd,bsd->bzs", [qry, ctx])
        else:
            raise q.SumTingWongException("qry has unsupported dimension: {}".format(qry.dim()))
        return ret


class SimpleFwdAttComp(AttComp):
    def __init__(self, qrydim=None, ctxdim=None, encdim=None, **kw):
        super(SimpleFwdAttComp, self).__init__(**kw)
        self.qrylin = torch.nn.Linear(qrydim, encdim, bias=False)
        self.ctxlin = torch.nn.Linear(ctxdim, encdim, bias=False)
        self.summ = torch.nn.Linear(encdim, bias=False)

    def forward(self, qry, ctx, ctx_mask=None):
        qryout = self.qrylin(qry)   # (batsize, encdim)
        ctxout = self.ctxlin(ctx)   # (batsize, seqlen, encdim)
        enc = qryout.unsqueeze(1) + ctxout
        out = self.summ(enc).squeeze(-1)
        return out


class FwdAttComp(AttComp):
    def __init__(self, qrydim=None, ctxdim=None, encdim=None, numlayers=1, dropout=0, **kw):
        super(FwdAttComp, self).__init__(**kw)
        layers = [torch.nn.Linear(qrydim + ctxdim, encdim)] \
                 + [torch.nn.Linear(encdim, encdim) for _ in range(numlayers - 1)]
        acts = [torch.nn.Tanh() for _ in range(len(layers))]
        layers = [a for b in zip(layers, acts) for a in b]
        layers.append(torch.nn.Dropout(dropout))
        layers.append(torch.nn.Linear(encdim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, qry, ctx, ctx_mask=None):
        """
        :param qry:     (batsize, qrydim)
        :param ctx:     (batsize, seqlen, ctxdim)
        :param ctx_mask:    (batsize, seqlen)
        :return:
        """
        inp = torch.cat([ctx, qry.unsqueeze(1).repeat(1, ctx.size(1), 1)], 2)
        out = self.mlp(inp)
        ret = out.squeeze(-1)
        return ret


class SumSummComp(SummComp):
    def forward(self, values, alphas):
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return summary


class BasicAttention(Attention):
    def __init__(self, **kw):
        attcomp = DotAttComp()
        summcomp = SumSummComp()
        super(BasicAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, **kw)


class FwdAttention(Attention):
    def __init__(self, qrydim, ctxdim, encdim, dropout=0., **kw):
        attcomp = FwdAttComp(qrydim=qrydim, ctxdim=ctxdim, encdim=encdim, dropout=dropout)
        summcomp = SumSummComp()
        super(FwdAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, **kw)


# endregion
