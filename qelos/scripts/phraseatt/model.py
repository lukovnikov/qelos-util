import torch
import qelos as q
import numpy as np


__all__ = ["AttComp", "SummComp", "Attention", "DotAttComp", "SumSummComp"]


# region normal attention
class AttComp(torch.nn.Module):
    def forward(self, qry, ctx, ctx_mask=None):
        raise NotImplemented()


class SummComp(torch.nn.Module):
    def forward(self, values, alphas):
        raise NotImplemented()


class Attention(torch.nn.Module):
    """ Computes phrase attention. For use with encoders and decoders from rnn.py """
    def __init__(self, attcomp:AttComp=None, summcomp:SummComp=None, score_norm=None):
        """
        :param attcomp:     used to compute attention scores
        :param summcomp:    used to compute summary
        """
        super(Attention, self).__init__()
        self.prevatts = None    # holds previous attention vectors
        self.prevatt_ptr = None     # for every example, contains a list with pointers to indexes of prevatts
        self.attcomp = attcomp if attcomp is not None else DotAttComp()
        self.summcomp = summcomp if summcomp is not None else SumSummComp()
        self.score_norm = torch.nn.Softmax(-1) if score_norm is None else score_norm

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


class SumSummComp(SummComp):
    def forward(self, values, alphas):
        summary = values * alphas.unsqueeze(2)
        summary = summary.sum(1)
        return summary
# endregion


# region Phrase Attention

# function with a custom backward for getting gradients to both parent and children in PhraseAttention
# forward is an elementwise min
# backward:     - alphas always gets whole gradient
#               - parent_alphas is increased when gradient > 0 else nothing
class ParentOverlapFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parent_alphas, alphas):
        ctx.save_for_backward(parent_alphas, alphas)
        ret = torch.min(parent_alphas, alphas)
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        gradzeros = torch.zeros_like(grad_output)
        parent_grads = torch.max(gradzeros, grad_output)
        return parent_grads, grad_output


parent_overlap_f = ParentOverlapFunction.apply


def test_custom_f(lr=0):
    x = torch.randn(5)
    x.requires_grad = True
    y = torch.randn(5)
    y.requires_grad = True
    z = parent_overlap_f(x, y)
    l = z #z.sum()
    l.backward(gradient=torch.tensor([-1,1,-1,1,1]).float())
    print(x.grad)
    print(y.grad)


class PhraseAttention(Attention):       # for depth-first decoding
    def __init__(self, attcomp:AttComp=None, summcomp:SummComp=None, score_norm=None):
        super(PhraseAttention, self).__init__(attcomp=attcomp, summcomp=summcomp, score_norm=score_norm)
        self.prevatts = None            # (batsize, declen_so_far, enclen)
        self.prevatt_ptr = None         # for every example, keeps a list of pointers to positions in prevatts
        self.prevatt_siblings = None    # for every example, keeps a list of sets of pointers to groups of siblings

    def batch_reset(self):
        self.prevatts, self.prevatt_ptr = None, None
        self.prevatt_siblings = None

    def get_sibling_overlap(self):
        """
        Gets overlap in siblings based on current state of prevatts and prevatt_ptr.
        Must be called after a batch and before batch reset.
        """
        ids = torch.zeros(self.prevatts.size(0), self.prevatts.size(1), self.prevatts.size(1),
                          dtype=torch.long, device=self.prevatts.device) - 1
        maxnumsiblingses, maxnumsiblings = 0, 0
        for eid, siblingses in enumerate(self.prevatt_siblings):    # list of lists of ids in prevatts
            maxnumsiblingses = max(maxnumsiblingses, len(siblingses))
            for sgid, siblings in enumerate(siblingses):             # list of ids in prevatts
                maxnumsiblings = max(maxnumsiblings, len(siblings))
                for sid, sibling in enumerate(siblings):
                    ids[eid, sgid, sid] = sibling
        ids = ids[:, :maxnumsiblingses, :maxnumsiblings]

        prependix = torch.zeros(self.prevatts.size(0), 1, self.prevatts.size(2), dtype=self.prevatts.dtype, device=self.prevatts.device) + 1
        prevatts = torch.cat([prependix, self.prevatts], 1)
        ids = ids + 1

        idsmask= ((ids != 0).sum(2, keepdim=True) != 0).float()

        _ids = ids.view(ids.size(0), -1).unsqueeze(-1).repeat(1, 1, prevatts.size(2))
        prevatts_gathered = torch.gather(prevatts, 1, _ids)
        prevatts_gathered = prevatts_gathered.view(prevatts.size(0), ids.size(1), ids.size(2), prevatts.size(2))

        overlaps = prevatts_gathered.prod(2)
        overlaps = overlaps * idsmask
        overlaps = overlaps.sum(2).sum(1)
        overlaps = overlaps.mean(0)
        return overlaps

    def forward(self, qry, ctx, ctx_mask=None, values=None, pushpop=None):
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
        # constrain alphas to parent's alphas:
        if self.prevatts is not None:   # there is history
            parent_ptr = [prevatt_ptr_e[-2][-1] for prevatt_ptr_e in self.prevatt_ptr]
            parent_ptr = torch.tensor(parent_ptr).long().to(self.prevatts.device)
            parent_alphas = self.prevatts.index_select(1, parent_ptr)
            alphas = parent_overlap_f(parent_alphas, alphas)

        # compute summary
        values = ctx if values is None else values
        summary = self.summcomp(values, alphas)

        # store alphas and update ptrs
        if self.prevatts is not None:   # there is history --> append
            self.prevatts = torch.cat([self.prevatts, alphas.unsqueeze(1)], 1)
        else:
            self.prevatts = alphas.unsqueeze(1)

        if self.prevatt_ptr is None:    # then initialize
            self.prevatt_ptr = [[[]] for _ in range(len(pushpop))]
            self.prevatt_siblings = [[] for _ in range(len(pushpop))]

        k = self.prevatts.size(1) - 1
        for i in range(len(pushpop)):
            self.prevatt_ptr[i][-1].append(k)
            if pushpop[i] > 0:  # PUSH (new parent)
                self.prevatt_ptr[i].append([])
            elif pushpop[i] < 0:    # POP (siblings finished)
                siblings = self.prevatt_ptr[i].pop()
                self.prevatt_siblings.append(siblings)
            else:
                pass
        return alphas, summary, scores


def test_phrase_attention_sibling_overlap(lr=0):
    m = PhraseAttention()
    batsize, seqlen = 2, 6
    allatts = torch.rand(batsize, seqlen, seqlen)
    allatts.requires_grad = True
    m.prevatts = allatts
    m.prevatt_siblings = [[[0, 1], [2, 3, 4, 5]],
                          [[1, 2], [3, 4], [0, 5]]]
    l = m.get_sibling_overlap()
    l.backward()
    print(allatts)
    print(allatts.grad)




# endregion

if __name__ == '__main__':
    # q.argprun(test_custom_f)
    q.argprun(test_phrase_attention_sibling_overlap)