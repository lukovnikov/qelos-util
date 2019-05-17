import torch
import qelos as q
import numpy as np
import re
from abc import ABC, abstractproperty


__all__ = ["RNNCell", "GRUCell", "LSTMCell", "TFDecoder", "FreeDecoder", "ThinDecoder", "BeamDecoder", "LuongCell", "BahdanauCell",
           "DecoderCell", "RNNEncoder", "GRUEncoder", "LSTMEncoder", "RNNCellEncoder", "GRUCellEncoder",
           "LSTMCellEncoder", "DecCellMerge", "ConcatDecCellMerge", "FwdDecCellMerge"]


# region recurrent cells
class RecCell(torch.nn.Module, q.basic.Stateful):
    celltype = None
    statevars = ["h_tm1", "dropout_rec", "dropout_in"]
    def __init__(self, indim, outdim, bias=True, dropout_in=0., dropout_rec=0., **kw):
        super(RecCell, self).__init__(**kw)
        self.indim, self.outdim, self.bias = indim, outdim, bias

        self.cell = self.celltype(self.indim, self.outdim, bias=self.bias)

        # dropouts etc
        self.dropout_in, self.dropout_rec, = None, None
        if dropout_in > 0.:
            self.dropout_in = q.RecDropout(p=dropout_in)
        if dropout_rec > 0.:
            self.dropout_rec = q.RecDropout(p=dropout_rec)
        assert(isinstance(self.dropout_in, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec, (torch.nn.Dropout, q.RecDropout, type(None))))
        self.h_tm1 = None
        self.register_buffer("h_0", torch.zeros(1, self.outdim))
        self.reset_parameters()

    def batch_reset(self):
        self.h_tm1 = None

    def reset_parameters(self):
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def apply_mask_t(self, *statepairs, **kw):
        """ interpolates between previous and new state inside a timestep in a batch based on mask"""
        mask_t = q.getkw(kw, "mask_t", None)
        if mask_t is not None:
            mask_t = mask_t.float().unsqueeze(1)
            ret = [h_t * mask_t + h_tm1 * (1 - mask_t) for h_tm1, h_t in statepairs]
            return tuple(ret)
        else:
            return tuple([statepair[1] for statepair in statepairs])

    def forward(self, x_t, mask_t=None, **kw):
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous state
        h_tm1 = self.h_0.expand(batsize, -1) if self.h_tm1 is None else self.h_tm1
        h_tm1 = self.dropout_rec(h_tm1) if self.dropout_rec is not None else h_tm1

        h_t = self.cell(x_t, h_tm1)

        # next state
        h_t, = self.apply_mask_t((h_tm1, h_t), mask_t=mask_t)
        self.h_tm1 = h_t
        return h_t


class RNNCell(RecCell):
    celltype = torch.nn.RNNCell


class GRUCell(RecCell):
    celltype = torch.nn.GRUCell


class LSTMCell(RecCell):
    celltype = torch.nn.LSTMCell
    statevars = ["y_tm1", "c_tm1", "dropout_in", "dropout_rec", "dropout_rec_c"]
    def __init__(self, indim, outdim, bias=True, dropout_in=0., dropout_rec=0., **kw):
        super(RecCell, self).__init__(**kw)
        self.indim, self.outdim, self.bias = indim, outdim, bias

        self.cell = self.celltype(self.indim, self.outdim, bias=self.bias)

        # dropouts etc
        self.dropout_in, self.dropout_rec, self.dropout_rec_c = None, None, None
        if dropout_in > 0.:
            self.dropout_in = q.RecDropout(p=dropout_in)
        if dropout_rec > 0.:
            self.dropout_rec = q.RecDropout(p=dropout_rec)
            self.dropout_rec_c = q.RecDropout(p=dropout_rec)
        assert(isinstance(self.dropout_in, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec, (torch.nn.Dropout, q.RecDropout, type(None))))
        assert(isinstance(self.dropout_rec_c, (torch.nn.Dropout, q.RecDropout, type(None))))
        self.register_buffer("y_0", torch.zeros(1, self.outdim))
        self.register_buffer("c_0", torch.zeros(1, self.outdim))
        self.y_tm1 = None
        self.c_tm1 = None
        self.reset_parameters()

    def batch_reset(self):
        self.y_tm1 = None
        self.c_tm1 = None

    def forward(self, x_t, mask_t=None, **kw):
        batsize = x_t.size(0)
        x_t = self.dropout_in(x_t) if self.dropout_in else x_t

        # previous states
        y_tm1 = self.y_0.expand(batsize, -1) if self.y_tm1 is None else self.y_tm1
        c_tm1 = self.c_0.expand(batsize, -1) if self.c_tm1 is None else self.c_tm1
        y_tm1 = self.dropout_rec(y_tm1) if self.dropout_rec is not None else y_tm1
        c_tm1 = self.dropout_rec_c(c_tm1) if self.dropout_rec_c is not None else c_tm1

        y_t, c_t = self.cell(x_t, (y_tm1, c_tm1))

        # next state
        y_t, c_t = self.apply_mask_t((y_tm1, y_t), (c_tm1, c_t), mask_t=mask_t)
        self.y_tm1, self.c_tm1 = y_t, c_t
        return y_t
# endregion


# region decoder
class Decoder(torch.nn.Module):
    """ abstract decoder """
    def __init__(self, cell, **kw):
        """
        :param cell:    must produce probabilities as first output
        :param kw:
        """
        super(Decoder, self).__init__(**kw)
        self.cell = cell

    def forward(self, xs, **kw):
        """
        :param xs:      argument(s) that will be time-sliced
        :param kw:      are passed to decoder cell (unless used in decoder itself)
        :return:
        """
        raise NotImplemented("use subclass")


class ThinDecoder(Decoder):
    """
    Thin decoder, cells have full control and decoder only provides time steps and merges outputs.
    Cell must implement:
        - forward(t, *args, **kw) -- cell must save all necessary outputs by itself, inputs are forwarded from decoder
        - stop() to terminate decoding
    """
    def forward(self, *args, **kw):
        q.batch_reset(self.cell)
        outs = []
        out_is_seq = False
        for t in range(10e9):
            outs_t = self.cell(t, *args, **kw)
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
            if self.cell.stop():
                break
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs


class TFDecoder(Decoder):
    def forward(self, xs, **kw):
        # q.batch_reset(self.cell)
        x_is_seq = True
        if not q.issequence(xs):
            x_is_seq = False
            xs = (xs,)
        outs = []
        out_is_seq = False
        for t in range(xs[0].size(1)):
            x_ts = tuple([x[:, t] for x in xs])
            x_ts = x_ts[0] if not x_is_seq else x_ts
            outs_t = self.cell(x_ts, **kw)
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs


class FreeDecoder(Decoder):
    def __init__(self, cell, maxtime=None, **kw):
        super(FreeDecoder, self).__init__(cell, **kw)
        self.maxtime = maxtime

    def forward(self, xs, maxtime=None, **kw):
        """
        :param xs:      typically (batsize, ) int ids but can be anything
        :param maxtime:
        :param kw:      are passed directly into cell at every time step
        :return:
        """
        # q.batch_reset(self.cell)
        maxtime = maxtime if maxtime is not None else self.maxtime
        outs = []
        x_ts = xs
        out_is_seq = False
        for t in range(maxtime):
            outs_t = self.cell(x_ts, **kw)
            x_ts = self._get_xs_from_ys(outs_t)      # --> get inputs from previous outputs
            if q.issequence(outs_t):
                out_is_seq = True
            else:
                outs_t = (outs_t,)
            outs.append(outs_t)
        outs = zip(*outs)
        outs = tuple([torch.cat([a_i.unsqueeze(1) for a_i in a], 1) for a in outs])
        outs = outs[0] if not out_is_seq else outs
        return outs

    def _get_xs_from_ys(self, ys):
        if hasattr(self.cell, "get_xs_from_ys"):
            xs = self.cell.get_xs_from_ys(ys)
        else:
            xs = self.get_xs_from_ys(ys)
        return xs

    def get_xs_from_ys(self, ys):       # default argmax implementation with out output
        assert(not q.issequence(ys))
        assert(ys.dim() == 2)   # (batsize, outsyms)
        _, argmax_ys = torch.max(ys, 1)
        xs = argmax_ys
        return xs


class BeamDecoder(Decoder):
    def __init__(self, cell, beamsize=1, maxtime=None, **kw):
        super(BeamDecoder, self).__init__(cell, **kw)
        self.beamsize = beamsize
        self.maxtime = maxtime

    def forward(self, xs, maxtime=None, **kw):
        """
        :param xs:      typically (batsize, ) int ids but can be anything
        :param maxtime:
        :param kw:      are passed directly into cell at every time step
        :return:
        """
        # q.batch_reset(self.cell)
        maxtime = maxtime if maxtime is not None else self.maxtime
        batsize = xs.size(0) if not q.issequence(xs) else xs[0].size(0)
        device = xs.device if not q.issequence(xs) else xs[0].device
        x_ts = [xs for _ in range(self.beamsize)]                       # beam inputs
        cell_states = q.Stateful.get_state(self.cell)
        cell_states = [cell_states for _ in range(self.beamsize)]       # beam states
        beam_seqs = [[[] for _ in range(batsize)] for _ in range(self.beamsize)]
        beam_scores = torch.zeros(batsize, self.beamsize)
        out_is_seq = False
        for t in range(maxtime+1):
            # 1. advance one timestep using the cell, record outputs and states
            trans_scores = beam_scores.unsqueeze(2).repeat(1, 1, self.beamsize)   # (batsize, from_ray, to_ray)
            saved_x_ts = []
            for j in range(self.beamsize):
                self.cell.set_state(cell_states[j])
                outs_t_j = self.cell(x_ts[j], **kw)
                cell_states[j] = q.Stateful.get_state(self.cell)
                new_x_ts, x_scores = self._get_xs_and_logprobs_from_ys(outs_t_j)
                saved_x_ts.append(new_x_ts)
                trans_scores[:, j, :] += x_scores
            # 2. find new best updates
            _, best_at_t = torch.sort(trans_scores.view(batsize, -1), dim=-1, descending=True)
            best_at_t = best_at_t[:, :self.beamsize]
            best_at_t_from = best_at_t // self.beamsize
            best_at_t_to = best_at_t % self.beamsize
            best_at_t = torch.stack([best_at_t_from, best_at_t_to], 1)
            # ==> best_at_t contains for every example, self.beamsize number of from_ray-to_ray pairs
            #     saved_x_ts: (from ray, batsize, to ray) input tokens
            #     trans_scores: (batsize, from ray, to ray) scores of the tokens, taking into account the ray they were taken from
            # 3. extract new best rays
            newbeams = [[None for _ in range(self.beamsize)] for _ in range(batsize)]
            for i in range(batsize):
                for j in range(self.beamsize):  # goes over new rays
                    best_from, best_to = best_at_t_from[i, j].detach().cpu().item(), best_at_t_to[i, j].detach().cpu().item()
                    sym_ij = saved_x_ts[best_from][i][best_to]      # which symbol to continue with
                    prevseq_ij = beam_seqs[best_from][i]            # the sequence from where to continue
                    seq_ij = prevseq_ij + [sym_ij]
                    score_ij = trans_scores[i, best_from, best_to]  # score of this seq
                    csbf = cell_states[best_from]
                    states_ij = {k: csbf[k][i]
                                    if hasattr(csbf, "__getitem__")
                                    else csbf[k]
                                 for k in csbf}
                    newbeams[i][j] = (sym_ij, seq_ij, score_ij, states_ij)
            # 4. populate variables using new beams
            beam_scores = torch.zeros_like(beam_scores)
            x_ts = [torch.zeros_like(x_ts_j) for x_ts_j in x_ts]
            for i, newbeams_i in enumerate(newbeams):         # per example
                for j, newbeams_ij in enumerate(newbeams_i):  # per ray
                    x_ts[j][i] = newbeams_ij[0]
                    beam_seqs[j][i] = newbeams_ij[1]
                    beam_scores[i, j] = newbeams_ij[2]
                    for k in newbeams_ij[3]:                  # per state component
                        if hasattr(newbeams_ij[3][k], "__setitem__"):
                            cell_states[j][k][i] = newbeams_ij[3][k]
                        else:
                            assert(not hasattr(cell_states[j][k], "__setitem__"))
                            cell_states[j][k] = newbeams_ij[3][k]
        ret = [torch.stack(beam_seqs[0][i], 0) for i in range(batsize)]
        ret = torch.stack(ret, 0)
        return ret

    def _get_xs_and_logprobs_from_ys(self, ys):
        # return torch.arange(ys.size(1)).unsqueeze(0).repeat(ys.size(0), 1), ys
        scores, xs = torch.sort(ys, -1, descending=True)    # --> get inputs from previous outputs
        xs, scores = xs[:, :self.beamsize], scores[:, :self.beamsize]
        return xs, scores

# endregion


# region decoder cells
class BasicDecoderCell(torch.nn.Module):
    def __init__(self, emb, core, out):
        super(BasicDecoderCell, self).__init__()
        self.emb, self.core, self.out = emb, core, out

    def forward(self, x_t, ctx=None, **kw):
        """
        :param x_t:     tensor or list of tensors for this time step
        :param ctx:     inputs for all time steps, forwarded by decoder
        :param kw:      kwargs for all time steps, forwarded by decoder
        :return:
        """
        emb = self.emb(x_t)
        acts = self.core(emb)
        outs = self.out(acts)
        return outs


class DecCellMerge(torch.nn.Module):
    def forward(self, core_out, summary, core_inp):
        """
        :param core_out:    (batsize, decdim) output of core rnn (at time t)
        :param summary:     (batsize, encdim) attention summary (at time t)
        :param core_inp:    (batsize, dim) input to core rnn (at time t)
        :return:
        """
        raise NotImplemented()


class ConcatDecCellMerge(DecCellMerge):    # concat
    def forward(self, core_out, summ, _):
        return torch.cat([core_out, summ], 1)


class FwdDecCellMerge(DecCellMerge):  # !!! UNTESTED TODO: test
    def __init__(self, coredim, encdim, outdim=None, act=torch.nn.Tanh(), **kw):
        super(FwdDecCellMerge, self).__init__(**kw)
        outdim = coredim if outdim is None else outdim
        self.lin = torch.nn.Linear(coredim + encdim, outdim, bias=False)
        self.act = act

    def forward(self, coreout, summ, _):
        z = torch.cat([coreout, summ], 1)
        z = self.lin(z)
        z = self.act(z)
        return z


class LuongCell(torch.nn.Module, q.Stateful):
    statevars = ["_outvec_tm1", "outvec_t0", "_saved_ctx", "_saved_ctx_mask"]
    def __init__(self, emb=None, core=None, att=None, merge:DecCellMerge=ConcatDecCellMerge(),
                 out=None, feed_att=False, return_alphas=False, return_scores=False, return_other=False,
                 dropout=0, **kw):
        """

        :param emb:
        :param core:
        :param att:
        :param merge:
        :param out:         if None, out_vec (after merge) is returned
        :param feed_att:
        :param h_hat_0:
        :param kw:
        """
        super(LuongCell, self).__init__(**kw)
        self.emb, self.core, self.att, self.merge, self.out = emb, core, att, merge, out
        self.feed_att = feed_att
        self._outvec_tm1 = None    # previous attention summary
        self.outvec_t0 = None
        self.return_alphas = return_alphas
        self.return_scores = return_scores
        self.return_other = return_other
        self.dropout = torch.nn.Dropout(dropout)
        self._saved_ctx, self._saved_ctx_mask = None, None

    def save_ctx(self, ctx, ctx_mask=None):
        self._saved_ctx, self._saved_ctx_mask = ctx, ctx_mask

    def batch_reset(self):
        self.outvec_t0 = None
        self._outvec_tm1 = None
        self._saved_ctx = None
        self._saved_ctx_mask = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        if ctx is None:
            ctx, ctx_mask = self._saved_ctx, self._saved_ctx_mask
        assert (ctx is not None)

        if isinstance(self.out, AutoMaskedOut):
            self.out.update(x_t)

        embs = self.emb(x_t)        # embed input tokens
        if q.issequence(embs) and len(embs) == 2:   # unpack if necessary
            embs, mask = embs

        if self.feed_att:
            if self._outvec_tm1 is None:
                assert (self.outvec_t0 is not None)   #"h_hat_0 must be set when feed_att=True"
                self._outvec_tm1 = self.outvec_t0
            core_inp = torch.cat([embs, self._outvec_tm1], 1)     # append previous attention summary
        else:
            core_inp = embs

        core_out = self.core(core_inp)  # feed through rnn

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)  # do attention
        out_vec = self.merge(core_out, summaries, core_inp)
        out_vec = self.dropout(out_vec)
        self._outvec_tm1 = out_vec      # store outvec (this is how Luong, 2015 does it)

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            _out_vec = self.out(out_vec)
            ret += (_out_vec,)

        # other returns
        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret


class DecoderCell(LuongCell):
    pass


class BahdanauCell(torch.nn.Module, q.Stateful):
    """ Almost Bahdanau-style cell, except c_tm1 is fed as input to top decoder layer (core2),
            instead of as part of state """
    statevars = ["summ_0", "_summ_tm1", "_saved_ctx", "_saved_ctx_mask"]
    def __init__(self, emb=None, core1=None, core2=None, att=None, out=None,
                 return_alphas=False, return_scores=False, return_other=False, **kw):
        super(BahdanauCell, self).__init__(**kw)
        self.emb, self.core1, self.core2, self.att, self.out = emb, core1, core2, att, out
        self.summ_0 = None
        self._summ_tm1 = None
        self.return_alphas = return_alphas
        self.return_other = return_other
        self.return_scores = return_scores
        self._saved_ctx, self._saved_ctx_mask = None, None

    def save_ctx(self, ctx, ctx_mask=None):
        self._saved_ctx, self._saved_ctx_mask = ctx, ctx_mask

    def batch_reset(self):
        self.summ_0 = None
        self._summ_tm1 = None
        self._saved_ctx = None
        self._saved_ctx_mask = None

    def forward(self, x_t, ctx=None, ctx_mask=None, **kw):
        if ctx is None:
            ctx, ctx_mask = self._saved_ctx, self._saved_ctx_mask
        assert (ctx is not None)
        print("WARNING: BahdanauCell outdated")

        if isinstance(self.out, AutoMaskedOut):
            self.out.update(x_t)

        embs = self.emb(x_t)
        if q.issequence(embs) and len(embs) == 2:
            embs, mask = embs

        core_inp = embs
        core_out = self.core1(core_inp)

        if self._summ_tm1 is None:
            assert (self.summ_0 is not None)    # "summ_0 must be set"
            self._summ_tm1 = self.summ_0

        core_inp = torch.cat([core_out, self._summ_tm1], 1)
        core_out = self.core2(core_inp)

        alphas, summaries, scores = self.att(core_out, ctx, ctx_mask=ctx_mask, values=ctx)
        self._summ_tm1 = summaries

        out_vec = core_out

        ret = tuple()
        if self.out is None:
            ret += (out_vec,)
        else:
            out_scores = self.out(out_vec)
            ret += (out_scores,)

        if self.return_alphas:
            ret += (alphas,)
        if self.return_scores:
            ret += (scores,)
        if self.return_other:
            ret += (embs, core_out, summaries)
        return ret[0] if len(ret) == 1 else ret
# endregion


# region Encoders
# region RNN layer encoders
# region overridden RNN layers
class RNNLayerOverriddenBase(torch.nn.Module):
    def __init__(self, this, *args, **kwargs):
        super(RNNLayerOverriddenBase, self).__init__(*args, **kwargs)
        self.this = [this]

    @property
    def all_weights(self):
        this = self.this[0]
        acc = []
        for weights in self._all_weights:
            iacc = []
            for weight in weights:
                if hasattr(this, weight) and getattr(this, weight) is not None:
                    iacc.append(getattr(this, weight))
                else:
                    iacc.append(getattr(self, weight))
            acc.append(iacc)
        return acc


class LSTMOverridden(RNNLayerOverriddenBase, torch.nn.LSTM):
    pass


class GRUOverridden(RNNLayerOverriddenBase, torch.nn.GRU):
    pass


class RNNOverridden(RNNLayerOverriddenBase, torch.nn.RNN):
    pass


class OverriddenRNNLayerBase(torch.nn.Module):
    """ Fastest LSTM encoder layer using torch's built-in fast LSTM.
        Provides a more convenient interface.
        States are stored in .y_n and .c_n (initial states in .y_0 and .c_0).
        !!! Dropout_in, dropout_rec are shared among all examples in a batch (and across timesteps) !!!"""
    rnnlayertype = None

    def __init__(self, input_size=None, hidden_size=None, num_layers=1, bidirectional=False,
                 bias=True, batch_first=False, dropout_rec=0., dropconnect=0., **kw):
        super(OverriddenRNNLayerBase, self).__init__(**kw)
        assert(batch_first == True)
        assert(num_layers == 1)
        self.layer = self.rnnlayertype(self, input_size=input_size,
                                       hidden_size=hidden_size, num_layers=num_layers,
                                       bidirectional=bidirectional, bias=bias, batch_first=True)
        self.dropout_rec = torch.nn.Dropout(dropout_rec) if dropout_rec > 0 else None
        self.dropconnect = torch.nn.Dropout(dropconnect) if dropconnect > 0 else None
        self.reset_parameters()

        # weight placeholders
        for weights in self.layer._all_weights:
            for weight in weights:
                setattr(self, weight, None)

    def reset_parameters(self):
        for t in [param for name, param in self.layer.named_parameters() if "weight_ih" in name]:
            torch.nn.init.xavier_uniform_(t)
        for t in [param for name, param in self.layer.named_parameters() if "weight_hh" in name]:
            torch.nn.init.orthogonal_(t)
        for t in [param for name, param in self.layer.named_parameters() if "bias" in name]:
            torch.nn.init.constant_(t, 0)

    def forward(self, vecs, h_0=None):
        if self.dropout_rec is not None:
            weights = ["weight_hh_l0", "weight_hh_l0_reverse"]
            weights = [x for x in weights if hasattr(self, x)]
            for weight in weights:
                layer_weight = getattr(self.layer, weight)
                dropoutmask = torch.ones(layer_weight.size(1)).to(layer_weight.device)
                dropoutmask = self.dropout_rec(dropoutmask)
                new_weight_hh = getattr(self.layer, weight) * dropoutmask.unsqueeze(0)
                setattr(self, weight, new_weight_hh)
        if self.dropconnect is not None:
            for weight in [name for name, param in self.layer.named_parameters() if "weight_hh" in name]:
                layer_weight = getattr(self.layer, weight)
                new_weight = self.dropconnect(layer_weight)
                setattr(self, weight, new_weight)
        if h_0 is None:
            out, h_n = self.layer(vecs)
        else:
            out, h_n = self.layer(vecs, h_0)
        return out, h_n


class OverriddenLSTMLayer(OverriddenRNNLayerBase):
    rnnlayertype = LSTMOverridden


class OverriddenGRULayer(OverriddenRNNLayerBase):
    rnnlayertype = GRUOverridden


class OverriddenRNNLayer(OverriddenRNNLayerBase):
    rnnlayertype = RNNOverridden
# endregion


# region RNN layer encoders
class RNNLayerEncoderBase(torch.nn.Module):
    rnnlayertype = None
    rnnlayertype_dropout_rec = None         # this one is used if dropout_rec > 0

    def __init__(self, indim, *dims, bidir=False, bias=True,
                 dropout_in=0., dropout_in_shared=0., dropout_rec=0., dropconnect=0., layer_norm=False):
        super(RNNLayerEncoderBase, self).__init__()
        if dropout_rec > 0 or dropconnect > 0:
            print("WARNING: using hacky batch-shared and time-shared dropout on recurrent connection")
            self.rnnlayertype = self.rnnlayertype_dropout_rec
        if not q.issequence(dims):
            dims = (dims,)
        dims = (indim,) + dims
        self.dims = dims
        self.layers = torch.nn.ModuleList()
        self.bidir = bidir
        self.bias = bias
        self.layer_norm = torch.nn.ModuleList() if layer_norm is True else None
        self.dropout_in = torch.nn.Dropout(dropout_in, inplace=False) if dropout_in > 0 else None
        self.dropout_in_shared = torch.nn.Dropout(dropout_in_shared, inplace=False) if dropout_in_shared > 0 else None
        self.dropconnect = dropconnect
        self.dropout_rec = dropout_rec
        self.make_layers()
        self.reset_parameters()
        self.ret_all_states = False     # set to True to return all states, instead of return state of last layer

    def make_layers(self):
        for i in range(1, len(self.dims)):
            if self.dropout_rec > 0 or self.dropconnect > 0:        # uses overridden rnn layers --> support dropout_rec in constructor
                layer = self.rnnlayertype(input_size=self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                          hidden_size=self.dims[i], num_layers=1, dropout_rec=self.dropout_rec,
                                          dropconnect=self.dropconnect,
                                          bidirectional=self.bidir, bias=self.bias, batch_first=True)
            else:
                layer = self.rnnlayertype(input_size=self.dims[i - 1] * (1 if not self.bidir or i == 1 else 2),
                                          hidden_size=self.dims[i], num_layers=1,
                                          bidirectional=self.bidir, bias=self.bias, batch_first=True)
            self.layers.append(layer)
            if self.layer_norm is not None:
                layernormlayer = torch.nn.LayerNorm(self.dims[i-1])
                self.layer_norm.append(layernormlayer)

    def reset_parameters(self):
        ih = (param for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform_(t)
        for t in hh:
            torch.nn.init.orthogonal_(t)
        for t in b:
            torch.nn.init.constant_(t, 0)

    def forward(self, x, mask=None, h_0s=None, ret_states=False):
        ret = self._forward(x, mask=mask, states_0=(h_0s,), ret_states=ret_states)
        return ret

    def _forward(self, x, mask=None, states_0=None, ret_states=False):
        """ top layer states return last """
        order = None
        packsorter = None
        if mask is not None:
            assert (not isinstance(x, torch.nn.utils.rnn.PackedSequence))
            x, order, packsorter = q.seq_pack(x, mask=mask, ret_sorter=True)
        out = x

        # init states -- topmost layer matches latest provided states, if not enough states, bottoms get None
        assert(states_0 is not None)
        h_0s = []       # list of all states this rnn has
        for state_0 in states_0:
            h_0s_e = [] if state_0 is None else state_0     # one element of h_0s contains a list of states for a certain state of this rnn
            assert(len(h_0s_e) <= len(self.layers))
            if order is not None:       # TODO: test !!! if x was packed, and init states provided, states must be sorted like x was sorted during packing
                h_0s_e =  [h_0s_e_e.index_select(0, packsorter) for h_0s_e_e in h_0s_e]
            h_0s_e = [h_0s_e_e.transpose(1, 0) for h_0s_e_e in h_0s_e]      # transpose incoming states (they are batch-first while layers expect direction*numlayers first)
            h_0s_e = [None] * (len(self.layers) - len(h_0s_e)) + h_0s_e
            h_0s.append(h_0s_e)
        h_0s = list(zip(*h_0s))
        # --> make a list of state tuples, per layer, then per state slot (from per state slot then per layer)
        # e.g. LSTM: [(y^0, y^1, y^2), (c^0, c^1, c^2)] => [(y^0, c^0), (y^1, c^1), (y^2, c^2)]

        states_to_ret = []

        i = 0
        for layer, h0 in zip(self.layers, h_0s):
            # region regularization
            if self.layer_norm is not None:
                if mask is not None:
                    out_normed = self.layer_norm[i](out.data)
                    out = torch.nn.utils.rnn.PackedSequence(out_normed, out.batch_sizes)
                else:
                    batsize, seqlen, hdim = out.size()
                    _out = out.view(-1, hdim)
                    _out = self.layer_norm[i](_out)
                    out = out.view(batsize, seqlen, hdim)
                # TODO test layernorm
            if self.dropout_in is not None and self.training:
                if mask is not None:       # then sequence has been packed
                    out_dropout = self.dropout_in(out.data)
                    out = torch.nn.utils.rnn.PackedSequence(out_dropout, out.batch_sizes)
                else:
                    out = self.dropout_in(out)
                # TODO: test dropouts
            if self.dropout_in_shared is not None and self.training:
                if mask is not None:
                    dropout_mask = torch.ones_like(out.data[0:1])
                    dropout_mask = self.dropout_in_shared(dropout_mask)
                    out_dropout = out.data * dropout_mask
                    out = torch.nn.utils.rnn.PackedSequence(out_dropout, out.batch_sizes)
                else:
                    dropout_mask = torch.ones_like(out[0:1, 0:1])
                    dropout_mask = self.dropout_in_shared(dropout_mask)
                    out = out * dropout_mask
                # TODO: test dropouts
            # endregion
            if h0[0] is None:
                for h0_e in h0[1:]:
                    assert(h0_e is None)
                out, h_i_n = layer(out)
            else:
                for h0_e in h0:
                    assert(h0_e is not None)
                statearg = tuple(h0) if len(h0) > 1 else h0[0]
                out, h_i_n = layer(out, statearg)
            if not q.issequence(h_i_n):
                h_i_n = (h_i_n,)
            h_i_n = [h_i_n_e.transpose(1, 0).contiguous() for h_i_n_e in h_i_n]
            if order is not None:
                h_i_n = [h_i_n_e.index_select(0, order) for h_i_n_e in h_i_n]
            states_to_ret.append(tuple(h_i_n))
            i += 1
        if mask is not None:
            out, rmask = q.seq_unpack(out, order)
        if ret_states:
            stateret = states_to_ret if self.ret_all_states is True else states_to_ret[-1][0]
            return out, stateret
        else:
            return out


class RNNEncoder(RNNLayerEncoderBase):
    rnnlayertype = torch.nn.RNN
    rnnlayertype_dropout_rec = OverriddenRNNLayer


class GRUEncoder(RNNLayerEncoderBase):
    rnnlayertype = torch.nn.GRU
    rnnlayertype_dropout_rec = OverriddenGRULayer


class LSTMEncoder(RNNLayerEncoderBase):
    rnnlayertype = torch.nn.LSTM
    rnnlayertype_dropout_rec = OverriddenLSTMLayer

    def forward(self, x, mask=None, batsize=None, y_0s=None, c_0s=None, ret_states=False):
        ret = self._forward(x, mask=mask, states_0=(y_0s, c_0s,), ret_states=ret_states)
        return ret
# endregion
# endregion


# region rec cell encoders
class RecCellEncoder(torch.nn.Module):
    celltype = None

    def __init__(self, indim, *dims, bidir=False, bias=True, dropout_in=0., dropout_rec=0., **kw):
        super(RecCellEncoder, self).__init__(**kw)
        if not q.issequence(dims):
            dims = (dims,)
        dims = (indim,) + dims
        self.dims = dims
        self.dropout_in, self.dropout_rec = dropout_in, dropout_rec
        self.layers = torch.nn.ModuleList()
        self.rev_layers = torch.nn.ModuleList() if bidir else None
        self.bidir = bidir
        self.bias = bias
        self.make_layers()
        self.ret_all_states = False

    def make_layers(self):
        for i in range(1, len(self.dims)):
            layer = self.celltype(self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                             self.dims[i],
                             dropout_in=self.dropout_in, dropout_rec=self.dropout_rec,
                             bias=self.bias)
            self.layers.append(layer)
            # add reverse layer if bidir
            if self.rev_layers is not None:
                layer = self.celltype(self.dims[i-1] * (1 if not self.bidir or i == 1 else 2),
                                 self.dims[i],
                                 dropout_in=self.dropout_in, dropout_rec=self.dropout_rec,
                                 bias=self.bias)
                self.rev_layers.append(layer)

    def forward(self, x, gate=None, mask=None, ret_states=False):
        out = x
        out = torch.split(out, 1, 1)    # split in sequence dimension
        out = [out_e.squeeze(1) for out_e in out]

        mask = (gate if mask is None else mask.float() * gate) \
                if gate is not None \
                else (mask.float() if mask is not None else None)

        assert(len(self.layers) > 0)
        i = 0
        for layer in self.layers:
            # go forward in time
            acc = []
            t = 0
            while t < len(out):
                y_t = layer(out[t], mask_t=mask[:, t] if mask is not None else None)
                acc.append(y_t)
                t += 1
            final_state = acc[-1].unsqueeze(1)
            # go backward in time
            if self.rev_layers is not None:
                rev_layer = self.rev_layers[i]
                rev_acc = []
                t = len(out)-1
                while t >= 0:
                    y_t = rev_layer(out[t], mask_t=mask[:, t] if mask is not None else None)
                    rev_acc.append(y_t)
                    t -= 1
                final_state = torch.cat([acc[-1].unsqueeze(1),
                                         rev_acc[-1].unsqueeze(1)], 1)
                rev_acc = rev_acc[::-1]     # reverse for merge
                acc = [torch.cat([acc_i, rev_acc_i], 1) for acc_i, rev_acc_i in zip(acc, rev_acc)]    # merge
            out = acc
            i += 1

        if ret_states:
            if self.ret_all_states:
                raise NotImplemented("ret_all_states is not implemented, use states on the individual cells instead")
            stateret = final_state
            return torch.stack(out, 1), stateret
        else:
            return torch.stack(out, 1)


class RNNCellEncoder(RecCellEncoder):
    celltype = RNNCell


class GRUCellEncoder(RecCellEncoder):
    celltype = GRUCell


class LSTMCellEncoder(RecCellEncoder):
    celltype = LSTMCell
# endregion
# endregion


# region automasker
class AutoMaskedOut(torch.nn.Module):
    def __init__(self, core, automasker=None, do_softmax=False, do_logsoftmax=False, **kw):
        """
        :param core:        must return scores over output tokens
        :param automasker:  an AutoMasker
        :param kw:
        """
        super(AutoMaskedOut, self).__init__(**kw)
        assert(not (do_softmax and do_logsoftmax))
        self.core = core
        self.automasker = automasker
        self.sm = torch.nn.Softmax(-1) if do_softmax else (torch.nn.LogSoftmax(-1) if do_logsoftmax else None)

    def update(self, x):
        if self.automasker is not None:
            self.automasker.update(x)

    def forward(self, *args, **kw):
        coreret = self.core(*args, **kw)    # (batsize, [seqlen, ] outvocsize) - logits/logprobs/probs for every out voc
        mask = self.automasker.get_out_mask().to(coreret.device).float()        # 0/1 mask
        coreret += torch.log(mask)
        if self.sm is not None:
            coreret = self.sm(coreret)
        return coreret

        # raise NotImplemented("use subclass")
        # """ assumes first of *args is a tensor on the right device """
        # assert ("mask" not in kw)
        # mask = self.automasker.get_out_mask()
        # device = args[0].device
        # mask = mask.to(device)
        # kw["mask"] = mask
        # ret = self.baseout(*args, **kw)
        # return ret



class AutoMasker(torch.nn.Module):
    """ Subclass this with your own rules
        How to use:
            - write a subclass implementing at least .get_out_tokens_for_history()
            - create an AutoMaskedOut with this AutoMasker and an output layer
            - plug the AutoMaskedOut into a supporting DecoderCell (must feed its input to AutoMaskedOut)
    """
    def __init__(self, inpD, outD, mode="allow", **kw):
        super(AutoMasker, self).__init__(**kw)
        self.inpD, self.outD = inpD, outD
        self.RinpD = {v: k for k, v in self.inpD.items()}
        self.mode = mode     # "allow" or "refuse"
        self.history = None     # will hold all tokens fed for every example in a batch
        self.device = None
        self.test_only = False

    def reset(self):
        self.history = None
        # self.device = None

    def batch_reset(self):
        self.reset()

    def update(self, x):
        """ updates automasker with next element in the sequence (which was the output at the previous timestep)
        :param x:   (batsize,) integer ids in inpD """
        if self.training and self.test_only:
            pass
        else:
            assert(x.dim() == 1)
            self.device = x.device
            intokens = []
            for i in range(len(x)):
                intokens.append(self.RinpD[x[i].detach().cpu().item()])
            self.update_tokens(intokens)

    def update_tokens(self, intokens):
        """ update given input tokens for batch
            :param intokens:    list of batsize of strings """
        if self.history is None:
            self.history = [[x] for x in intokens]
        else:
            assert(len(intokens) == len(self.history))
            for intoken, example_history in zip(intokens, self.history):
                example_history.append(intoken)

    def get_out_mask(self):
        """ returns a mask over outD """
        if self.training and self.test_only:
            return None
        else:
            tokenses = self.get_out_tokens()    # list of lists
            vocsize = max(self.outD.values()) + 1
            startcreator = torch.zeros if self.mode == "allow" else torch.ones
            mask = startcreator(len(tokenses), vocsize)
            for i, tokens in enumerate(tokenses):
                a = 1 if self.mode == "allow" else 0
                if tokens is None:
                    mask[i, :] = a
                else:
                    for token in tokens:
                        mask[i, self.outD[token]] = a
            return mask.to(self.device)

    def get_out_tokens(self):
        """ get valid tokens for output
            must return a list of list-like of strings """
        if self.history is None:
            raise q.SumTingWongException("can't produce out tokens without history --> TODO")
        ret = [self.get_out_tokens_for_history(i, example_history)
               for i, example_history in enumerate(self.history)]
        return ret

    def get_out_tokens_for_history(self, i, hist):
        """
        Must return a list of tokens given the given history.
        :param hist: list-like of strings (tokens)
        :return:
        """
        raise NotImplemented("use subclass")
# endregion


class FlatEncoder(torch.nn.Module):
    def __init__(self, embdim, dims, word_dic, bidir=False, dropout_in=0., dropout_rec=0., gfrac=0., meanpoolskip=False):
        """ embdim for embedder, dims is a list of dims for RNN"""
        super(FlatEncoder, self).__init__()
        self.emb = q.PartiallyPretrainedWordEmb(embdim, worddic=word_dic, gradfracs=(1., gfrac))
        self.lstm = q.FastestLSTMEncoder(embdim, *dims, bidir=bidir, dropout_in=dropout_in, dropout_rec=dropout_rec)
        self.meanpoolskip = meanpoolskip
        self.adapt_lin = None
        outdim = dims[-1] * 2
        if meanpoolskip and outdim != embdim:
            self.adapt_lin = torch.nn.Linear(embdim, outdim, bias=False)
        self.debug = False

    def forward(self, x):
        embs, mask = self.emb(x)
        if self.debug:
            embs = torch.tensor(embs.detach().numpy())
            embs.requires_grad = True
        _ = self.lstm(embs, mask=mask)
        final_state = self.lstm.y_n[-1]
        final_state = final_state.contiguous().view(x.size(0), -1)
        if self.meanpoolskip:
            if self.adapt_lin is not None:
                embs = self.adapt_lin(embs)
            meanpool = embs.sum(1)
            masksum = mask.float().sum(1).unsqueeze(1)
            meanpool = meanpool / masksum
            final_state = final_state + meanpool
        if self.debug:
            return final_state, embs
        return final_state


class SimpleEncoder(torch.nn.Module):
    pass