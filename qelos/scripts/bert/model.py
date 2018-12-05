import torch
import qelos as q
import numpy as np
from copy import deepcopy


# region special BERT models
class AdaptedBERTEncoderSingle(torch.nn.Module):
    pad_id = 0

    def __init__(self, bert, numout=2, oldvocab=None, specialmaps={0: [0]}, **kw):
        """
        :param bert:
        :param numout:      number of outputs. If -1, output layer is not applied and the encoding is returned.
                                If 0, number of outputs is actually 1 and output is squeezed before returning.
        :param oldvocab:    word vocab used for inpids fed to this model. Will be translated to new (bert) vocab.
        :param kw:
        """
        super(AdaptedBERTEncoderSingle, self).__init__(**kw)
        self.bert = bert
        specialmaps = deepcopy(specialmaps)
        self.numout = numout
        if self.numout >= 0:
            numout = numout if numout > 0 else 1
            self.lin = torch.nn.Linear(bert.dim, numout)
            self.dropout = torch.nn.Dropout(p=bert.dropout)
            self.reset_parameters()
        self.oldD = oldvocab
        self.D = self.bert.D
        self.wp_tok = q.bert.WordpieceTokenizer(self.bert.D)
        # create mapper
        self.oldD2D = {}
        for k, v in self.oldD.items():
            if v in specialmaps:
                self.oldD2D[v] = specialmaps[v]
            else:
                key_pieces = self.wp_tok.tokenize(k)
                key_piece_ids = [self.D[key_piece] for key_piece in key_pieces]
                self.oldD2D[v] = key_piece_ids

    def reset_parameters(self):
        if self.numout >= 0:
            torch.nn.init.normal_(self.lin.weight, 0, self.bert.init_range)
            torch.nn.init.zeros_(self.lin.bias)

    def forward(self, inpids, ret_layer_outs=False):
        """
        :param inpids:  (batsize, seqlen) int ids in oldvocab
        :param ret_layer_outs: boolean, whether to return the raw transformer states
        :return:
        """
        # transform inpids
        maxlen = 0
        newinp = torch.zeros_like(inpids)
        for i in range(len(inpids)):    # iter over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(inpids.size(1)):     # iter over seqlen
                wordpieces = self.oldD2D[inpids[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
            newinp[i, last_nonpad] = self.D["[SEP]"]
            maxlen = max(maxlen, last_nonpad+1)
        newinp = newinp[:, :maxlen]

        # do forward
        typeids = torch.zeros_like(newinp)
        padmask = newinp != self.pad_id
        layerouts, poolout = self.bert(newinp, typeids, padmask)
        ret = None
        if self.numout >= 0:
            poolout = self.dropout(poolout)
            logits = self.lin(poolout)
            if self.numout == 0:
                logits = logits.squeeze(-1)     # (batsize,)
            ret = logits
        else:
            ret = poolout
        if ret_layer_outs:
            return ret, layerouts, padmask
        else:
            return ret


class AdaptedBERTEncoderPair(AdaptedBERTEncoderSingle):
    """
    Adapts from worddic.
    Output is generated by encoding the pair in one sequence.
    """
    def forward(self, a, b):
        """
        :param a:   (batsize, seqlen_a)
        :param b:   (batsize, seqlen_b)
        :return:
        """
        assert(len(a) == len(b))
        # transform inpids
        newinp = torch.zeros_like(a)
        typeflip = []
        maxlen = 0
        for i in range(len(a)):    # iter over examples
            k = 0
            newinp[:, 0] = self.D["[CLS]"]
            k += 1
            last_nonpad = 0
            for j in range(a.size(1)):     # iter over seqlen
                wordpieces = self.oldD2D[a[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
            newinp[i, last_nonpad] = self.D["[SEP]"]
            k = last_nonpad + 1
            typeflip.append(k)
            for j in range(b.size(1)):
                wordpieces = self.oldD2D[b[i, j].cpu().item()]
                for wordpiece in wordpieces:
                    newinp[i, k] = wordpiece
                    k += 1
                    if newinp.size(1) <= k + 1:
                        newinp = torch.cat([newinp, torch.zeros_like(newinp)], 1)
                if newinp[i, k-1].cpu().item() != self.pad_id:
                    last_nonpad = k
            newinp[i, last_nonpad] = self.D["[SEP]"]
            maxlen = max(maxlen, last_nonpad + 1)
        newinp = newinp[:, :maxlen]

        # do forward
        typeids = torch.zeros_like(newinp)
        for i, flip in enumerate(typeflip):
            typeids[i, flip:] = 1
        padmask = newinp != self.pad_id
        _, poolout = self.bert(newinp, typeids, padmask)
        poolout = self.dropout(poolout)
        logits = self.lin(poolout)
        return logits


class AdaptedBERTCompare(torch.nn.Module):
    """
    Encodes left, encodes right, compares with a special function.
    """
    oldvocab_norel_tok = "ft"
    norel_tok = "—"
    def __init__(self, bert, oldvocab=None, specialmaps={0: [0]}, share_left_right=True, **kw):
        super(AdaptedBERTCompare, self).__init__(**kw)
        specialmaps = deepcopy(specialmaps)
        self.left_enc = AdaptedBERTEncoderSingle(bert, numout=-1, oldvocab=oldvocab, specialmaps=specialmaps)
        rightbert = bert if share_left_right else deepcopy(bert)
        if self.oldvocab_norel_tok in oldvocab:
            specialmaps[oldvocab[self.oldvocab_norel_tok]] = [bert.D[self.norel_tok]]
        self.right_enc = AdaptedBERTEncoderSingle(rightbert, numout=-1, oldvocab=oldvocab, specialmaps=specialmaps)

    def forward(self, q, rel):
        aenc = self.left_enc(q)
        benc = self.right_enc(rel)
        # dot
        dots = torch.einsum("bd,bd->b", [aenc, benc])
        return dots


class AdaptedBERTCompareSlotPtr(AdaptedBERTCompare):
    def __init__(self, bert, oldvocab=None, specialmaps={0: [0]}, share_left_right=True, **kw):
        super(AdaptedBERTCompareSlotPtr, self).__init__(bert, oldvocab=oldvocab, specialmaps=specialmaps,
                                                        share_left_right=share_left_right)
        self.linear = torch.nn.Linear(bert.dim, 2)
        self.sm = torch.nn.Softmax(1)

    def forward(self, q, rel1, rel2):
        qenc_final, qenc_layerouts, qenc_mask = self.left_enc(q, ret_layer_outs=True)
        ys = qenc_layerouts[-1]

        # compute scores
        scores = self.linear(ys)
        scores = scores + torch.log(qenc_mask.float().unsqueeze(-1))
        scores = self.sm(scores)    # (batsize, seqlen, 2)

        # get summaries
        ys = ys.unsqueeze(2)
        scores = scores.unsqueeze(3)
        b = ys * scores     # (batsize, seqlen, 2, dim)
        summaries = b.sum(1)    # (batsize, 2, dim)
        summaries = summaries.view(summaries.size(0), -1)   # (batsize, 2*dim)

        # compute relation encodings
        rel1_enc = self.right_enc(rel1)
        rel2_enc = self.right_enc(rel2)
        relenc = torch.cat([rel1_enc, rel2_enc], 1)     # (batsize, 2*dim)

        # compute output scores
        dots = torch.einsum("bd,bd->b", [summaries, relenc])
        return dots


def test_adapted_bert_encoder_single(lr=0):
    vocab = "<PAD> [UNK] the a cucumber apple mango banana fart art plant"
    vocab = dict(zip(vocab.split(), range(len(vocab.split()))))
    print(vocab)

    x = [[1,2,3,4,5,6,0,0],[2,4,5,6,7,8,9,10]]
    x = torch.tensor(x)

    bert = q.bert.TransformerBERT.load_from_dir("../../../data/bert/bert-base/")
    m = AdaptedBERTEncoderSingle(bert, oldvocab=vocab)

    print("made model")

    y = m(x)
    print(y)
    pass


def test_adapted_bert_encoder_pair(lr=0):
    vocab = "<PAD> [UNK] the a cucumber apple mango banana fart art plant"
    vocab = dict(zip(vocab.split(), range(len(vocab.split()))))
    print(vocab)

    a = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    a = torch.tensor(a)

    b = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    b = torch.tensor(b)

    bert = q.bert.TransformerBERT.load_from_dir("../../../data/bert/bert-base/")
    m = AdaptedBERTEncoderPair(bert, oldvocab=vocab)

    print("made model")

    y = m(a, b)
    print(y)
    pass


def test_adapted_bert_compare(lr=0):
    vocab = "<PAD> [UNK] the a cucumber apple mango banana fart art plant ft"
    vocab = dict(zip(vocab.split(), range(len(vocab.split()))))
    print(vocab)

    a = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    a = torch.tensor(a)

    b = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    b = torch.tensor(b)

    c = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    c = torch.tensor(c)

    bert = q.bert.TransformerBERT.load_from_dir("../../../data/bert/bert-base/")
    m = AdaptedBERTCompare(bert, oldvocab=vocab)

    print("made model")

    y = m(a, b)
    print(y)
    pass


def test_adapted_bert_compare_slotptr(lr=0):
    vocab = "<PAD> [UNK] the a cucumber apple mango banana fart art plant ft"
    vocab = dict(zip(vocab.split(), range(len(vocab.split()))))
    print(vocab)

    a = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    a = torch.tensor(a)

    b = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    b = torch.tensor(b)

    c = [
        [1, 2, 3, 0, 0, 0, 0, 0],
        [2, 3, 4, 5, 6, 7, 8, 0],
        [9, 10, 0, 0, 0, 0, 0, 0]
    ]
    c = torch.tensor(c)

    bert = q.bert.TransformerBERT.load_from_dir("../../../data/bert/bert-base/")
    m = AdaptedBERTCompareSlotPtr(bert, oldvocab=vocab)

    print("made model")

    y = m(a, b, c)
    print(y)
    pass

# endregion


if __name__ == '__main__':
    # q.argprun(test_adapted_bert_encoder_pair)
    q.argprun(test_adapted_bert_compare_slotptr)