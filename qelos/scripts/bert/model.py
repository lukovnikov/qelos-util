import torch
import qelos as q
import numpy as np


# region special BERT models
class AdaptedBERTEncoderSingle(torch.nn.Module):
    pad_id = 0

    def __init__(self, bert, numout=2, oldvocab=None, specialmaps={0: [0]}, **kw):
        """
        :param bert:
        :param numout:
        :param oldvocab:    word vocab used for inpids fed to this model. Will be translated to new (bert) vocab.
        :param kw:
        """
        super(AdaptedBERTEncoderSingle, self).__init__(**kw)
        self.bert = bert
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
        torch.nn.init.normal_(self.lin.weight, 0, self.bert.init_range)
        torch.nn.init.zeros_(self.lin.bias)

    def forward(self, inpids):
        """
        :param inpids:  (batsize, seqlen) int ids in oldvocab
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
        _, poolout = self.bert(newinp, typeids, padmask)
        poolout = self.dropout(poolout)
        logits = self.lin(poolout)
        return logits


class AdaptedBERTEncoderPair(AdaptedBERTEncoderSingle):
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
# endregion


if __name__ == '__main__':
    q.argprun(test_adapted_bert_encoder_pair)