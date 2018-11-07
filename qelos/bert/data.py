import qelos as q
import json
import re
import numpy as np


__all__ = ["WordPieceMatrix", "BERTMatrixSingle", "BERTMatrixPair"]


class WordPieceMatrix(object):
    def __init__(self, vocab, start="[START]", end="[END]",
                 unk="[UNK]", pad="[PAD]",
                 do_lower_case=True, maxlen=np.infty):
        super(WordPieceMatrix, self).__init__()
        self.D = vocab
        self.start_token, self.end_token = start, end
        self.unk_token, self.pad_token = unk, pad
        for tok in [self.start_token, self.end_token, self.unk_token, self.pad_token]:
            assert(tok in self.D)
        self.matrix = None     # the matrix
        self.tokenizer = q.tokens.FullTokenizer(self.D, do_lower_case=do_lower_case)
        self.mattokens = []
        self.maxlen = maxlen
        self._x_maxlen = -1

    def add(self, x):   # x is a string of one example
        if self.matrix is not None:
            raise q.SumTingWongException("can't add to finalized {}".format(self.__class__.__name__))
        xtokens = self.tokenizer.tokenize(x)
        addspace = sum([1 if tok is not None else 0 for tok in [self.start_token, self.end_token]])
        if len(xtokens) > self.maxlen - addspace:
            xtokens = xtokens[:self.maxlen-addspace]
        if self.start_token is not None:
            xtokens = [self.start_token] + xtokens
        if self.end_token is not None:
            xtokens = xtokens + [self.end_token]
        self._x_maxlen = max(self._x_maxlen, len(xtokens))
        self.mattokens.append(xtokens)

    def __getitem__(self, i):
        ids = list(self.matrix[i])
        rD = {v: k for k, v in self.D.items()}
        tokens = [rD[x] for x in ids if x != self.D[self.pad_token]]
        _tokens = " ".join(tokens)
        return _tokens

    def finalize(self):
        mat = np.ones((len(self.mattokens), self._x_maxlen), dtype="int64") * self.D[self.pad_token]
        for i, xtokens in enumerate(self.mattokens):
            mat[i, :len(xtokens)] = [self.D[tok] for tok in xtokens]
        self.matrix = mat
        return self


class BERTMatrixSingle(WordPieceMatrix):
    def __init__(self, vocab, start="[CLS]", end="[SEP]",
                 unk="[UNK]", pad="[PAD]",
                 do_lower_case=True, maxlen=np.infty):
        super(BERTMatrixSingle, self).__init__(vocab, start=start, end=end, unk=unk, pad=pad,
                                               do_lower_case=do_lower_case, maxlen=maxlen)
    def __getitem__(self, i):
        ids = list(self.matrix[0][i])
        rD = {v: k for k, v in self.D.items()}
        tokens = [rD[x] for x in ids if x != self.D[self.pad_token]]
        _tokens = " ".join(tokens)
        types = " ".join(map(str, list(self.matrix[1][i])[:len(tokens)]))
        masks = " ".join(map(str, list(self.matrix[2][i])[:len(tokens)]))
        return _tokens, types, masks

    def finalize(self):
        super(BERTMatrixSingle, self).finalize()
        typemat = np.zeros_like(self.matrix)
        maskmat = (self.matrix != self.D[self.pad_token]).astype("int64")
        self.matrix = [self.matrix, typemat, maskmat]
        return self


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class BERTMatrixPair(BERTMatrixSingle):
    def __init__(self, vocab, start="[CLS]", end="[SEP]", sep="[SEP]",
                 unk="[UNK]", pad="[PAD]",
                 do_lower_case=True, maxlen=np.infty):
        super(BERTMatrixPair, self).__init__(vocab, start=start, end=end, unk=unk, pad=pad,
                                               do_lower_case=do_lower_case, maxlen=maxlen)
        self.sep_token = sep
        self.sep_positions = []

    def add(self, x, y):   # x is a string of one example
        if self.matrix is not None:
            raise q.SumTingWongException("can't add to finalized {}".format(self.__class__.__name__))
        xtokens = self.tokenizer.tokenize(x)
        ytokens = self.tokenizer.tokenize(y)
        if len(xtokens) + len(ytokens) > self.maxlen - 3:
            _truncate_seq_pair(xtokens, ytokens, self.maxlen - 3)
        tokens = [self.start_token] + xtokens + [self.sep_token] + ytokens + [self.end_token]
        self._x_maxlen = max(self._x_maxlen, len(tokens))
        self.sep_positions.append((len(xtokens) + 1, len(tokens)))
        self.mattokens.append(tokens)

    def finalize(self):
        super(BERTMatrixPair, self).finalize()
        for i, seppos in enumerate(self.sep_positions):
            self.matrix[1][i, seppos[0]:seppos[1]] = 1
        return self


def test_load(vocabpath):
    """ for loading single-string tasks for BERT """
    vocab = q.tokens.load_vocab(vocabpath)
    bm = BERTMatrixPair(vocab)
    bm.add("is this wrecklinghausen?", "no it is not.")
    bm.add("is this jacksonville?", "not now.")
    bm.finalize()
    print("\n".join(bm[0]))
    print("\n".join(bm[1]))

    bms = BERTMatrixSingle(vocab)
    bms.add("is this jacksonville?")
    bms.add("is this wrecklinghausen?")
    bms.add("is this hottentottententententoonstelling?")
    bms.finalize()
    print("\n".join(bms[0]))
    print("\n".join(bms[1]))
    print("\n".join(bms[2]))


def run(vocabpath="../../data/bert/bert-base/vocab.txt"):
    ret = test_load(vocabpath)
    return ret


if __name__ == '__main__':
    q.argprun(run)
