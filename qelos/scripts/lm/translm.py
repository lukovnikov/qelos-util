import qelos as q
import torch
import numpy as np
import os
import random
from functools import partial
import math


def load_data(p="../../../datasets/wikitext2/",
              batsize=100, eval_batsize=10, seqlen=35, subsample_eval=10):

    class Dictionary(object):
        def __init__(self):
            self.word2idx = {}
            self.idx2word = []

        def add_word(self, word):
            if word not in self.word2idx:
                self.idx2word.append(word)
                self.word2idx[word] = len(self.idx2word) - 1
            return self.word2idx[word]

        def __len__(self):
            return len(self.idx2word)

    class Corpus(object):
        def __init__(self, path):
            self.dictionary = Dictionary()
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))

        def tokenize(self, path):
            """Tokenizes a text file."""
            assert os.path.exists(path)
            # Add words to the dictionary
            with open(path, 'r', encoding="utf8") as f:
                tokens = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    tokens += len(words)
                    for word in words:
                        self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r', encoding="utf8") as f:
                ids = torch.LongTensor(tokens)
                token = 0
                for line in f:
                    words = line.split() + ['<eos>']
                    for word in words:
                        ids[token] = self.dictionary.word2idx[word]
                        token += 1

            return ids

    corpus = Corpus(p)

    def batchify(data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        return data

    D = corpus.dictionary.word2idx
    train_data = LMLoader(corpus.train, seqlen, batsize=batsize)
    # valid_data = batchify(corpus.valid, eval_batsize)
    # valid_data = LMLoader_Test(valid_data, seqlen)
    valid_data = LMLoader_Test(corpus.valid, seqlen, batsize=batsize, subsample=subsample_eval)
    # test_data = batchify(corpus.test, eval_batsize)
    # test_data = LMLoader_Test(test_data, seqlen)
    test_data = LMLoader_Test(corpus.test, seqlen, batsize=eval_batsize, subsample=1)
    return train_data, valid_data, test_data, D


class LMLoader(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen, batsize):
        super(LMLoader, self).__init__()
        self.data = data
        self.seqlen = seqlen
        self.batsize = batsize

    def __iter__(self):
        return _LMLoaderIter(self)

    def __len__(self):
        return self.data.size(0) // (self.seqlen * self.batsize)


class _LMLoaderIter(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter, self).__init__()
        self.lml = lmloader
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.lml)

    def __next__(self):
        if self.i >= len(self.lml):
            raise StopIteration()
        self.i += 1
        out = []
        for k in range(self.lml.batsize):
            start = random.randint(0, self.lml.data.size(0) - self.lml.seqlen)
            out.append(self.lml.data[start: start+self.lml.seqlen])
        out = torch.stack(out, 0)
        gold = out[:, 1:]
        out = out[:, :-1]
        return out, gold


class LMLoader_Test(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen, batsize, subsample=1):
        super(LMLoader_Test, self).__init__()
        self.data = data        # (totallen,)
        self.seqlen = seqlen
        self.batsize = batsize
        self.seglen = data.size(0) // batsize
        self.starts = [i*self.seglen for i in range(batsize)]
        d = [data[i: i+self.seglen] for i in self.starts]
        self._data = torch.stack(d, 0)
        self.subsample = subsample

    def __iter__(self):
        return _LMLoaderIter_Test(self)

    def __len__(self):
        return self.data.size(0) // (self.batsize * self.subsample)


class _LMLoaderIter_Test(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter_Test, self).__init__()
        self.lml = lmloader
        self.i = 1

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.lml)

    def __next__(self):
        if self.i >= self.lml._data.size(1):
            raise StopIteration()
        out = self.lml._data[:, max(0, self.i-self.lml.seqlen):self.i]
        gold = self.lml._data[:, self.i:self.i+1]
        self.i += self.lml.subsample
        return out, gold


# region transformer language model
class TransformerLM(torch.nn.Module):
    def __init__(self, dim=512, worddic=None, numlayers=3, numheads=8, activation=q.GeLU,
                 embedding_dropout=0., attention_dropout=0., residual_dropout=0.,
                 word_dropout=0., relpos=True, tie_wordvecs=False, maxlen=512):
        super(TransformerLM, self).__init__()
        self.wordemb = q.WordEmb(dim, worddic=worddic, word_dropout=word_dropout)
        posemb = None
        if relpos is False:
            print("using learned absolute position embeddings")
            posembD = dict(zip(range(maxlen), range(maxlen)))
            posemb = q.WordEmb(dim, worddic=posembD)
        self.transformer = q.TransformerDecoder(dim=dim, numlayers=numlayers, numheads=numheads, activation=activation,
                                                embedding_dropout=embedding_dropout, attention_dropout=attention_dropout,
                                                residual_dropout=residual_dropout, relpos=relpos, noctx=True, maxlen=maxlen,
                                                posemb=posemb)
        q.RecDropout.convert_to_standard_in(self.transformer)
        self.wordout = q.WordLinout(dim, worddic=worddic)
        if tie_wordvecs:
            self.wordout.weight = self.wordemb.weight

    def forward(self, x):   # (batsize, seqlen) wordids
        xemb, _ = self.wordemb(x)
        enc = self.transformer(xemb)
        out = self.wordout(enc)
        return out


class TransformerLMCell(torch.nn.Module):
    def __init__(self, core:TransformerLM):
        super(TransformerLMCell, self).__init__()
        self.core = core

    def forward(self, x):   # (batsize, seqlen) wordids
        out = self.core(x)
        out = out[:, -1].unsqueeze(1)
        return out
# endregion


class PPLfromCE(q.LossWrapper):
    def __init__(self, celosswrapper, **kw):
        super(PPLfromCE, self).__init__(celosswrapper.loss, **kw)
        self.celosswrapper = celosswrapper

    def __call__(self, pred, gold):
        return 0

    def get_epoch_error(self):
        return math.exp(self.celosswrapper.get_epoch_error())


def run(lr=2.5e-4,
        edropout=0.1,
        wdropout=0.1,
        rdropout=0.1,
        adropout=0.1,
        dropout=-1.,
        numlayers=2,
        numheads=8,
        abspos=False,
        tie_wordvecs=False,
        gradnorm=0.5,
        epochs=200,
        dim=256,
        seqlen=50,
        batsize=32,
        eval_batsize=64,
        cuda=False,
        gpu=0,
        test=True,
        subsampleeval=10,
        wreg=1e-6,
        lrcycle=5,
        lrwarmup=3,
        ):
    tt = q.ticktock("script")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
    tt.tick("loading data")
    train_batches, valid_batches, test_batches, D = \
        load_data(batsize=batsize, eval_batsize=eval_batsize, seqlen=seqlen, subsample_eval=subsampleeval)
    tt.tock("data loaded")
    print("{} batches in train".format(len(train_batches)))
    if dropout >= 0.:
        edropout, adropout, rdropout, wdropout = dropout, dropout, dropout, dropout
    relpos = not abspos

    tt.tick("creating model")

    m = TransformerLM(dim=dim, worddic=D, numlayers=numlayers, numheads=numheads,
                      activation=q.GeLU, embedding_dropout=edropout, attention_dropout=adropout,
                      word_dropout=wdropout, residual_dropout=rdropout, relpos=relpos,
                      tie_wordvecs=tie_wordvecs, maxlen=2*seqlen).to(device)
    valid_m = TransformerLMCell(m)

    if test:
        for i, batch in enumerate(valid_batches):
            batch = [batch_e.to(device) for batch_e in batch]
            y = valid_m(batch[0])
            if i > 5:
                break
        for i, batch in enumerate(valid_batches):
            pass
        print(i, batsize, seqlen, valid_batches.data.size(0))
        print(y.size())
        # return
    # return

    loss = q.LossWrapper(q.CELoss(mode="logits"))
    validloss = q.LossWrapper(q.CELoss(mode="logits"))
    validlosses = [validloss, PPLfromCE(validloss)]
    testloss = q.LossWrapper(q.CELoss(mode="logits"))
    testlosses = [testloss, PPLfromCE(testloss)]
    for l in [loss] + validlosses + testlosses:   # put losses on right device
        l.loss.to(device)

    # optim = torch.optim.SGD(m.parameters(), lr=lr)
    numbats = len(train_batches)
    print("{} batches in training".format(numbats))
    optim = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=wreg)
    # lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=1/4, patience=0, verbose=True)
    # lrp_f = lambda: lrp.step(validloss.get_epoch_error())
    sched = q.CosineLRwithWarmup(optim, lrcycle * numbats, warmup=lrwarmup * numbats)

    train_batch_f = partial(q.train_batch,
                            on_before_optim_step=[lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm),
                                                  lambda: sched.step()])
    train_epoch_f = partial(q.train_epoch, model=m, dataloader=train_batches, optim=optim, losses=[loss],
                            device=device, _train_batch=train_batch_f)
    valid_epoch_f = partial(q.test_epoch, model=valid_m, dataloader=valid_batches, losses=validlosses, device=device)
    tt.tock("created model")
    tt.tick("training")
    q.run_training(train_epoch_f, valid_epoch_f, max_epochs=epochs, validinter=1)
    tt.tock("trained")

    tt.tick("testing")
    testresults = q.test_epoch(model=valid_m, dataloader=test_batches, losses=testlosses, device=device)
    print(testresults)
    tt.tock("tested")


if __name__ == '__main__':
    q.argprun(run)