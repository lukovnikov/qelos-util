import torch
import qelos as q
import os
import numpy as np
import math
from functools import partial


def load_data(p="../../../datasets/wikitext2/",
              batsize=100, eval_batsize=10, seqlen=35):

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

    train_data = batchify(corpus.train, batsize)
    valid_data = batchify(corpus.valid, eval_batsize)
    test_data = batchify(corpus.test, eval_batsize)
    D = corpus.dictionary.word2idx
    train_data = LMLoader(train_data, seqlen)
    valid_data = LMLoader(valid_data, seqlen)
    test_data = LMLoader(test_data, seqlen)
    return train_data, valid_data, test_data, D


class VariableSeqlen(object):
    def __init__(self, minimum=5, mu=50, maximum_offset=10, sigma=5):
        super(VariableSeqlen, self).__init__()
        self.min = minimum
        self.mu = mu
        self.sigma = sigma
        self.max_off = maximum_offset

    def __q_v__(self):
        return round(min(max(self.min, np.random.normal(self.mu, self.sigma)), self.max_off + self.mu))


class LMLoader(object):
    """ data loader for LM data """
    def __init__(self, data, seqlen:VariableSeqlen=None):
        super(LMLoader, self).__init__()
        self.data = data
        self.seqlen = seqlen

    def __iter__(self):
        return _LMLoaderIter(self)

    def __len__(self):
        return 1 + ((len(self.data)-1) // self.seqlen.mu)


class _LMLoaderIter(object):
    def __init__(self, lmloader):
        super(_LMLoaderIter, self).__init__()
        self.lml = lmloader
        self.i = 0

    def __iter__(self):
        return self

    def __len__(self):
        return 1 + ((len(self.lml.data)-1) // self.lml.seqlen.mu)

    def __next__(self):
        if self.i < len(self.lml.data)-1:
            seqlen = min(q.v(self.lml.seqlen), len(self.lml.data) - self.i - 1)
            batch = self.lml.data[self.i: self.i + seqlen]
            batch_g = self.lml.data[self.i+1: self.i+1 + seqlen]
            self.i += seqlen
            return batch.transpose(1, 0), batch_g.transpose(1, 0)
        else:
            self.i = 0
            raise StopIteration()


class RNNLayer_LM(torch.nn.Module):
    encodertype = q.LSTMEncoder

    def __init__(self, *dims:int, worddic:dict=None, bias:bool=True,
                 dropout:float=0., dropouti:float=0., dropouth:float=0., dropoute:float=0., **kw):
        super(RNNLayer_LM, self).__init__(**kw)
        self.dims = dims
        self.D = worddic
        self.states = None
        # make layers
        self.emb = q.WordEmb(dims[0], worddic=self.D)
        self.out = q.WordLinout(dims[-1], worddic=self.D)
        self.rnn = self.encodertype(*dims, bidir=False, bias=bias, dropout_in=dropout)
        self.rnn.ret_all_states = True
        self.dropout = torch.nn.Dropout(p=dropout)
        self.dropouti = torch.nn.Dropout(p=dropouti)
        self.dropoute = torch.nn.Dropout(p=dropoute)
        self.dropouth = torch.nn.Dropout(p=dropouth)

    def epoch_reset(self):
        self.states = None

    def forward(self, x):
        emb, xmask = self.emb(x)
        # do actual forward
        states_0 = ((None, None) if self.encodertype == q.LSTMEncoder else (None,)) \
            if self.states is None else self.states
        out, all_states = self.rnn._forward(emb, mask=xmask, states_0=states_0, ret_states=True)
        # backup states
        all_states = [[all_state_e.detach() for all_state_e in all_state] for all_state in all_states]
        self.states = list(zip(*all_states))

        # output
        out = self.dropout(out)
        out = self.out(out)
        return out


class PPLfromCE(q.LossWrapper):
    def __init__(self, celosswrapper, **kw):
        super(PPLfromCE, self).__init__(celosswrapper.loss, **kw)
        self.celosswrapper = celosswrapper

    def __call__(self, pred, gold):
        return 0

    def get_epoch_error(self):
        return math.exp(self.celosswrapper.get_epoch_error())


def run(lr=30.,
        dropout=0.2,
        dropconnect=0.2,
        gradnorm=0.25,
        epochs=25,
        embdim = 400,
        encdim = 1150,
        numlayers = 3,
        seqlen=20,
        batsize=8,
        eval_batsize=80,
        cuda=False,
        gpu=0,
        test=False
        ):
    tt = q.ticktock("script")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
    tt.tick("loading data")
    train_batches, valid_batches, test_batches, D = \
        load_data(batsize=batsize, eval_batsize=eval_batsize,
                  seqlen=VariableSeqlen(minimum=5, maximum_offset=10, mu=seqlen, sigma=5))
    tt.tock("data loaded")
    print("{} batches in train".format(len(train_batches)))

    tt.tick("creating model")
    dims = [embdim] + ([encdim] * numlayers)
    m = RNNLayer_LM(*dims, worddic=D, dropout=dropout).to(device)

    if test:
        for i, batch in enumerate(train_batches):
            y = m(batch[0])
            if i > 5:
                break
        print(y.size())

    loss = q.LossWrapper(q.CELoss(mode="logits"))
    validloss = q.LossWrapper(q.CELoss(mode="logits"))
    validlosses = [validloss, PPLfromCE(validloss)]
    testloss = q.LossWrapper(q.CELoss(mode="logits"))
    testlosses = [testloss, PPLfromCE(testloss)]

    for l in [loss] + validlosses + testlosses:   # put losses on right device
        l.loss.to(device)

    optim = torch.optim.SGD(m.parameters(), lr=lr)

    train_batch_f = partial(q.train_batch, on_before_optim_step=[lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm)])
    lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=1 / 4, patience=0, verbose=True)
    lrp_f = lambda: lrp.step(validloss.get_epoch_error())

    train_epoch_f = partial(q.train_epoch, model=m, dataloader=train_batches, optim=optim, losses=[loss],
                            device=device, _train_batch=train_batch_f)
    valid_epoch_f = partial(q.test_epoch, model=m, dataloader=valid_batches, losses=validlosses, device=device,
                            on_end=lrp_f)

    tt.tock("created model")
    tt.tick("training")
    q.run_training(train_epoch_f, valid_epoch_f, max_epochs=epochs, validinter=1)
    tt.tock("trained")

    tt.tick("testing")
    q.test_epoch(model=m, dataloader=test_batches, losses=testlosses, device=device)
    tt.tock("tested")




if __name__ == '__main__':
    q.argprun(run)