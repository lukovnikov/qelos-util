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

    def __init__(self, *dims:int, worddic:dict=None, bias:bool=True, tieweights=False,
                 dropout:float=0., dropouti:float=0., dropouth:float=0., dropoute:float=0., **kw):
        super(RNNLayer_LM, self).__init__(**kw)
        self.dims = dims
        self.D = worddic
        self.states = None
        # make layers
        self.emb = q.WordEmb(dims[0], worddic=self.D)
        self.out = q.WordLinout(dims[-1], worddic=self.D)
        if tieweights:
            self.out.weight = self.emb.weight
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


class GloveGoldGetter(torch.nn.Module):
    """ Compute similarities between all words in worddic based on dim-dimensional pretrained glove embeddings """
    def __init__(self, path="../../../data/glove/glove.50d", worddic=None, docosine=False):
        super(GloveGoldGetter, self).__init__()
        self.emb = q.WordEmb.load_pretrained_path(path, selectD=worddic)
        # compute similarities based on glove vectors between every word in worddic
        # - do dots
        with torch.no_grad():
            sims = torch.einsum("ai,bi->ab", (self.emb.weight, self.emb.weight))
            print(sims.min(), sims.max())
            # - do cosine
            if docosine:
                norms = self.emb.weight.norm(2, 1, keepdim=True).clamp(1e-6, np.infty)
                sims = (sims / norms) / norms.t()
                sims = ((sims - sims.min()) / (sims.max() - sims.min())) * 2. - 1.
                print(sims.min(), sims.max())
                sims = torch.log(sims * 0.5 + 0.5)      # spread out -1 to 1 --> -infty, 0
            # - do mask (set word ids not in self.emb.D to -infty
            revD = {v: k for k, v in self.emb.D.items()}
            oov_count = 0
            for i in range(sims.size(0)):
                if i not in revD:
                    sims[:, i] = -np.infty
                    sims[i, :] = -np.infty
                    sims[i, i] = 0
                    oov_count += 1
            print("{:.2f}% ({}/{}) words not in glove dic".format((100*oov_count / sims.size(0)), oov_count, sims.size(0)))
        self.register_buffer("sims", sims)
        # self.sims = sims

    def __call__(self, x):
        """
        :param x:   (batsize, seqlen) int ids of gold, torch tensor
        """
        ret = self.sims[x]
        return ret


def test_glove_gold_getter():
    D = "cat dog pup person the a child kid icetea arizonagreen".split()
    D = dict(zip(D, range(len(D))))
    m = GloveGoldGetter(path="../../../data/glove/miniglove.50d", worddic=D)
    print(m)
    x = torch.randint(0, 9, (5, 4)).long()
    y = m(x)
    print(y.size())


def run(lr=20.,
        dropout=0.2,
        dropconnect=0.2,
        gradnorm=0.25,
        epochs=25,
        embdim=200,
        encdim=200,
        numlayers=2,
        tieweights=False,
        distill="glove",        # "rnnlm", "glove"
        seqlen=35,
        batsize=20,
        eval_batsize=80,
        cuda=False,
        gpu=0,
        test=False,
        repretrain=False,       # retrain base model instead of loading it
        savepath="rnnlm.base.pt",        # where to save after training
        glovepath="../../../data/glove/glove.300d"
        ):
    tt = q.ticktock("script")
    device = torch.device("cpu")
    if cuda:
        device = torch.device("cuda", gpu)
    tt.tick("loading data")
    train_batches, valid_batches, test_batches, D = \
        load_data(batsize=batsize, eval_batsize=eval_batsize,
                  seqlen=VariableSeqlen(minimum=5, maximum_offset=10, mu=seqlen, sigma=0))
    tt.tock("data loaded")
    print("{} batches in train".format(len(train_batches)))

    # region base training
    loss = q.LossWrapper(q.CELoss(mode="logits"))
    validloss = q.LossWrapper(q.CELoss(mode="logits"))
    validlosses = [validloss, PPLfromCE(validloss)]
    testloss = q.LossWrapper(q.CELoss(mode="logits"))
    testlosses = [testloss, PPLfromCE(testloss)]

    for l in [loss] + validlosses + testlosses:   # put losses on right device
        l.loss.to(device)

    if os.path.exists(savepath) and repretrain is False:
        tt.tick("reloading base model")
        with open(savepath, "rb") as f:
            m = torch.load(f)
            m.to(device)
        tt.tock("reloaded base model")
    else:
        tt.tick("preparing training base")
        dims = [embdim] + ([encdim] * numlayers)

        m = RNNLayer_LM(*dims, worddic=D, dropout=dropout, tieweights=tieweights).to(device)

        if test:
            for i, batch in enumerate(train_batches):
                y = m(batch[0])
                if i > 5:
                    break
            print(y.size())

        optim = torch.optim.SGD(m.parameters(), lr=lr)

        train_batch_f = partial(q.train_batch, on_before_optim_step=[lambda: torch.nn.utils.clip_grad_norm_(m.parameters(), gradnorm)])
        lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=1 / 4, patience=0, verbose=True)
        lrp_f = lambda: lrp.step(validloss.get_epoch_error())

        train_epoch_f = partial(q.train_epoch, model=m, dataloader=train_batches, optim=optim, losses=[loss],
                                device=device, _train_batch=train_batch_f)
        valid_epoch_f = partial(q.test_epoch, model=m, dataloader=valid_batches, losses=validlosses, device=device,
                                on_end=[lrp_f])

        tt.tock("prepared training base")
        tt.tick("training base model")
        q.run_training(train_epoch_f, valid_epoch_f, max_epochs=epochs, validinter=1)
        tt.tock("trained base model")

        with open(savepath, "wb") as f:
            torch.save(m, f)

    tt.tick("testing base model")
    testresults = q.test_epoch(model=m, dataloader=test_batches, losses=testlosses, device=device)
    print(testresults)
    tt.tock("tested base model")
    # endregion

    # region distillation
    tt.tick("preparing training student")
    dims = [embdim] + ([encdim] * numlayers)
    ms = RNNLayer_LM(*dims, worddic=D, dropout=dropout, tieweights=tieweights).to(device)

    loss = q.LossWrapper(q.DistillLoss(temperature=2.))
    validloss = q.LossWrapper(q.CELoss(mode="logits"))
    validlosses = [validloss, PPLfromCE(validloss)]
    testloss = q.LossWrapper(q.CELoss(mode="logits"))
    testlosses = [testloss, PPLfromCE(testloss)]

    for l in [loss] + validlosses + testlosses:  # put losses on right device
        l.loss.to(device)

    optim = torch.optim.SGD(ms.parameters(), lr=lr)

    train_batch_f = partial(train_batch_distill,
                            on_before_optim_step=[lambda: torch.nn.utils.clip_grad_norm_(ms.parameters(), gradnorm)])
    lrp = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=1 / 4, patience=0, verbose=True)
    lrp_f = lambda: lrp.step(validloss.get_epoch_error())

    if distill == "rnnlm":
        mbase = m
        goldgetter = None
    elif distill == "glove":
        mbase = None
        tt.tick("creating gold getter based on glove")
        goldgetter = GloveGoldGetter(glovepath, worddic=D)
        goldgetter.to(device)
        tt.tock("created gold getter")
    else:   raise q.SumTingWongException("unknown distill mode {}".format(distill))

    train_epoch_f = partial(train_epoch_distill, model=ms, dataloader=train_batches, optim=optim, losses=[loss],
                            device=device, _train_batch=train_batch_f, mbase=mbase, goldgetter=goldgetter)
    valid_epoch_f = partial(q.test_epoch, model=ms, dataloader=valid_batches, losses=validlosses, device=device,
                            on_end=[lrp_f])

    tt.tock("prepared training student")
    tt.tick("training student model")
    q.run_training(train_epoch_f, valid_epoch_f, max_epochs=epochs, validinter=1)
    tt.tock("trained student model")

    tt.tick("testing student model")
    testresults = q.test_epoch(model=ms, dataloader=test_batches, losses=testlosses, device=device)
    print(testresults)
    tt.tock("tested student model")
    # endregion


def train_batch_distill(batch=None, model=None, optim=None, losses=None, device=torch.device("cpu"),
                batch_number=-1, max_batches=0, current_epoch=0, max_epochs=0,
                on_start=tuple(), on_before_optim_step=tuple(), on_after_optim_step=tuple(), on_end=tuple(), run=False,
                mbase=None, goldgetter=None):
    """
    Runs a single batch of SGD on provided batch and settings.
    :param _batch:  batch to run on
    :param model:   torch.nn.Module of the model
    :param optim:       torch optimizer
    :param losses:      list of losswrappers
    :param device:      device
    :param batch_number:    which batch
    :param max_batches:     total number of batches
    :param current_epoch:   current epoch
    :param max_epochs:      total number of epochs
    :param on_start:        collection of functions to call when starting training batch
    :param on_before_optim_step:    collection of functions for before optimization step is taken (gradclip)
    :param on_after_optim_step:     collection of functions for after optimization step is taken
    :param on_end:              collection of functions to call when batch is done
    :param mbase:           base model where to distill from. takes inputs and produces output distributions to match by student model. if goldgetter is specified, this is not used.
    :param goldgetter:      takes the gold and produces a softgold
    :return:
    """
    # if run is False:
    #     kwargs = locals().copy()
    #     return partial(train_batch, **kwargs)

    [e() for e in on_start]
    optim.zero_grad()
    model.train()

    batch = (batch,) if not q.issequence(batch) else batch
    batch = q.recmap(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

    batch_in = batch[:-1]
    gold = batch[-1]

    # run batch_in through teacher model to get teacher output distributions
    if goldgetter is not None:
        softgold = goldgetter(gold)
    elif mbase is not None:
        mbase.eval()
        q.batch_reset(mbase)
        with torch.no_grad():
            softgold = mbase(*batch_in)
    else:
        raise q.SumTingWongException("goldgetter and mbase can not both be None")

    q.batch_reset(model)
    modelouts = model(*batch_in)

    trainlosses = []
    for loss_obj in losses:
        loss_val = loss_obj(modelouts, (softgold, gold))
        loss_val = [loss_val] if not q.issequence(loss_val) else loss_val
        trainlosses.extend(loss_val)

    cost = trainlosses[0]
    cost.backward()

    [e() for e in on_before_optim_step]
    optim.step()
    [e() for e in on_after_optim_step]

    ttmsg = "train - Epoch {}/{} - [{}/{}]: {}".format(
                current_epoch+1,
                max_epochs,
                batch_number+1,
                max_batches,
                q.pp_epoch_losses(*losses),
                )

    [e() for e in on_end]
    return ttmsg


def train_epoch_distill(model=None, dataloader=None, optim=None, losses=None, device=torch.device("cpu"), tt=q.ticktock("-"),
             current_epoch=0, max_epochs=0, _train_batch=train_batch_distill, on_start=tuple(), on_end=tuple(), run=False,
             mbase=None, goldgetter=None):
    """
    Performs an epoch of training on given model, with data from given dataloader, using given optimizer,
    with loss computed based on given losses.
    :param model:
    :param dataloader:
    :param optim:
    :param losses:  list of loss wrappers
    :param device:  device to put batches on
    :param tt:
    :param current_epoch:
    :param max_epochs:
    :param _train_batch:    train batch function, default is train_batch
    :param on_start:
    :param on_end:
    :return:
    """
    # if run is False:
    #     kwargs = locals().copy()
    #     return partial(train_epoch, **kwargs)

    for loss in losses:
        loss.push_epoch_to_history(epoch=current_epoch-1)
        loss.reset_agg()

    [e() for e in on_start]

    q.epoch_reset(model)
    if mbase is not None:
        q.epoch_reset(mbase)

    for i, _batch in enumerate(dataloader):
        ttmsg = _train_batch(batch=_batch, model=model, optim=optim, losses=losses, device=device,
                             batch_number=i, max_batches=len(dataloader), current_epoch=current_epoch, max_epochs=max_epochs,
                             run=True, mbase=mbase, goldgetter=goldgetter)
        tt.live(ttmsg)

    tt.stoplive()
    [e() for e in on_end]
    ttmsg = q.pp_epoch_losses(*losses)
    return ttmsg


if __name__ == '__main__':
    # test_glove_gold_getter()
    q.argprun(run)