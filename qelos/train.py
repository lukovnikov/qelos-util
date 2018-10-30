import qelos as q
import torch
import numpy as np
from IPython import embed
from functools import partial


__all__ = ["batch_reset", "epoch_reset", "LossWrapper", "train_batch", "train_epoch", "test_epoch", "run_training"]


def batch_reset(module):        # performs all resetting operations on module before using it in the next batch
    for modu in module.modules():
        if hasattr(modu, "batch_reset"):
            modu.batch_reset()


def epoch_reset(module):        # performs all resetting operations on module before using it in the next epoch
    batch_reset(module)
    for modu in module.modules():
        if hasattr(modu, "epoch_reset"):
            modu.epoch_reset()


class LossWrapper(object):
    """ Wraps a normal loss with aggregating and other functionality """

    def __init__(self, loss, name=None, mode="mean", **kw):
        """
        :param loss:    actual loss class
        :param name:    name for this loss (class name by default)
        :param mode:    "mean" or "sum"
        :param kw:
        """
        super(LossWrapper, self).__init__()
        self.loss, self.aggmode = loss, mode
        self.name = name if name is not None else loss.__class__.__name__

        self.agg_history = []
        self.agg_epochs = []

        self.epoch_agg_values = []
        self.epoch_agg_sizes = []

    def get_epoch_error(self):
        """ returns the aggregated error for this epoch so far """
        if self.aggmode == "mean":
            if len(self.epoch_agg_sizes) == 0:
                ret = 0.
            else:
                total = sum(self.epoch_agg_sizes)
                fractions = [x/total for x in self.epoch_agg_sizes]
                parts = [x * y for x, y in zip(self.epoch_agg_values, fractions)]
                ret = sum(parts)
        else:
            ret = sum(self.epoch_agg_values)
        return ret

    def push_epoch_to_history(self, epoch=None):
        self.agg_history.append(self.get_epoch_error())
        if epoch is not None:
            self.agg_epochs.append(epoch)

    def __call__(self, pred, gold, **kw):
        l = self.loss(pred, gold, **kw)

        numex = pred.size(0) if not q.issequence(pred) else pred[0].size(0)
        if isinstance(l, tuple) and len(l) == 2:     # loss returns numex too
            numex = l[1]
            l = l[0]
        if isinstance(l, torch.Tensor):
            lp = l.item()
        else:
            lp = l
        self.epoch_agg_values.append(lp)
        self.epoch_agg_sizes.append(numex)
        return l

    def _reset(self):   # full reset
        self.reset_agg()
        self.agg_history = []
        self.agg_epochs = []

    def reset_agg(self):    # reset epoch stats
        self.epoch_agg_values = []
        self.epoch_agg_sizes = []


def no_gold(losses):
    all_linear = True
    some_linear = False
    for loss in losses:
        if isinstance(loss.loss, (q.LinearLoss, q.SelectedLinearLoss)):
            some_linear = True
        else:
            all_linear = False
    assert(all_linear == some_linear)
    return all_linear


def pp_epoch_losses(*losses:LossWrapper):
    values = [loss.get_epoch_error() for loss in losses]
    ret = " :: ".join("{:.4f}".format(value) for value in values)
    return ret


# region loops
def eval_loop(model, dataloader, device=torch.device("cpu")):
    tto = q.ticktock("testing")
    tto.tick("testing")
    tt = q.ticktock("-")
    totaltestbats = len(dataloader)
    model.eval()
    epoch_reset(model)
    outs = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = (batch,) if not q.issequence(batch) else batch
            batch = q.recmap(batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)

            batch_reset(model)
            modelouts = model(*batch)

            tt.live("eval - [{}/{}]"
                .format(
                i + 1,
                totaltestbats
            )
            )
            outs.append(modelouts)
    ttmsg = "eval done"
    tt.stoplive()
    tt.tock(ttmsg)
    tto.tock("tested")
    out = torch.cat(outs, 0)
    return out


def train_batch(_batch=None, model=None, optim=None, losses=None, device=torch.device("cpu"),
                batch_number=-1, max_batches=0, current_epoch=0, max_epochs=0,
                on_start=tuple(), on_before_optim_step=tuple(), on_after_optim_step=tuple(), on_end=tuple(), run=False):
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
    :return:
    """
    if run is False:
        kwargs = locals().copy()
        return partial(train_batch, **kwargs)

    [e() for e in on_start]
    optim.zero_grad()
    model.train()

    _batch = (_batch,) if not q.issequence(_batch) else _batch
    _batch = q.recmap(_batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
    batch = _batch

    if no_gold(losses):
        batch_in = batch
        gold = None
    else:
        batch_in = batch[:-1]
        gold = batch[-1]

    batch_reset(model)
    modelouts = model(*batch_in)

    trainlosses = []
    for loss_obj in losses:
        loss_val = loss_obj(modelouts, gold)
        loss_val = [loss_val] if not q.issequence(loss_val) else loss_val
        trainlosses.extend(loss_val)

    cost = trainlosses[0]

    if torch.isnan(cost).any():
        print("Cost is NaN!")
        embed()

    cost.backward()

    [e() for e in on_before_optim_step]
    optim.step()
    [e() for e in on_after_optim_step]

    ttmsg = "train - Epoch {}/{} - [{}/{}]: {}".format(
                current_epoch+1,
                max_epochs,
                batch_number+1,
                max_batches,
                pp_epoch_losses(*losses),
                )

    [e() for e in on_end]
    return ttmsg


def train_epoch(model=None, dataloader=None, optim=None, losses=None, device=torch.device("cpu"), tt=q.ticktock("-"),
             current_epoch=0, max_epochs=0, _train_batch=train_batch, on_start=tuple(), on_end=tuple(), run=False):
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
    if run is False:
        kwargs = locals().copy()
        return partial(train_epoch, **kwargs)

    for loss in losses:
        loss.push_epoch_to_history(epoch=current_epoch-1)
        loss.reset_agg()

    [e() for e in on_start]

    epoch_reset(model)

    for i, _batch in enumerate(dataloader):
        ttmsg = _train_batch(_batch=_batch, model=model, optim=optim, losses=losses, device=device,
                             batch_number=i, max_batches=len(dataloader), current_epoch=current_epoch, max_epochs=max_epochs,
                             run=True)
        tt.live(ttmsg)

    tt.stoplive()
    [e() for e in on_end]
    ttmsg = pp_epoch_losses(*losses)
    return ttmsg


def test_epoch(model=None, dataloader=None, losses=None, device=torch.device("cpu"),
            current_epoch=0, max_epochs=0,
            on_start=tuple(), on_start_batch=tuple(), on_end_batch=tuple(), on_end=tuple(), run=False):
    """
    Performs a test epoch. If run=True, runs, otherwise returns partially filled function.
    :param model:
    :param dataloader:
    :param losses:
    :param device:
    :param current_epoch:
    :param max_epochs:
    :param on_start:
    :param on_start_batch:
    :param on_end_batch:
    :param on_end:
    :return:
    """
    if run is False:
        kwargs = locals().copy()
        return partial(test_epoch, **kwargs)

    tt = q.ticktock("-")
    model.eval()
    epoch_reset(model)
    [e() for e in on_start]
    with torch.no_grad():
        for loss_obj in losses:
            loss_obj.push_epoch_to_history()
            loss_obj.reset_agg()
        for i, _batch in enumerate(dataloader):
            [e() for e in on_start_batch]

            _batch = (_batch,) if not q.issequence(_batch) else _batch
            _batch = q.recmap(_batch, lambda x: x.to(device) if isinstance(x, torch.Tensor) else x)
            batch = _batch

            if no_gold(losses):
                batch_in = batch
                gold = None
            else:
                batch_in = batch[:-1]
                gold = batch[-1]

            batch_reset(model)
            modelouts = model(*batch_in)

            testlosses = []
            for loss_obj in losses:
                loss_val = loss_obj(modelouts, gold)
                loss_val = [loss_val] if not q.issequence(loss_val) else loss_val
                testlosses.extend(loss_val)

            tt.live("test - Epoch {}/{} - [{}/{}]: {}".format(
                current_epoch + 1,
                max_epochs,
                i + 1,
                len(dataloader),
                pp_epoch_losses(*losses)
            )
            )
            [e() for e in on_end_batch]
    tt.stoplive()
    [e() for e in on_end]
    ttmsg = pp_epoch_losses(*losses)
    return ttmsg


def run_training(run_train_epoch=None, run_valid_epoch=None, max_epochs=1, validinter=1,
                 print_on_valid_only=False):
    """

    :param run_train_epoch:     function that performs an epoch of training. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param run_valid_epoch:     function that performs an epoch of testing. must accept current_epoch and max_epochs. Tip: use functools.partial
    :param max_epochs:
    :param validinter:
    :param print_on_valid_only:
    :return:
    """
    tt = q.ticktock("runner")
    validinter_count = 0
    current_epoch = 0
    stop_training = current_epoch >= max_epochs
    while stop_training is not True:
        tt.tick()
        ttmsg = run_train_epoch(current_epoch=current_epoch, max_epochs=max_epochs, run=True)
        ttmsg = "Epoch {}/{} -- {}".format(current_epoch+1, max_epochs, ttmsg)
        validepoch = False
        if run_valid_epoch is not None and validinter_count % validinter == 0:
            ttmsg_v = run_valid_epoch(current_epoch=current_epoch, max_epochs=max_epochs, run=True)
            ttmsg += " -- " + ttmsg_v
            validepoch = True
        validinter_count += 1
        if not print_on_valid_only or validepoch:
            tt.tock(ttmsg)
        current_epoch += 1
        stop_training = current_epoch >= max_epochs


# endregion
