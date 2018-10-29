import qelos as q
import torch
import numpy as np


__all__ = ["batch_reset", "epoch_reset"]


def batch_reset(module):        # performs all resetting operations on module before using it in the next batch
    for modu in module.modules():
        if hasattr(modu, "batch_reset"):
            modu.batch_reset()


def epoch_reset(module):        # performs all resetting operations on module before using it in the next epoch
    for modu in module.modules():
        if hasattr(modu, "epoch_reset"):
            modu.epoch_reset()