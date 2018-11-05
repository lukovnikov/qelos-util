from unittest import TestCase
import torch
import numpy as np
import qelos as q
import math
from matplotlib import pyplot as plt


class TestCosineLRwithWarmup(TestCase):
    def test_it(self):
        m = torch.nn.Embedding(50,50)
        optim = torch.optim.Adam(m.parameters(), lr=0.001)
        sched = q.CosineLRwithWarmup(optim, cyc_len=100, warmup=100)
        x = []
        for i in range(600):
            sched.step()
            x.append(optim.param_groups[0]["lr"])
        plt.plot(x)
        plt.show()


