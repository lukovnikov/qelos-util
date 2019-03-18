from unittest import TestCase
import math
import numpy as np
import torch
import qelos as q
from qelos.loss import nan2zero
import random


class TestCELoss(TestCase):
    def test_it_2D(self):
        x = torch.randn(5, 4)
        g = torch.randint(0, 4, (5,)).long()
        m = q.CELoss(mode="logits")
        l = m(x, g)
        print(l)

        # reference
        logprobs = torch.nn.LogSoftmax(-1)(x)
        logprobs = torch.gather(logprobs, 1, g.unsqueeze(-1))
        lref = logprobs.mean()
        print(lref)
        self.assertTrue(l.item() == - lref.item())

    def test_it_3D(self):
        x = torch.randn(5, 3, 4)
        g = torch.randint(0, 4, (5, 3)).long()
        m = q.CELoss(mode="logits")
        l = m(x, g)
        print(l)

        # reference
        logprobs = torch.nn.LogSoftmax(-1)(x)
        logprobs = torch.gather(logprobs, -1, g.unsqueeze(-1))
        lref = logprobs.mean()
        print(lref)
        self.assertTrue(l.item() == - lref.item())

    def test_it_3D_nored(self):
        x = torch.randn(5, 3, 4)
        g = torch.randint(0, 4, (5, 3)).long()
        m = q.CELoss(mode="logits", reduction="none")
        l = m(x, g)
        print(l.size())

        # reference
        logprobs = torch.nn.LogSoftmax(-1)(x)
        logprobs = torch.gather(logprobs, -1, g.unsqueeze(-1)).squeeze(-1)
        print(logprobs.size())

        self.assertTrue(np.allclose(l.detach().numpy(), -logprobs.detach().numpy()))

    def test_it_4D_nored(self):
        x = torch.randn(5, 3, 4, 4)
        g = torch.randint(0, 4, (5, 3, 4)).long()
        m = q.CELoss(mode="logits", reduction="none")
        l = m(x, g)
        print(l.size())

        # reference
        logprobs = torch.nn.LogSoftmax(-1)(x)
        logprobs = torch.gather(logprobs, -1, g.unsqueeze(-1)).squeeze(-1)
        print(logprobs.size())

        self.assertTrue(np.allclose(l.detach().numpy(), -logprobs.detach().numpy()))

    def test_it_5D_nored(self):
        x = torch.randn(5, 3, 4, 5, 4)
        g = torch.randint(0, 4, (5, 3, 4, 5)).long()
        m = q.CELoss(mode="logits", reduction="none")
        l = m(x, g)
        print(l.size())

        # reference
        logprobs = torch.nn.LogSoftmax(-1)(x)
        logprobs = torch.gather(logprobs, -1, g.unsqueeze(-1)).squeeze(-1)
        print(logprobs.size())

        self.assertTrue(np.allclose(l.detach().numpy(), -logprobs.detach().numpy()))

    def test_it_5D_withmask(self):
        x = torch.randn(5, 3, 4, 5, 4)
        g = torch.randint(0, 4, (5, 3, 4, 5)).long()
        g[:, :, :, -1] = 0
        m = q.CELoss(mode="logits", ignore_index=0)
        l = m(x, g)
        print(l)

        # reference
        mask = (1-(g == 0).float())
        logprobs = torch.nn.LogSoftmax(-1)(x + torch.log(mask.unsqueeze(-1)))
        logprobs = - torch.gather(logprobs, -1, g.unsqueeze(-1)).squeeze(-1)
        logprobs = nan2zero(logprobs)
        s = logprobs.sum()
        t = mask.sum()
        lref = s / t
        print(lref)
        self.assertTrue((l - lref).norm(1).item() < 1e-6)


class TestSmoothedCELoss(TestCase):
    def test_it(self):
        m = q.SmoothedCELoss(smoothing=0.2, mode="logits")
        x = torch.randn(5, 6)
        g = torch.randint(0, 6, (5,)).long()
        l = m(x, g)
        print(l)

        uniform = torch.ones_like(x) / x.size(1)
        # print(uniform)
        kl = torch.nn.KLDivLoss(reduction="none")(x, uniform).sum(-1).mean()
        ce = q.CELoss(mode="logits")(x, g)
        print(kl, ce)
        print(kl*0.2 + ce*0.8)

    def test_it_with_weights(self):
        weights = torch.tensor([0.1, 0.2, 0.3, 1., 1., 1.])
        m = q.SmoothedCELoss(smoothing=0.2, mode="logits", weight=weights)
        x = torch.randn(5, 6)
        g = torch.randint(0, 6, (5,)).long()
        l = m(x, g)
        print(l)

        uniform = torch.ones_like(x) / x.size(1)
        # print(uniform)
        kl = torch.nn.KLDivLoss(reduction="none")(x, uniform).sum(-1).mean()
        ce = q.CELoss(mode="logits")(x, g)
        print(kl, ce)
        print(kl*0.2 + ce*0.8)


class TestDiffSmoothedCELoss(TestCase):
    def test_it(self):
        m = q.DiffSmoothedCELoss()
        x = torch.randn(5, 6)
        g = torch.randint(0, 6, (5,)).long()
        l = m(x, g)
        print(l)


class TestDistillLoss(TestCase):
    def test_nan2zero(self):
        x = torch.randn(5)
        x[0] = np.nan
        x.requires_grad = True
        y = nan2zero(x)
        l = y.sum()
        l.backward()
        print("backwarded")
        print(x.grad)

        try:
            x = torch.randn(5)
            x[0] = np.nan
            x.requires_grad = True
            x[x != x] = 0
            l = x.sum()
            l.backward()
        except Exception as e:
            print("didn't backward")
        #
        # x = torch.randn(5)
        # x[0] = np.nan
        # x.requires_grad = True
        # y = torch.zeros_like(x)
        # y[x == x] = x
        # l = y.sum()
        # l.backward()

    def test_it(self):
        m = q.DistillLoss(temperature=2., ignore_index=0)
        probs = torch.randn(2, 3, 4)
        softgold = torch.randn(2, 3, 4)
        hardgold = torch.randint(1, 4, (2, 3)).to(torch.int64)
        hardgold[0, 1] = random.choice((1, 2))
        hardgold[:, -1] = 0
        softgold[0, 1, 1:3] = -np.infty
        probs[0, -1, 0] = -np.infty
        probs[0, 1, [2, 3]] = -np.infty
        l = m(probs, (softgold, hardgold))
        print(l)
        if hardgold[0, 1].item() in (2, 3):
            print("l is infty")
            self.assertTrue(l.item() == np.infty)
        else:
            print("l is not infty")
            self.assertFalse(l.item() == np.infty)

    def test_equivalent_to_kl(self):
        m = q.DistillLoss(temperature=1., ignore_index=-100, mixture=1)
        probs = torch.randn(2, 3, 4)
        softgold = torch.randn(2, 3, 4)
        hardgold = torch.randint(1, 4, (2, 3)).long()
        l = m(probs, (softgold, hardgold))
        print(l)

        # reference
        kls = -torch.nn.Softmax(-1)(softgold) * (torch.nn.LogSoftmax(-1)(probs) - torch.nn.LogSoftmax(-1)(softgold))
        kl = kls.sum(-1).mean()
        print(kl)

        print(l.item() - kl.item())
        self.assertTrue((l - kl).norm(1).item() < 1e-6)

    def test_equivalent_to_ce(self):
        m = q.DistillLoss(temperature=1., ignore_index=-100, mixture=0)
        probs = torch.randn(2, 3, 4)
        softgold = torch.randn(2, 3, 4)
        hardgold = torch.randint(1, 4, (2, 3)).long()
        l = m(probs, (softgold, hardgold))
        print(l)

        # reference
        ce = q.CELoss(mode="logits")(probs, hardgold)
        print(ce)

        print(l.item() - ce.item())
        self.assertTrue((l - ce).norm(1).item() < 1e-6)




