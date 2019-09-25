from unittest import TestCase
import torch
import qelos as q


class TestWarmupCosineWithHardRestartsSchedule(TestCase):
    def test_it(self):
        m = torch.nn.Linear(10, 10)
        optim = torch.optim.SGD(m.parameters(), lr=1)
        t_total = 100
        sched = q.WarmupCosineWithHardRestartsSchedule(optim, 0, t_total, cycles=3)
        for i in range(t_total):
            sched.step()
            print(optim.param_groups[0]["lr"])
