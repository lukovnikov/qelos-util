from unittest import TestCase
import qelos as q
import torch


class TestBucketRandomSampler(TestCase):
    def test_indexes(self):
        bs = q.BucketedRandomSampler(500, 11, 17)
        allids = []
        for x in bs:
            print(len(x))
            for xe in x:
                allids.append(xe.item())
        print(len(allids))
        self.assertTrue(len(set(range(500)) - set(allids)) == 0)    # all ids covered
        self.assertTrue(len(allids) == len(set(allids)))            # all ids unique


class TestPadclipCollate(TestCase):
    def test_it(self):
        x = torch.randint(1, 5, (5, 10)).long()
        # print(x)
        x[0, 3:] = 0
        x[1, 4:] = 0
        x[2, 7:] = 0
        x[3, 0:] = 0
        x[4, 5:] = 0
        print(x)
        y = q.padclip_collate_fn(list(zip([xe.squeeze(0) for xe in x.split(1)], [xe.squeeze(0) for xe in x.split(1)])))
        print(y)
        self.assertEqual(y[0].size(), (5, 7))

    def test_it_3D(self):
        x = torch.randint(1, 5, (5, 1, 10)).long()
        # print(x)
        x[0, 0, 3:] = 0
        x[1, 0, 4:] = 0
        x[2, 0, 7:] = 0
        x[3, 0, 0:] = 0
        x[4, 0, 5:] = 0
        print(x)
        with self.assertRaises(q.SumTingWongException):
            y = q.padclip_collate_fn(list(zip([xe.squeeze(0) for xe in x.split(1)], [xe.squeeze(0) for xe in x.split(1)])))
        # print(y)
        # self.assertEqual(y[0].size(), (5, 1, 7))



