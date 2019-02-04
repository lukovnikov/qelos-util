from unittest import TestCase
import qelos as q
import torch


class TestBucketRandomSampler(TestCase):
    def test_indexes(self):
        bs = q.BucketedRandomBatchSampler(500, 11, 17)
        allids = []
        a = 0
        for x in bs:
            print(len(x))
            for xe in x:
                allids.append(xe.item())
            a += 1
        print("bs len", len(bs), a)
        print(len(allids))
        self.assertTrue(len(set(range(500)) - set(allids)) == 0)    # all ids covered
        self.assertTrue(len(allids) == len(set(allids)))            # all ids unique
        self.assertTrue(len(bs) == a)


class TestPadClip(TestCase):
    def test_it(self):
        x = torch.randint(1, 5, (5, 10)).long()
        # print(x)
        x[0, 3:] = 0
        x[1, 4:] = 0
        x[2, 7:] = 0
        x[3, 0:] = 0
        x[4, 5:] = 0
        print(x)
        y = q.pad_clip(x)
        print(y)
        self.assertEqual(y.size(), (5, 7))
        self.assertTrue((x[:, :7] == y).all().item() == 1)

    def test_it_3D(self):
        x = torch.randint(1, 5, (4, 5, 10)).long()
        # print(x)
        x[:, 0, 3:] = 0
        x[:, 1, 4:] = 0
        x[:, 2, 7:] = 0
        x[:, 3, 0:] = 0
        x[:, 4, 5:] = 0
        print(x)
        y = q.pad_clip(x)
        print(y)
        self.assertEqual(y.size(), (4, 5, 7))
        self.assertTrue((x[:, :, :7] == y).all().item() == 1)

    def test_it_4D(self):
        x = torch.randint(1, 5, (3, 4, 5, 10)).long()
        # print(x)
        x[:, :, 0, 3:] = 0
        x[:, :, 1, 4:] = 0
        x[:, :, 2, 7:] = 0
        x[:, :, 3, 0:] = 0
        x[:, :, 4, 5:] = 0
        print(x)
        y = q.pad_clip(x)
        print(y)
        self.assertEqual(y.size(), (3, 4, 5, 7))
        self.assertTrue((x[:, :, :, :7] == y).all().item() == 1)


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
        self.assertTrue((x[:, :7] == y[0]).all().item() == 1)

    def test_it_3D(self):
        x = torch.randint(1, 5, (5, 1, 10)).long()
        # print(x)
        x[0, 0, 3:] = 0
        x[1, 0, 4:] = 0
        x[2, 0, 7:] = 0
        x[3, 0, 0:] = 0
        x[4, 0, 5:] = 0
        print(x)
        y = q.padclip_collate_fn(list(zip([xe.squeeze(0) for xe in x.split(1)], [xe.squeeze(0) for xe in x.split(1)])))
        print(y)
        self.assertEqual(y[0].size(), (5, 1, 7))
        self.assertTrue((x[:, :, :7] == y[0]).all().item() == 1)


class TestPackUnpack(TestCase):
    def test_pack_unpack_equal(self):
        x = torch.randn(5, 7, 3)
        mask = torch.tensor([
            [1, 1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 0],
            [1, 1, 1, 0, 0, 0, 0],
        ])
        x = x * mask.unsqueeze(2).float()
        packed_seq, unsorter = q.seq_pack(x, mask)
        print(packed_seq)
        unpacked_seq, outmask = q.seq_unpack(packed_seq, unsorter)
        print(outmask)

        print(x)
        print(unpacked_seq)
        assert(torch.allclose(x, unpacked_seq))
        assert(torch.allclose(mask, outmask))




