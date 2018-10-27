from unittest import TestCase
import qelos as q
from torch.autograd import Variable
import torch
from torch import nn
import numpy as np


class TestWordEmb(TestCase):
    def test_creation_simple(self):
        dic = dict(zip(map(chr, range(97, 122)), range(122-97)))
        m = q.WordEmb(10, worddic=dic)
        embedding, _ = m(Variable(torch.LongTensor([0,1,2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.embedding.weight.cpu().detach().numpy()[0]
        self.assertTrue(np.allclose(trueemb, embedding[0].detach().numpy()))

    def test_creation_masked(self):
        dic = dict(zip(map(chr, range(97, 122)), range(1, 122-97+1)))
        dic[q.WordEmb.masktoken] = 0
        m = q.WordEmb(10, worddic=dic)
        embedding, mask = m(Variable(torch.LongTensor([0, 1, 2])))
        self.assertEqual(embedding.size(), (3, 10))
        trueemb = m.weight.cpu().detach().numpy()[1]
        self.assertTrue(np.allclose(trueemb, embedding[1].detach().numpy()))
        self.assertTrue(np.allclose(embedding[0].detach().numpy(), np.zeros((10,))))
        print(mask)
        self.assertTrue(np.allclose(mask.detach().numpy(), [0,1,1]))


class TestGlove(TestCase):
    def setUp(self):
        path = "../data/glove/miniglove.50d"
        self.glove = q.WordEmb.load_pretrained_path(path)

    def test_loaded(self):
        thevector = self.glove.weight[self.glove.D["the"]].detach().numpy()
        truevector = np.asarray([  4.18000013e-01,   2.49679998e-01,  -4.12420005e-01,
         1.21699996e-01,   3.45270008e-01,  -4.44569997e-02,
        -4.96879995e-01,  -1.78619996e-01,  -6.60229998e-04,
        -6.56599998e-01,   2.78430015e-01,  -1.47670001e-01,
        -5.56770027e-01,   1.46579996e-01,  -9.50950012e-03,
         1.16579998e-02,   1.02040000e-01,  -1.27920002e-01,
        -8.44299972e-01,  -1.21809997e-01,  -1.68009996e-02,
        -3.32789987e-01,  -1.55200005e-01,  -2.31309995e-01,
        -1.91809997e-01,  -1.88230002e+00,  -7.67459989e-01,
         9.90509987e-02,  -4.21249986e-01,  -1.95260003e-01,
         4.00710011e+00,  -1.85939997e-01,  -5.22870004e-01,
        -3.16810012e-01,   5.92130003e-04,   7.44489999e-03,
         1.77780002e-01,  -1.58969998e-01,   1.20409997e-02,
        -5.42230010e-02,  -2.98709989e-01,  -1.57490000e-01,
        -3.47579986e-01,  -4.56370004e-02,  -4.42510009e-01,
         1.87849998e-01,   2.78489990e-03,  -1.84110001e-01,
        -1.15139998e-01,  -7.85809994e-01])
        self.assertEqual(self.glove.D["the"], 0)
        print(np.linalg.norm(thevector - truevector))
        self.assertTrue(np.allclose(thevector, truevector))
        self.assertEqual(self.glove.weight.size(), (4000, 50))

    def test_partially_loaded(self):
        D = "<MASK> <RARE> cat dog person arizonaiceteaa".split()
        D = dict(zip(D, range(len(D))))
        baseemb = q.WordEmb(dim=50, worddic=D)
        baseemb = baseemb.override(self.glove)

        q.PartiallyPretrainedWordEmb.defaultpath = "../data/glove/miniglove.%dd"

        plemb = q.PartiallyPretrainedWordEmb(dim=50, worddic=D,
                         value=baseemb.base.embedding.weight.detach().numpy(),
                         gradfracs=(1., 0.5))

        x = torch.tensor(np.asarray([0, 1, 2, 3, 4, 5]), dtype=torch.int64)
        base_out, base_mask = baseemb(x)
        pl_out, mask = plemb(x)

        self.assertTrue(np.allclose(base_out[2:].detach().numpy(), pl_out[2:].detach().numpy()))

        # test gradients
        l = pl_out.sum()
        l.backward()

        gradnorm = plemb.embedding.weight.grad.norm()
        thegrad = plemb.embedding.weight.grad

        print(gradnorm)
        self.assertTrue(np.all(thegrad.detach().numpy()[0, :] == 0))
        self.assertTrue(np.all(thegrad.detach().numpy()[[1,2,5], :] == 1.))
        self.assertTrue(np.all(thegrad.detach().numpy()[[3, 4], :] == 0.5))

        print(base_out - pl_out)