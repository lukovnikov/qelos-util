from unittest import TestCase
import torch
import qelos as q
from qelos.transformer import *
from qelos.other.transformer_huggingface import Attention as HF_Attention
import numpy as np


class TestAttention(TestCase):
    def test_it(self):
        x = torch.randn(2, 3, 12)
        x_ref = x + 0.
        x.requires_grad = True
        x_ref.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False, scale=True)
        m_ref = HF_Attention(12, 3, cfg_n_head=numheads, scale=True)
        # m.qkv_proj = m_ref.c_attn
        # m.vw_proj = m_ref.c_proj
        m_ref.c_attn.w = torch.nn.Parameter(
            torch.cat([m.q_proj.weight.t(),
                       m.k_proj.weight.t(),
                       m.v_proj.weight.t()], 1)
        )
        m_ref.c_proj.w = torch.nn.Parameter(m.vw_proj.weight.t())
        m_ref.c_attn.b = torch.nn.Parameter(
            torch.cat([m.q_proj.bias,
                       m.k_proj.bias,
                       m.v_proj.bias], 0)
        )
        m_ref.c_proj.b = m.vw_proj.bias

        y = m(x)
        y_ref, _ = m_ref(x_ref)

        # print(_q.size(), _qref.size())
        # print((_q - _qref).norm())
        # print("done")

        # print(_x.size(), _x_ref.size())
        # print((_x - _x_ref).norm())
        #
        # print(vw.size(), vw_ref.size())
        # print((vw - vw_ref).norm())

        # y, _x, q, k, v, w_a, w_b = m(x)
        # y_ref, _x_ref, _q, _k, _v, _w_a, _w_b = m_ref(x)
        #
        # print((_x - _x_ref).norm())
        # print((q - _q.transpose(1, 2)).norm())
        # print((k - _k.permute(0, 3, 1, 2)).norm())
        # print((v - _v.transpose(1, 2)).norm())
        # print((w_a - _w_a).norm())
        # print((w_b - _w_b).norm())
        print(y.size(), y_ref.size())
        print((y - y_ref).norm())
        self.assertTrue((np.allclose(y.detach().numpy(), y_ref.detach().numpy(), atol=1e-5)))
        print("outputs good")

        y.sum().backward()
        y_ref.sum().backward()

        print((x.grad - x_ref.grad).norm())
        self.assertTrue((np.allclose(x.grad.detach().numpy(), x_ref.grad.detach().numpy(), atol=1e-5)))
        print("grads good")

    def test_mask(self):
        x = torch.randn(4, 4, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads)
        # mask = None
        mask = torch.tensor([[1,1,1,0],[1,0,0,0],[1,1,1,1],[1,0,1,0]])

        y = m(x, mask=mask)
        print(y.size())
        l = y.sum()
        l.backward()
        print(x.grad[0, :, :].norm(1, dim=1))
        print(x.grad.norm(1, dim=2))


class TestAttentionCell(TestCase):
    def test_it(self):
        x = torch.randn(4, 5, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        for i in range(x.size(1)):
            y = mc(x[:, i].unsqueeze(1))
            print(y.size())
            ys.append(y)

        ys = torch.cat(ys, 1)
        l = ys.sum()
        l.backward()
        xgrad = x.grad
        print(xgrad.norm(1, 2))
        m.zero_grad()
        mc.zero_grad()

        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True
        ys_ref = m(x)
        l = ys_ref.sum()
        l.backward()
        print(x.grad.norm(1, 2))

        print((ys - ys_ref).norm())
        self.assertTrue(np.allclose(ys.detach().numpy(), ys_ref.detach().numpy(), atol=1e-6))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), x.grad.detach().numpy(), atol=1e-6))

    def tst_it_relpos_out_of_horizon_debug(self):
        seqlen = 4
        horizon = 3
        x = torch.randn(3, seqlen, 12)
        x.requires_grad = True

        m = MultiHeadAttention(12, numheads=4, bidir=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True, horizon=horizon)

        y, k = m(x[:, 1:])
        y.norm(1).backward()

        xgrad = x.grad
        # print(ys_ref.norm(1, 2))
        # print(xgrad.norm(1, 2))

        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        ks = []
        for i in range(x.size(1)):
            ys_, k_ = mc(x[:, i].unsqueeze(1))
            ys.append(ys_)
            ks.append(k_)
        ys = torch.cat(ys, 1)
        # print(ys.norm(1, 2))

        print(y.norm(1, 2))
        print(ys.norm(1, 2))

        print(k.size())
        print(k.norm(1, 2))
        print(ks[-1].size())
        print(ks[-1].norm(1, 2))

    def test_it_relpos_out_of_horizon(self):
        seqlen = 7
        horizon = 7
        x = torch.randn(3, seqlen, 12)
        x.requires_grad = True

        m = MultiHeadAttention(12, numheads=4, bidir=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        allys = []
        for i in range(horizon, seqlen+1):
            y = m(x[:, i-horizon:i])
            allys.append(y)
            ys.append(y[:, -1].unsqueeze(1))
        ys_ref = torch.cat(ys, 1)
        print(ys_ref.size())
        ys_ref.norm(1).backward()
        allys = allys[0]

        xgrad = x.grad
        print(ys_ref.norm(1, 2))
        # print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(seqlen):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.size())
        _allys = ys[:, :horizon]
        ys = ys[:, -(seqlen-horizon+1):]
        print(ys.norm(1, 2))
        ys.norm(1).backward()
        xsgrad = x.grad
        # print(xsgrad.norm(1, 2))

        # print("ALL YS")
        # print(allys.norm(1,2))
        # print( "___")
        # print(_allys.norm(1, 2))

    def test_it_relpos(self):
        x = torch.randn(4, 5, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False, relpos=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        for i in range(x.size(1)):
            y = mc(x[:, i].unsqueeze(1))
            print(y.size())
            ys.append(y)

        ys = torch.cat(ys, 1)
        l = ys.sum()
        l.backward()
        xgrad = x.grad
        print(xgrad.norm(1, 2))
        m.zero_grad()
        mc.zero_grad()

        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True
        ys_ref = m(x)
        l = ys_ref.sum()
        l.backward()
        print(x.grad.norm(1, 2))

        print("norm of out diff")
        print((ys - ys_ref).norm())
        print("norm of grad diff")
        print((xgrad - x.grad).norm())

        self.assertTrue(np.allclose(ys.detach().numpy(), ys_ref.detach().numpy(), atol=1e-6))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), x.grad.detach().numpy(), atol=1e-6))

    def test_it_window(self):
        x = torch.randn(4, 5, 12)
        x.requires_grad = True
        numheads = 6
        m = MultiHeadAttention(12, numheads=numheads, bidir=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        for i in range(x.size(1)):
            y = mc(x[:, i].unsqueeze(1))
            print(y.size())
            ys.append(y)

        l = y.sum()
        l.backward(retain_graph=True)
        # TODO: check that outside window, grad on x is zero
        x.grad = None

        ys = torch.cat(ys, 1)
        l = ys[:, 2].sum()
        l.backward()
        xgrad = x.grad
        print(xgrad.norm(1, 2))

        m.zero_grad()
        mc.zero_grad()

        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True
        ys_ref = m(x)
        l = ys_ref[:, 2].sum()
        l.backward()
        print(x.grad.norm(1, 2))

        self.assertTrue(np.allclose(xgrad.detach().numpy(), x.grad.detach().numpy()))



class TestEncoderBlock(TestCase):
    def test_it(self):
        indim = 10
        kdim = 12
        vdim = 15
        numheads = 3
        m = TransformerEncoderBlock(indim, kdim=kdim, vdim=vdim, bidir=True,
                         numheads=numheads, activation=q.GeLU)

        batsize = 4
        seqlen = 5

        x = torch.randn(batsize, seqlen, indim)
        x.requires_grad = True
        xmask = torch.ones_like(x[:, :, 0]).byte()
        xmask[:, -1] = 0
        xmask[0, -1] = 1
        xmask[1, -2] = 0
        print(xmask)

        #
        # y = m(x)
        y = m(x, mask=xmask)

        print(y.size())
        l = y.norm(1)
        l.backward()

        print(x.grad.norm(1, 2))

    def test_set_bidir(self):
        indim = 10
        kdim = 12
        vdim = 15
        numheads = 3
        m = TransformerEncoderBlock(indim, kdim=kdim, vdim=vdim, bidir=True,
                         numheads=numheads, activation=q.GeLU)
        q.MultiHeadAttention.set_bidir(m, False)


class TestDecoderBlock(TestCase):
    def test_it(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True

        m = TransformerDecoderBlock(12, numheads=4, noctx=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        y = m(x)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_deps(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True

        m = TransformerDecoderBlock(12, numheads=4, noctx=True)
        y = m(x)

        y[:, -2].norm(1).backward(retain_graph=True)

        print(x.grad)

    def test_it_relpos(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True

        m = TransformerDecoderBlock(12, numheads=4, noctx=True, relpos=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        y = m(x)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_it_relpos_out_of_horizon(self):
        seqlen = 7
        horizon = 7
        x = torch.randn(3, seqlen, 12)
        x.requires_grad = True

        m = TransformerDecoderBlock(12, numheads=4, bidir=False, noctx=True, relpos=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        allys = []
        for i in range(horizon, seqlen+1):
            y = m(x[:, i-horizon:i])
            allys.append(y)
            ys.append(y[:, -1].unsqueeze(1))
        ys_ref = torch.cat(ys, 1)
        print(ys_ref.size())
        ys_ref.norm(1).backward()
        allys = allys[0]

        xgrad = x.grad
        print(ys_ref.norm(1, 2))
        # print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(seqlen):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.size())
        _allys = ys[:, :horizon]
        ys = ys[:, -(seqlen-horizon+1):]
        print(ys.norm(1, 2))
        ys.norm(1).backward()
        xsgrad = x.grad
        # print(xsgrad.norm(1, 2))

        # print("ALL YS")
        # print(allys.norm(1,2))
        # print( "___")
        # print(_allys.norm(1, 2))

class TestDecoderTransformer(TestCase):
    def test_it(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True

        m = TransformerDecoder(12, numheads=4, numlayers=2, noctx=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        y = m(x)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_its_mask(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True
        xmask = torch.tensor([[1,1,0,1],[1,0,1,0],[1,1,0,0]])

        m = TransformerDecoder(12, numheads=4, numlayers=2, noctx=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        y = m(x, mask=xmask)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))
        print(xmask)
        self.assertTrue(np.all(((xmask == 0) == (xgrad.norm(1, 2) == 0)).detach().numpy()))
        # return

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1), mask=xmask[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_it_relpos(self):
        x = torch.randn(3, 5, 12)
        x.requires_grad = True

        m = TransformerDecoder(12, numheads=4, numlayers=2, noctx=True, relpos=True)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        y = m(x)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_it_relpos_out_of_horizon(self):
        seqlen = 7 #10
        horizon = 7
        x = torch.randn(3, seqlen, 12)
        x.requires_grad = True

        m = TransformerDecoder(12, numheads=4, numlayers=2, noctx=True, relpos=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        ys = []
        allys = []
        for i in range(horizon, seqlen+1):
            y = m(x[:, i-horizon:i])
            allys.append(y)
            ys.append(y[:, -1].unsqueeze(1))
        ys_ref = torch.cat(ys, 1)
        print(ys_ref.size())
        ys_ref.norm(1).backward()
        allys = allys[0]

        xgrad = x.grad
        print(ys_ref.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(seqlen):
            ys.append(mc(x[:, i].unsqueeze(1)))
        ys = torch.cat(ys, 1)
        print(ys.size())
        _allys = ys[:, :horizon]
        ys = ys[:, -(seqlen-horizon+1):]
        print(ys.norm(1, 2))
        ys.norm(1).backward()
        xsgrad = x.grad
        print(xsgrad.norm(1, 2))

        print("ALL YS")
        print(allys.norm(1,2))
        print( "___")
        print(_allys.norm(1, 2))

        self.assertTrue(np.allclose(ys_ref.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_it_with_ctx(self):
        x = torch.randn(3, 4, 12)
        ctx = torch.randn(3, 5, 12)
        x.requires_grad = True

        m = TransformerDecoder(12, numheads=4, numlayers=2, noctx=False)
        mc = q.deep_copy(m)
        mc.set_cell_mode(True)
        assert(m._cell_mode == False)
        assert(mc._cell_mode == True)

        y = m(x, ctx)
        y.norm(1).backward()

        xgrad = x.grad
        print(y.norm(1, 2))
        print(xgrad.norm(1, 2))

        # x = torch.randn(3, 4, 12)
        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True

        ys = []
        for i in range(x.size(1)):
            ys.append(mc(x[:, i].unsqueeze(1), ctx))
        ys = torch.cat(ys, 1)
        print(ys.norm(1, 2))
        ys.norm(1).backward()

        xsgrad = x.grad
        print(xsgrad.norm(1, 2))
        self.assertTrue(np.allclose(y.detach().numpy(), ys.detach().numpy(), atol=1e-5))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), xsgrad.detach().numpy(), atol=1e-5))

    def test_deps(self):
        x = torch.randn(3, 4, 12)
        x.requires_grad = True

        m = TransformerDecoder(12, numheads=4, numlayers=3, noctx=True)
        y = m(x)

        y[:, 1].norm(1).backward(retain_graph=True)

        print(x.grad.norm(1, 2))

        self.assertTrue(np.allclose(x.grad[:, 2:].detach().numpy(),
                                    np.zeros((3, 2, 12))))

class TestTS2S(TestCase):
    def test_it(self):
        x = torch.randn(4, 5, 12)
        y = torch.randn(4, 5, 12)
        x.requires_grad = True
        y.requires_grad = True
        numheads = 6
        m = TS2S_arg(dim=12, numlayers=2, numheads=numheads)
        z = m(x, y)
        print(z.size())
        z[:, -1].norm(1).backward()
        xgrad = x.grad
        ygrad = y.grad
        print(xgrad.norm(1, 2))
        print(ygrad.norm(1, 2))
        zref = z

        mc = q.deep_copy(m)
        mc.set_cell_mode(True)

        x = torch.tensor(x.detach().numpy() + 0.)
        x.requires_grad = True
        y = torch.tensor(y.detach().numpy() + 0.)
        y.requires_grad = True
        print("y size: ", y.size())

        zs = []
        for i in range(y.size(1)):
            z = mc(x, y[:, i].unsqueeze(1))
            print(z.size())
            zs.append(z)

        z = torch.cat(zs, 1)
        z[:, -1].norm(1).backward()
        print(x.grad.norm(1, 2))
        print(y.grad.norm(1, 2))

        print(z.norm(1, 2), zref.norm(1, 2))
        print((z-zref).norm())
        self.assertTrue(np.allclose(z.detach().numpy(), zref.detach().numpy(), atol=1e-6))

        print((x.grad - xgrad).norm(1))
        self.assertTrue(np.allclose(ygrad.detach().numpy(), y.grad.detach().numpy(), atol=1e-6))
        self.assertTrue(np.allclose(xgrad.detach().numpy(), x.grad.detach().numpy(), atol=1e-5))

