import torch
import qelos as q
from unittest import TestCase
import numpy as np
from pprint import PrettyPrinter

from qelos.rnn import OverriddenLSTMLayer, OverriddenGRULayer, OverriddenRNNLayer


# region TEST CELLS
class TestGRUCell(TestCase):
    def test_stateful(self):
        batsize = 3

        pp = PrettyPrinter()
        cell = q.GRUCell(9, 10, dropout_rec=.4)
        cell.h_tm1 = torch.rand(batsize, 10)
        startstate = cell.get_state()
        print(pp.pprint(startstate))

        x_0 = torch.randn(batsize, 9)
        y_0 = cell(x_0)
        state1 = cell.get_state()
        print(pp.pprint(state1))

        cell.set_state(startstate)
        y_0bis = cell(x_0)
        state1bis = cell.get_state()
        print(pp.pprint(state1bis))
        print(y_0)
        print(y_0bis)
        self.assertFalse(np.allclose(y_0.detach().numpy(), y_0bis.detach().numpy()))

        cell.set_state(state1)
        x_1 = torch.randn(batsize, 9)
        y_1 = cell(x_1)
        state2 = cell.get_state()
        print(pp.pprint(state2))

        cell.set_state(state1)
        y_1bis = cell(x_1)
        state2bis = cell.get_state()
        print(pp.pprint(state2bis))

        self.assertTrue(np.allclose(y_1.detach().numpy(), y_1bis.detach().numpy()))
        for k, v in state2.items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(np.allclose(state2[k].detach().numpy(), state2bis[k].detach().numpy()))
            else:
                self.assertTrue(state2[k] == state2bis[k])

    def test_gru_shapes(self):
        batsize = 5
        gru = q.GRUCell(9, 10)
        x_t = torch.randn(batsize, 9)
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = torch.nn.Parameter(torch.tensor(h_tm1))
        y_t = gru(x_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

    def test_dropout_rec(self):
        batsize = 5
        gru = q.GRUCell(9, 10, dropout_rec=0.5)
        x_t = torch.randn(batsize, 9)
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = torch.tensor(h_tm1)
        y_t = gru(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(gru.training, True)
        gru.train(False)
        self.assertEqual(gru.training, False)

        q.batch_reset(gru)
        pred1 = gru(x_t)
        q.batch_reset(gru)
        pred2 = gru(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        gru.train(True)
        self.assertEqual(gru.training, True)

        q.batch_reset(gru)
        pred1 = gru(x_t)
        q.batch_reset(gru)
        pred2 = gru(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_mask_t(self):
        batsize = 5
        gru = q.GRUCell(9, 10)
        x_t = torch.randn(batsize, 9)
        mask_t = torch.tensor([1, 1, 0, 1, 0])
        h_tm1 = torch.randn(1, 10)
        gru.h_0 = torch.tensor(h_tm1)
        y_t = gru(x_t, mask_t=mask_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), y_t[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[4].detach().numpy()))

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), gru.h_tm1[4].detach().numpy()))


class TestLSTM(TestCase):

    def test_stateful(self):
        batsize = 3

        pp = PrettyPrinter()
        cell = q.LSTMCell(9, 10, dropout_rec=.4)
        cell.c_tm1 = torch.rand(batsize, 10)
        cell.y_tm1 = torch.rand(batsize, 10)
        startstate = cell.get_state()
        # print(pp.pprint(startstate))

        x_0 = torch.randn(batsize, 9)
        y_0 = cell(x_0)
        state1 = cell.get_state()
        print(f"State 1:\n{pp.pprint(state1)}")

        cell.set_state(startstate)
        y_0bis = cell(x_0)
        state1bis = cell.get_state()
        # print(pp.pprint(state1bis))
        # print(y_0)
        # print(y_0bis)
        self.assertFalse(np.allclose(y_0.detach().numpy(), y_0bis.detach().numpy()))

        cell.set_state(state1)
        x_1 = torch.randn(batsize, 9)
        y_1 = cell(x_1)
        state2 = cell.get_state()
        print(pp.pprint(state2))

        cell.set_state(state1)
        y_1bis = cell(x_1)
        state2bis = cell.get_state()
        print(pp.pprint(state2bis))

        self.assertTrue(np.allclose(y_1.detach().numpy(), y_1bis.detach().numpy()))
        for k, v in state2.items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(np.allclose(state2[k].detach().numpy(), state2bis[k].detach().numpy()))
            else:
                self.assertTrue(state2[k] == state2bis[k])

    def test_lstm_shapes(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10)
        x_t = torch.randn(batsize, 9)
        c_tm1 = torch.randn(1, 10)
        y_tm1 = torch.randn(1, 10)
        lstm.c_0 = torch.tensor(c_tm1)
        lstm.y_0 = torch.tensor(y_tm1)

        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(lstm.y_tm1.detach().numpy(), y_t.detach().numpy()))

        q.batch_reset(lstm)

    def test_dropout_rec(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10, dropout_rec=0.5)
        x_t = torch.randn(batsize, 9)
        c_tm1 = torch.randn(1, 10)
        y_tm1 = torch.randn(1, 10)
        lstm.c_0 = torch.tensor(c_tm1)
        lstm.y_0 = torch.tensor(y_tm1)
        y_t = lstm(x_t)
        self.assertEqual((5, 10), y_t.detach().numpy().shape)

        self.assertEqual(lstm.training, True)
        lstm.train(False)
        self.assertEqual(lstm.training, False)

        q.batch_reset(lstm)
        pred1 = lstm(x_t)
        q.batch_reset(lstm)
        pred2 = lstm(x_t)

        self.assertTrue(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

        lstm.train(True)
        self.assertEqual(lstm.training, True)

        q.batch_reset(lstm)
        pred1 = lstm(x_t)
        q.batch_reset(lstm)
        pred2 = lstm(x_t)

        self.assertFalse(np.allclose(pred1.detach().numpy(), pred2.detach().numpy()))

    def test_mask_t(self):
        batsize = 5
        lstm = q.LSTMCell(9, 10)
        x_t = torch.randn(batsize, 9)
        mask_t = torch.tensor([1, 1, 0, 1, 0])
        c_tm1 = torch.randn(1, 10)
        h_tm1 = torch.randn(1, 10)
        lstm.c_0 = torch.tensor(c_tm1)
        lstm.y_0 = torch.tensor(h_tm1)
        y_t = lstm(x_t, mask_t=mask_t)
        self.assertEqual((batsize, 10), y_t.detach().numpy().shape)

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), y_t[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), y_t[4].detach().numpy()))

        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[2].detach().numpy()))
        self.assertFalse(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[1].detach().numpy()))
        self.assertTrue(np.allclose(h_tm1[0].detach().numpy(), lstm.y_tm1[4].detach().numpy()))


class TestLSTMCellEncoder(TestCase):
    encodertype = q.LSTMCellEncoder

    def test_it_without_gate_or_mask(self):
        batsize = 5
        seqlen = 6
        dim = 20
        outdim = 16

        lstm = self.encodertype(dim, outdim, outdim, bidir=True)

        x = torch.randn(batsize, seqlen, dim)
        x.requires_grad = True
        y_all, y_final = lstm(x, ret_states=True)
        print(y_final.size(), y_all.size())
        self.assertTrue(y_final.size() == (batsize, 2, outdim))
        self.assertTrue(y_all.size() == (batsize, seqlen, outdim * 2))

    def test_it_with_mask(self):
        batsize = 5
        seqlen = 6
        dim = 20
        outdim = 16

        lstm = self.encodertype(dim, outdim, bidir=True)

        x = torch.randn(batsize, seqlen, dim)
        mask = torch.ones(batsize, seqlen)
        mask[0, 3:] = 0
        mask[1, 4:] = 0
        mask[2, 1:] = 0
        mask[3, 0:] = 0
        print("mask: ", mask)
        x.requires_grad = True
        y_all, y_final = lstm(x, mask=mask, ret_states=True)
        # print(y_all[:, :, [0, -1]])

        # tests that final state is computed correctly given forward and backward directions
        self.assertTrue(np.allclose(y_final.cpu().detach().numpy()[0, 0],
                                    y_all.cpu().detach().numpy()[0, 3, :outdim]))
        self.assertTrue(np.allclose(y_final.cpu().detach().numpy()[0, 1],
                                    y_all.cpu().detach().numpy()[0, 0, outdim:]))

        self.assertTrue(np.allclose(y_final.cpu().detach().numpy()[1, 0],
                                    y_all.cpu().detach().numpy()[1, 4, :outdim]))
        self.assertTrue(np.allclose(y_final.cpu().detach().numpy()[1, 1],
                                    y_all.cpu().detach().numpy()[1, 0, outdim:]))

        # grad test: check that gradient is only on non-masked x vectors
        l = y_all.sum()
        l.backward(retain_graph=True)
        # print(x.grad)
        self.assertTrue(np.allclose((x.grad * mask.unsqueeze(2)).cpu().detach().numpy(),
                                    x.grad.cpu().detach().numpy()))
        print("total grads ok")

        x.grad = None
        # grad test: position-specific grad test
        # check grad from backward at t=1 only depends on t>=1
        l = y_all[0, 1, outdim:].sum()
        l.backward(retain_graph=True)
        # print(x.grad[0])
        self.assertTrue(np.allclose(np.zeros_like(x.grad[0, 0].cpu().detach().numpy()),
                                    x.grad[0, 0].cpu().detach().numpy()))
        x.grad = None
        l = y_all[0, 1, :outdim].sum()
        l.backward()
        # print(x.grad[0])
        self.assertTrue(np.allclose(np.zeros_like(x.grad[0, 2:].cpu().detach().numpy()),
                                    x.grad[0, 2:].cpu().detach().numpy()))

        print("fine grads ok")

    def test_it_with_dropout_and_mask(self):
        batsize = 5
        seqlen = 6
        dim = 20
        outdim = 16

        lstm = self.encodertype(dim, outdim, outdim, bidir=True, dropout_in=0.2, dropout_rec=0.2)

        x = torch.randn(batsize, seqlen, dim)
        mask = torch.ones(batsize, seqlen)
        mask[0, 3:] = 0
        mask[1, 4:] = 0
        mask[2, 1:] = 0
        mask[3, 0:] = 0
        print("mask: ", mask)
        x.requires_grad = True
        y_all, y_final = lstm(x, mask=mask, ret_states=True)

        self.assertTrue(y_final.size() == (batsize, 2, outdim))
        self.assertTrue(y_all.size() == (batsize, seqlen, outdim * 2))


class TestGRUCellEncoder(TestLSTMCellEncoder):
    encodertype = q.GRUCellEncoder


class TestRNNCellEncoder(TestLSTMCellEncoder):
    encodertype = q.RNNCellEncoder


# endregion

# region TEST RNN LAYER ENCODERS

class TestRNNEncoder(TestCase):
    encodertype = q.RNNEncoder
    _rnnlayertype = torch.nn.RNN
    _rnnlayertype_override = OverriddenRNNLayer

    def test_it(self):
        batsize = 5
        seqlen = 4
        lstm = self.encodertype(20, 26, 30)

        x = torch.randn(batsize, seqlen, 20)
        x.requires_grad = True

        y, y_T = lstm(x, ret_states=True)
        self.assertEqual((batsize, seqlen, 30), y.detach().numpy().shape)
        self.assertEqual((batsize, 1, 30), y_T.size())

        # test grad
        l = y[2, :, :].sum()
        l.backward()

        xgrad = x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[:, 0, :7])

        # test final state
        self.assertTrue(np.allclose(y_T.squeeze(1).detach().numpy(), y[:, -1].detach().numpy()))

    def test_with_mask(self):
        batsize = 3
        seqlen = 4
        lstm = self.encodertype(8, 9, 10)

        x = torch.randn(batsize, seqlen, 8)
        x.requires_grad = True
        x_mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0], [1, 1, 0, 0]], dtype=torch.int64)

        y, states = lstm(x, mask=x_mask, ret_states=True)

        l = states[1].sum()
        l.backward(retain_graph=True)

        self.assertTrue(x.grad[0].norm() == 0)
        self.assertTrue(x.grad[1].norm() > 0)
        self.assertTrue(x.grad[2].norm() == 0)
        self.assertTrue(x.grad[1][0].norm() > 0)
        self.assertTrue(x.grad[1][1].norm() == 0)
        self.assertTrue(x.grad[1][2].norm() == 0)
        self.assertTrue(x.grad[1][3].norm() == 0)

        x.grad = None
        l = states[2].sum()
        l.backward(retain_graph=True)
        self.assertTrue(x.grad[0].norm() == 0)
        self.assertTrue(x.grad[1].norm() == 0)
        self.assertTrue(x.grad[2].norm() > 0)
        self.assertTrue(x.grad[2][0].norm() > 0)
        self.assertTrue(x.grad[2][1].norm() > 0)
        self.assertTrue(x.grad[2][2].norm() == 0)
        self.assertTrue(x.grad[2][3].norm() == 0)

        print("done")

    def test_init_states(self):
        batsize = 5
        seqlen = 4
        lstm = self.encodertype(20, 26, 30)
        lstm.ret_all_states = True
        lstm.train(False)
        x = torch.randn(batsize, seqlen*2, 20)
        y_whole = lstm(x)

        y_first, states = lstm(x[:, :seqlen], ret_states=True)
        states = list(zip(*states))
        y_second = lstm(x[:, seqlen:], h_0s=states[0])

        y_part = torch.cat([y_first, y_second], 1)

        self.assertTrue(np.allclose(y_whole.detach().numpy(), y_part.detach().numpy()))

    def test_bidir(self):
        batsize = 5
        seqlen = 4
        lstm = self.encodertype(20, 26, 30, bidir=True)

        x = torch.nn.Parameter(torch.randn(batsize, seqlen, 20))

        y, y_T = lstm(x, ret_states=True)
        self.assertEqual((batsize, seqlen, 30*2), y.detach().numpy().shape)
        self.assertEqual((batsize, 2, 30), y_T.size())

        # test grad
        l = y[2, :, :].sum()
        l.backward()

        xgrad = x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[:, 0, :7])

        # test final state
        self.assertTrue(np.allclose(y_T[:, 0].detach().numpy(), y[:, -1, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[:, 1].detach().numpy(), y[:, 0, 30:].detach().numpy()))
        print(y_T.size())

    def test_bidir_masked(self):
        batsize = 5
        seqlen = 8
        lstm = self.encodertype(20, 26, 30, bidir=True)

        x = torch.randn(batsize, seqlen, 20)
        x.requires_grad = True
        mask = np.zeros((batsize, seqlen)).astype("int64")
        mask[0, :3] = 1
        mask[1, :] = 1
        mask[2, :5] = 1
        mask[3, :1] = 1
        mask[4, :4] = 1
        mask = torch.tensor(mask)

        y, y_T = lstm(x, mask=mask, ret_states=True)

        self.assertEqual((batsize, seqlen, 30 * 2), y.detach().numpy().shape)

        # test grad
        l = y[2, :, :].sum()
        l.backward(retain_graph=True)

        xgrad = x.grad.detach().numpy()

        # no gradient to examples that weren't used for loss
        self.assertTrue(np.allclose(xgrad[:2], np.zeros_like(xgrad[:2])))
        self.assertTrue(np.allclose(xgrad[3:], np.zeros_like(xgrad[3:])))

        # gradient on the example that was used for loss
        self.assertTrue(np.linalg.norm(xgrad) > 0)

        print(xgrad[2, :, :3])

        # test grad masked
        l = y.sum()
        l.backward()

        xgrad = x.grad.detach().numpy()

        mask = mask.detach().numpy()[:, :, np.newaxis]

        self.assertTrue(np.linalg.norm(xgrad * mask) > 0)
        self.assertTrue(np.allclose(xgrad * mask, xgrad))
        self.assertTrue(np.allclose(xgrad * (1 - mask), np.zeros_like(xgrad)))

        # test output mask
        self.assertTrue(np.linalg.norm(y.detach().numpy() * mask) > 0)
        self.assertTrue(np.allclose(y.detach().numpy() * (1 - mask), np.zeros_like(y.detach().numpy())))

        # test final states
        self.assertTrue(np.allclose(y_T[0, 0].detach().numpy(), y[0, 2, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[1, 0].detach().numpy(), y[1, -1, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[2, 0].detach().numpy(), y[2, 4, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[3, 0].detach().numpy(), y[3, 0, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[4, 0].detach().numpy(), y[4, 3, :30].detach().numpy()))
        self.assertTrue(np.allclose(y_T[:, 1].detach().numpy(), y[:, 0, 30:].detach().numpy()))
        print(y_T.size())

    def test_bidir_masked_equivalence_of_overridden(self):
        batsize = 5
        seqlen = 8
        self.encodertype.rnnlayertype = self._rnnlayertype_override
        lstm = self.encodertype(20, 26, 30, bidir=True)

        x = torch.randn(batsize, seqlen, 20)
        x.requires_grad = True
        mask = np.zeros((batsize, seqlen)).astype("int64")
        mask[0, :3] = 1
        mask[1, :] = 1
        mask[2, :5] = 1
        mask[3, :1] = 1
        mask[4, :4] = 1
        mask = torch.tensor(mask)

        y, yT = lstm(x, mask=mask, ret_states=True)

        # reference
        self.encodertype.rnnlayertype = self._rnnlayertype
        rf_lstm = self.encodertype(20, 26, 30, bidir=True)
        rf_lstm.layers[0].weight_ih_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.weight_ih_l0.detach().numpy()+0))
        rf_lstm.layers[0].weight_ih_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.weight_ih_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[0].weight_hh_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.weight_hh_l0.detach().numpy()+0))
        rf_lstm.layers[0].weight_hh_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.weight_hh_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[0].bias_ih_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.bias_ih_l0.detach().numpy()+0))
        rf_lstm.layers[0].bias_ih_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.bias_ih_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[0].bias_hh_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.bias_hh_l0.detach().numpy()+0))
        rf_lstm.layers[0].bias_hh_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[0].layer.bias_hh_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[1].weight_ih_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.weight_ih_l0.detach().numpy()+0))
        rf_lstm.layers[1].weight_ih_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.weight_ih_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[1].weight_hh_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.weight_hh_l0.detach().numpy()+0))
        rf_lstm.layers[1].weight_hh_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.weight_hh_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[1].bias_ih_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.bias_ih_l0.detach().numpy()+0))
        rf_lstm.layers[1].bias_ih_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.bias_ih_l0_reverse.detach().numpy()+0))
        rf_lstm.layers[1].bias_hh_l0 = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.bias_hh_l0.detach().numpy()+0))
        rf_lstm.layers[1].bias_hh_l0_reverse = torch.nn.Parameter(torch.tensor(lstm.layers[1].layer.bias_hh_l0_reverse.detach().numpy()+0))

        rf_x = torch.tensor(x.detach().numpy() + 0)
        rf_x.requires_grad = True
        assert(rf_x is not x)
        rf_y, rf_yT = rf_lstm(rf_x, mask=mask, ret_states=True)

        print((y - rf_y).norm())
        print(y.size())
        print(yT.size())
        print(rf_yT.size())
        print((yT - rf_yT).norm())
        self.assertTrue(np.allclose(yT.detach().numpy(), rf_yT.detach().numpy(), atol=10e-6))
        self.assertTrue(np.allclose(y.detach().numpy(), rf_y.detach().numpy(), atol=10e-6))
        print("outputs match")

        l = yT.sum()
        l.backward()

        rf_l = rf_yT.sum()
        rf_l.backward()

        print("input grad diff: {}".format((x.grad - rf_x.grad).norm()))
        self.assertTrue(np.allclose(rf_x.grad.detach().numpy(), x.grad.detach().numpy(), atol=10e-6))
        print(x.grad[:, :, 0])
        print("grad on inputs matches")

        for i in [0, 1]:
            for w in ["weight_ih_l0", "weight_hh_l0", "weight_ih_l0_reverse", "weight_hh_l0_reverse",
                      "bias_ih_l0", "bias_hh_l0", "bias_ih_l0_reverse", "bias_hh_l0_reverse"]:
                grad = getattr(lstm.layers[i].layer, w).grad.detach().numpy()
                rf_grad = getattr(rf_lstm.layers[i], w).grad.detach().numpy()
                print("grad for param {} in layer {} diff: {}".format(w, i, np.linalg.norm(grad - rf_grad)))
                self.assertTrue(np.allclose(grad, rf_grad, atol=10e-6))
                self.assertTrue(np.linalg.norm(grad) > 0)
                print("grad for param {} in layer {} matches and non-zero".format(w, i))

    def test_dropout_in(self):
        batsize = 5
        seqlen = 8
        lstm = self.encodertype(20, 30, bidir=False, dropout_in=0.3, dropout_rec=0.)

        x = torch.randn(batsize, seqlen, 20)
        x.requires_grad = True
        mask = np.zeros((batsize, seqlen)).astype("int64")
        mask[0, :3] = 1
        mask[1, :] = 1
        mask[2, :5] = 1
        mask[3, :1] = 1
        mask[4, :4] = 1
        mask = torch.tensor(mask)

        assert(lstm.training)

        y = lstm(x, mask=mask)

        y_t0_r0 = lstm(x, mask=mask)[:, 0, :30]
        y_t0_r1 = lstm(x, mask=mask)[:, 0, :30]
        y_t0_r2 = lstm(x, mask=mask)[:, 0, :30]

        self.assertTrue(not np.allclose(y_t0_r0.detach().numpy(), y_t0_r1.detach().numpy()))
        self.assertTrue(not np.allclose(y_t0_r1.detach().numpy(), y_t0_r2.detach().numpy()))
        self.assertTrue(not np.allclose(y_t0_r0.detach().numpy(), y_t0_r2.detach().numpy()))


class TestLSTMEncoder(TestRNNEncoder):
    encodertype = q.LSTMEncoder
    _rnnlayertype_override = OverriddenLSTMLayer
    _rnnlayertype = torch.nn.LSTM

    def test_init_states(self):
        batsize = 5
        seqlen = 4
        lstm = self.encodertype(20, 26, 30)
        lstm.ret_all_states = True
        lstm.train(False)
        x = torch.randn(batsize, seqlen*2, 20)
        y_whole = lstm(x)

        y_first, states = lstm(x[:, :seqlen], ret_states=True)
        states = list(zip(*states))
        y_second = lstm(x[:, seqlen:], y_0s=states[0], c_0s=states[1])

        y_part = torch.cat([y_first, y_second], 1)

        self.assertTrue(np.allclose(y_whole.detach().numpy(), y_part.detach().numpy()))


class TestGRUEncoder(TestRNNEncoder):
    encodertype = q.GRUEncoder
    _rnnlayertype_override = OverriddenGRULayer
    _rnnlayertype = torch.nn.GRU

# endregion


# region attention
class TestAttention(TestCase):
    def test_dot_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.DotAttention()

        alphas, summaries, scores = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_fwd_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.FwdAttention(ctxdim=7, qdim=7, attdim=7)

        alphas, summaries, scores = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_fwdmul_attention(self):
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        a = q.FwdAttention(ctxdim=7, qdim=7, attdim=7)

        alphas, summaries, scores = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))


class TestAttentionBases(TestCase):
    def test_it(self):
        a = q.DotAttention()

        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        alphas, summaries, scores = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

    def test_it_with_coverage(self):
        class DotAttentionWithCov(q.AttentionWithCoverage, q.DotAttention):
            pass

        a = DotAttentionWithCov()
        x = torch.randn(5, 10, 7)
        y = torch.randn(5, 7)

        alphas, summaries, scores = a(y, x)
        print(alphas.size())
        # test shapes
        self.assertEqual(alphas.size(), (5, 10))
        self.assertTrue(np.allclose(alphas.sum(1).detach().numpy(), np.ones_like(alphas.sum(1).detach().numpy())))
        self.assertEqual(summaries.size(), (5, 7))

# endregion


# region decoders
class TestDecoders(TestCase):
    def test_tf_decoder(self):
        decodercell = torch.nn.Linear(7, 12)
        x = torch.randn(5, 10, 7)
        decoder = q.TFDecoder(decodercell)
        y = decoder(x)
        self.assertEqual(y.size(), (5, 10, 12))

    def test_free_decoder(self):
        decodercell = torch.nn.Sequential(torch.nn.Embedding(12, 7),
                                          torch.nn.Linear(7, 12))
        x = torch.randint(0, 7, (5,), dtype=torch.int64)
        decoder = q.FreeDecoder(decodercell, maxtime=10)
        y = decoder(x)
        self.assertEqual(y.size(), (5, 10, 12))


class TestDecoderCell(TestCase):
    def test_it(self):
        x = np.random.randint(0, 100, (1000, 7))
        y_inp = x[:, :-1]
        y_out = x[:, 1:]
        wD = dict((chr(xi), xi) for xi in range(100))

        ctx = torch.randn(1000, 8, 30)

        decoder_emb = q.WordEmb(20, worddic=wD)
        decoder_lstm = q.LSTMCell(20, 30)
        decoder_att = q.DotAttention()
        decoder_out = q.WordLinout(60, worddic=wD)

        decoder_cell = q.DecoderCell(decoder_emb, decoder_lstm, decoder_att, None, decoder_out)
        decoder_tf = q.TFDecoder(decoder_cell)

        y = decoder_tf(torch.tensor(x), ctx=ctx)

        self.assertTrue(y.size(), (1000, 7, 100))


class TestLuongCell(TestCase):
    def test_stateful(self):
        pp = PrettyPrinter()
        D = "a b c d e f g".split()
        D = dict(zip(D, range(len(D))))
        emb = q.WordEmb(5, worddic=D)
        out = q.WordLinout(10, worddic=D)
        core = q.GRUCell(15, 5, dropout_rec=.4)
        core.h_tm1 = torch.rand(3, 5)
        att = q.BasicAttention()
        cell = q.LuongCell(emb=emb, core=core, att=att, out=out, feed_att=True)
        cell._outvec_tm1 = torch.rand(3, 10)
        x_m1 = torch.randint(0, max(D.values())+1, (3,))

        ctx = torch.rand(3, 4, 5)
        y_m1 = cell(x_m1, ctx=ctx)
        state0 = cell.get_state()

        x_0 = torch.randint(0, max(D.values())+1, (3,))
        y_0 = cell(x_0, ctx=ctx)
        state1 = cell.get_state()
        cell.set_state(state0)

        y_0bis = cell(x_0, ctx=ctx)
        state1bis = cell.get_state()

        print(y_0)
        print(y_0bis)
        print(pp.pprint(state1))
        print(pp.pprint(state1bis))

        self.assertTrue(np.allclose(y_0.detach().numpy(), y_0bis.detach().numpy()))
        for k, v in state1.items():
            if isinstance(v, torch.Tensor):
                self.assertTrue(np.allclose(state1[k].detach().numpy(), state1bis[k].detach().numpy()))
            else:
                self.assertTrue(state1[k] == state1bis[k])

# endregion


class TestBeamDecoder(TestCase):
    def test_it(self):
        class TestModule(torch.nn.Module):
            def __init__(self):
                super(TestModule, self).__init__()

            def forward(self, x):
                ret = torch.tensor([1., 2., 3., 4., 5.])
                ret = ret.unsqueeze(0).repeat(3, 1)
                ret = torch.log(torch.softmax(ret, -1))
                return ret

            def get_state(self):
                return {}

            def set_state(self, _):
                return

        m = TestModule()
        x = torch.tensor([0, 1, 2])
        y = m(x)
        print(y)

        beam = q.BeamDecoder(m, beamsize=2)
        y = beam(x, maxtime=4)
        print(y)
        self.assertTrue(torch.allclose(y, torch.ones_like(y)*4))

    def test_beam_vs_greedy_on_random(self):
        class TestModule(torch.nn.Module, q.Stateful):
            statevars = ["state"]
            def __init__(self):
                super(TestModule, self).__init__()
                self.transmat = torch.nn.Parameter(torch.randn(5, 5))
                self.state = None

            def batch_reset(self):
                self.state = None

            def forward(self, x):
                if self.state is None:
                    self.state = torch.rand(x.size(0), 7)
                ret = self.transmat[x]
                ret = torch.log(torch.softmax(ret, -1))
                return ret

            def get_state(self):
                return {}

            def set_state(self, _):
                return

        m = TestModule()
        x = torch.tensor([0, 1, 2])
        y = m(x)
        print(y)

        beam = q.BeamDecoder(m, maxtime=4, beamsize=1)
        greedy = q.FreeDecoder(m, maxtime=5)
        beam_y = beam(x)
        greedy_y = greedy(x)
        print(beam_y)
        print(torch.max(greedy_y, -1)[1])
        self.assertTrue(torch.all(beam_y == torch.max(greedy_y, -1)[1]))

