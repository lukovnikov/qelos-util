from unittest import TestCase
import numpy as np
import torch
import qelos as q


class TestTransformerBERT(TestCase):
    def test_bert_loaded(self):
        config_path="../data/bert/bert-base/bert_config.json"
        ckpt_path="../data/bert/bert-base/bert_model.ckpt"
        config = q.TransformerBERT.load_config(config_path)
        m = q.TransformerBERT.init_from_config(config)
        m.load_weights_from_tf_checkpoint(ckpt_path)
        m.eval()
        print(m)
        input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
        input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
        token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

        all_h, pooled_output = m(input_ids, token_type_ids, input_mask)
        all_h = torch.stack(all_h, 0)
        print(pooled_output.shape)

        # load original output from Hugging Face BERT for same input
        ref_out = np.load("_bert_ref_pool_out.npy")
        print(ref_out.shape)

        print((pooled_output.detach().numpy() - ref_out)[:, :10])

        all_ref_out = np.load("_bert_ref_all_out.npy")
        print(all_ref_out.shape)
        # print((all_h.detach().numpy() - all_ref_out)[0, :, :10])
        print((all_h.detach().numpy() - all_ref_out)[-1, :, :, :10])
        self.assertTrue(np.allclose(all_ref_out, all_h.detach().numpy(), atol=1e-4))
        self.assertTrue(np.allclose(ref_out, pooled_output.detach().numpy(), atol=1e-4))

