import qelos as q
import torch
import numpy as np
import six
import copy
import json
import re


""" Heavily borrowed from Hugging Face BERT """


__all__ = ["TransformerBERT", "BertConfig"]


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    --> From hugging face BERT
    """
    def __init__(self,
                vocab_size,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTEmbeddings(torch.nn.Module):
    def __init__(self, dim, numwords, maxlen, numtypes=2, dropout=0.):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
           --> from Hugging Face BERT
        """
        self.word_embeddings = torch.nn.Embedding(numwords, dim)
        self.position_embeddings = torch.nn.Embedding(maxlen, dim)
        self.token_type_embeddings = torch.nn.Embedding(numtypes, dim)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = torch.nn.LayerNorm(dim, eps=1e-12)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BERTPooler(torch.nn.Module):
    """ from Hugging Face BERT """
    def __init__(self, dim=None):
        super(BERTPooler, self).__init__()
        self.dense = torch.nn.Linear(dim, dim)
        self.activation = torch.nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TransformerEncoder(torch.nn.Module):
    def __init__(self, dim=512, kdim=None, vdim=None, innerdim=None, maxlen=512, numlayers=6, numheads=8, activation=torch.nn.ReLU,
                 embedding_dropout=0., attention_dropout=0., residual_dropout=0., scale=True,
                 relpos=False, posemb=None, **kw):
        """
        :param dim:     see MultiHeadAttention
        :param kdim:    see MultiHeadAttention
        :param vdim:    see MultiHeadAttention
        :param maxlen:  see MultiHeadAttention
        :param numlayers:   number of TransformerEncoderBlock layers used
        :param numheads:    see MultiHeadAttention
        :param activation:  which activation function to use in positionwise feedforward layers
        :param embedding_dropout:   dropout rate on embedding. Time-shared dropout. Not applied to position embeddings
        :param attention_dropout:   see MultiHeadAttention
        :param residual_dropout:    dropout rate on outputs of attention and feedforward layers
        :param scale:   see MultiHeadAttention
        :param relpos:  see MultiHeadAttention
        :param posemb:  if specified, must be a nn.Embedding-like, embeds position indexes in the range 0 to maxlen
        :param kw:
        """
        super(TransformerEncoder, self).__init__(**kw)
        self.maxlen = maxlen
        self.layers = torch.nn.ModuleList([
            q.TransformerEncoderBlock(dim, kdim=kdim, vdim=vdim, innerdim=innerdim, numheads=numheads, activation=activation,
                                    attention_dropout=attention_dropout, residual_dropout=residual_dropout,
                                    scale=scale, maxlen=maxlen, relpos=relpos)
            for _ in range(numlayers)
        ])

    def forward(self, x, mask=None):
        """
        :param x:       (batsize, seqlen, dim)
        :param mask:    optional mask (batsize, seqlen)
        :return:        (batsize, seqlen, outdim)
        """
        h = x
        all_h = []
        for layer in self.layers:
            h = layer(h, mask=mask)
            all_h.append(h)
        return all_h


class TransformerBERT(torch.nn.Module):
    def __init__(self, dim=768, numwords=-1, numlayers=12, numheads=12, innerdim=3072, hidden_act=q.GeLU,
                 dropout=0.1, attn_dropout=0.1, maxlen=512, numtypes=16, init_range=0.02):
        super(TransformerBERT, self).__init__()
        self.emb = BERTEmbeddings(dim, numwords, maxlen, numtypes=numtypes, dropout=dropout)
        self.encoder = TransformerEncoder(dim=dim, innerdim=innerdim, maxlen=maxlen, numlayers=numlayers,
                                            numheads=numheads, activation=hidden_act, embedding_dropout=dropout,
                                            residual_dropout=dropout, attention_dropout=attn_dropout,
                                            relpos=False, posemb=None, scale=True)
        q.RecDropout.convert_to_standard_in(self.encoder)
        self.pooler = BERTPooler(dim)

    def forward(self, input_ids, token_type_ids=None, mask=None):
        """ --> adapted from Hugging Face BERT """
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, from_seq_length]
        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        # extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.float()
        # extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.emb(input_ids, token_type_ids)
        all_h = self.encoder(embedding_output, mask=mask)
        sequence_output = all_h[-1]
        pooled_output = self.pooler(sequence_output)
        return all_h, pooled_output

    def load_weights_from_tf_checkpoint(self, ckpt_path):
        print("Loading tensorflow BERT weights from {}".format(ckpt_path))
        import tensorflow as tf

        # region from Hugging Face BERT
        init_vars = tf.train.list_variables(ckpt_path)
        names = []
        arrays = []
        for name, shape in init_vars:
            print("Loading {} with shape {}".format(name, shape))
            array = tf.train.load_variable(ckpt_path, name)
            print("Numpy array shape {}".format(array.shape))
            names.append(name)
            arrays.append(array)
        # endregion

        # load values from tf ckpt variable paths to our paths
        def mapname(a):
            """
            LayerNorm
                beta -> bias
                gamma -> weight
            (.+)_embeddings -> {1}_embeddings.weight
            /embeddings -> emb
            /encoder
                layer_(\d+) -> layers,{1}
                    attention/output/LayerNorm -> ln_slf
                    attention -> slf_attn
                        self/query -> q_proj
                        self/key -> k_proj
                        self/value -> v_proj
                        output/dense -> vw_proj
                    intermediate/dense -> mlp.projA
                    output/dense -> mlp.projB
                    output/LayerNorm -> ln_ff
                    """
            if re.match(".+LayerNorm.+", a):
                a = re.sub("LayerNorm/gamma$", "LayerNorm/weight", a)
                a = re.sub("LayerNorm/beta$", "LayerNorm/bias", a)
            if re.match(".+_embeddings$", a):
                a = re.sub("(.+_embeddings)$", "\g<1>/weight", a)
            # a = re.sub("kernel$", "weight", a)
            a = re.sub("^embeddings", "emb", a)
            if re.match("^encoder", a):
                if re.match("^encoder/layer_\d+", a):
                    a = re.sub("^(encoder/layer)_(\d+)", "encoder/layers/\g<2>", a)
                    a = re.sub("attention/output/LayerNorm", "ln_slf", a)
                    if re.match(".+attention.+", a):
                        a = re.sub("attention/self/query", "attention/q_proj", a)
                        a = re.sub("attention/self/key", "attention/k_proj", a)
                        a = re.sub("attention/self/value", "attention/v_proj", a)
                        a = re.sub("attention/output/dense", "attention/vw_proj", a)
                        a = re.sub("attention", "slf_attn", a)
                    a = re.sub("intermediate/dense", "mlp/projA", a)
                    a = re.sub("output/dense", "mlp/projB", a)
                    a = re.sub("output/LayerNorm", "ln_ff", a)
            return a

        for name, array in zip(names, arrays):
            name = name[5:]  # skip "bert/"
            print("Loading {}".format(name))
            name = mapname(name)
            name = name.split('/')
            if name[0] in ['redictions', 'eq_relationship']:
                print("Skipping")
                continue
            pointer = self
            for m_name in name:
                getname = m_name
                if m_name == "kernel":
                    getname = "weight"
                pointer = getattr(pointer, getname)
            if m_name == 'kernel':
                array = np.transpose(array)
            try:
                assert pointer.shape == array.shape
            except AssertionError as e:
                e.args += (pointer.shape, array.shape)
                raise
            pointer.data = torch.from_numpy(array)

    @classmethod
    def init_from_config(cls, config:BertConfig):
        hidden_act = config.hidden_act
        hidden_act = q.GeLU if hidden_act == "gelu" else (torch.nn.ReLU if hidden_act == "relu" else (q.Swish if hidden_act == "swish" else None))
        ret = cls(dim=config.hidden_size,
                  numwords=config.vocab_size,
                  numlayers=config.num_hidden_layers,
                  numheads=config.num_attention_heads,
                  innerdim=config.intermediate_size,
                  hidden_act=hidden_act,
                  dropout=config.hidden_dropout_prob,
                  attn_dropout=config.attention_probs_dropout_prob,
                  maxlen=config.max_position_embeddings,
                  numtypes=config.type_vocab_size,
                  init_range=config.initializer_range)
        return ret

    @staticmethod
    def load_config(path):
        return BertConfig.from_json_file(path)


def run(config_path="../data/bert/bert-base/bert_config.json",
        ckpt_path="../data/bert/bert-base/bert_model.ckpt"):
    config = BertConfig.from_json_file(config_path)
    m = TransformerBERT.init_from_config(config)
    m.load_weights_from_tf_checkpoint(ckpt_path)
    print(m)


if __name__ == '__main__':
    q.argprun(run)