import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from prepare_data import F2EDataSet


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, batch_size, num_layers=1, drop_prob=0):
        super(Encoder, self).__init__()
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(input_size=embed_size, hidden_size=self.hidden_size, num_layers=self.num_layers,
                          dropout=drop_prob)

    def forward(self, inputs, hidden):
        # embedding shape: [batch, seq_len, input_size]
        embedding = self.embedding(inputs)
        # embedding shape => [seq_len, batch, input_size]
        embedding = torch.transpose(embedding, 0, 1)
        print(f"embedding shape: {embedding.shape}, hidden shape: {hidden.shape}")
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        output, h_n = self.gru(embedding, hidden)
        # output of shape (seq_len, batch, num_directions * hidden_size)
        # h_n of shape (num_layers * num_directions, batch, hidden_size)
        print(f"output shape: {output.shape}, h_n shape: {h_n.shape}")
        return output, h_n

    def init_hidden(self):
        # h_0 of shape (num_layers * num_directions, batch, hidden_size):
        return torch.zeros(self.num_layers, self.batch_size, self.hidden_size)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, enc_hidden_size, dec_hidden_size, batch_size, num_layers=1,
                 attention_size=10, drop_prob=0):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.enc_hidden_size = enc_hidden_size
        self.dec_hidden_size = dec_hidden_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.attention_size = attention_size
        self.drop_prob = drop_prob

        self.embedding = nn.Embedding(self.vocab_size, self.embed_size)
        self.attention = nn.Sequential(
            nn.Linear(in_features=self.enc_hidden_size + self.dec_hidden_size, out_features=self.attention_size,
                      bias=False),
            nn.Tanh(),
            nn.Linear(in_features=self.attention_size, out_features=1, bias=False)
        )
        self.gru = nn.GRU(input_size=self.embed_size, hidden_size=self.dec_hidden_size, num_layers=self.num_layers,
                          dropout=self.drop_prob)

        # self.out = nn.Linear(in_features=0, out_features=self.vocab_size)

    def forward(self, dec_input, dec_hidden, enc_outputs):
        """
        :param dec_input: [seq_len, batch, input_size]
        :param dec_hidden: [num_layers * num_directions, batch, hidden_size] => [num_layers, batch, dec_hidden_size]
        :param enc_outputs: encoder outputs,  [seq_len, batch, num_directions * hidden_size]
        :return:
        """
        c = self.attention_forward(dec_hidden, enc_outputs)

    def attention_forward(self, dec_hidden, enc_outputs):
        dec_hiddens, enc_outputs = torch.broadcast_tensors(dec_hidden, enc_outputs)
        # enc_dec_states shape: [seq_len, batch_size, enc_hidden_size + dec_hidden_size]
        enc_dec_states = torch.cat((dec_hiddens, enc_outputs), dim=2)
        # e shape: [seq_len, batch_size, 1]
        e = self.attention(enc_dec_states)
        # 对整个输入句子做attention
        alpha = F.softmax(e, dim=0)
        # c shape: [batch_size, dec_hidden_size]
        c = torch.sum(alpha * enc_outputs, dim=0)
        return c

    def init_hidden(self, hidden):
        return hidden

    def init_input(self):
        # 解码器在最初时间步的输入是特殊字符BOS
        return torch.ones(self.batch_size, 1)


if __name__ == '__main__':
    data_set = F2EDataSet()
    encoder = Encoder(data_set.in_lang.token_n, embed_size=50, hidden_size=20, batch_size=32)
    decoder = Decoder(vocab_size=data_set.out_lang.token_n, embed_size=30, enc_hidden_size=20, dec_hidden_size=20,
                      batch_size=32)
    loader = DataLoader(data_set, batch_size=32, shuffle=True)
    for batch_idx, (in_seq, out_seq) in enumerate(loader):
        print(in_seq.shape, out_seq.shape)
        enc_outputs, enc_hidden = encoder(in_seq, encoder.init_hidden())
        # enc_outputs of shape (seq_len, batch, num_directions * hidden_size)
        # enc_hidden of shape (num_layers * num_directions, batch, hidden_size)
        print(enc_outputs.shape, enc_hidden.shape)

        decoder(decoder.init_input(), decoder.init_hidden(enc_hidden), enc_outputs)
        break
