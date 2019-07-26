import torch
import torch.nn as nn
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


if __name__ == '__main__':
    data_set = F2EDataSet()
    encoder = Encoder(data_set.in_lang.token_n, embed_size=50, hidden_size=20, batch_size=32)
    loader = DataLoader(data_set, batch_size=32, shuffle=True)
    for batch_idx, (in_seq, out_seq) in enumerate(loader):
        print(in_seq.shape, out_seq.shape)
        encoder(in_seq, encoder.init_hidden())
        break
