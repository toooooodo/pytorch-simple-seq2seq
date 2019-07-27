import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from prepare_data import F2EDataSet
from model import Encoder, Decoder
import numpy as np

torch.manual_seed(1234)
device = torch.device('cuda')
embed_size = 64
hidden_size = 64
num_layers = 2
batch_size = 32
attention_size = 10
drop_prob = 0.2
num_epochs = 50
lr = 1e-2


def main():
    data_set = F2EDataSet()
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    encoder = Encoder(data_set.in_lang.token_n, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      drop_prob=drop_prob).to(device)
    decoder = Decoder(vocab_size=data_set.out_lang.token_n, embed_size=embed_size, hidden_size=hidden_size,
                      num_layers=num_layers, attention_size=attention_size, drop_prob=drop_prob).to(device)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss(reduction='none').to(device)
    for epoch in range(num_epochs):
        for batch_idx, (in_seq, out_seq) in enumerate(loader):
            this_batch_size = in_seq.shape[0]
            # in_seq, out_seq shape: [batch_size, max_len]
            in_seq, out_seq = in_seq.to(device), out_seq.to(device)
            # enc_outputs of shape (seq_len, batch, num_directions * hidden_size)
            # enc_hidden of shape (num_layers * num_directions, batch, hidden_size)
            enc_outputs, enc_hidden = encoder(in_seq, encoder.init_hidden(this_batch_size, device=device))
            print(f"enc_outputs shape: {enc_outputs.shape}, enc_hidden shape: {enc_hidden.shape}")
            # 解码器在最初时间步的输入是BOS
            dec_input = decoder.init_input(this_batch_size, device=device)
            # initialize hidden state of decoder
            dec_hidden = decoder.init_hidden(enc_hidden)
            print(f"dec_hidden shape: {dec_hidden.shape}")
            # mask [batch_size]
            mask = torch.ones(this_batch_size, device=device)
            eos = torch.LongTensor([2] * this_batch_size).to(device)
            pad = torch.zeros(this_batch_size).to(device)
            num_not_pad_tokens = 0
            loss = 0
            for y in torch.transpose(out_seq, 0, 1):
                dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
                loss += torch.sum((criteon(dec_output, y) * mask), dim=0)
                # y: [batch_size] => [batch_size, 1]
                dec_input = torch.unsqueeze(y, dim=1)
                num_not_pad_tokens += torch.sum(mask, dim=0)
                # 当遇到EOS时，序列后面的词将均为PAD，相应位置的掩码设成0
                mask = torch.where(y != eos, mask, pad)
                # break
            # print(loss/num_not_pad_tokens)
            # print(num_not_pad_tokens)
        break


if __name__ == "__main__":
    main()
