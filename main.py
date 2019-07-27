import torch
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from prepare_data import F2EDataSet
from model import Encoder, Decoder

torch.manual_seed(1234)
device = torch.device('cuda')
embed_size = 64
hidden_size = 64
num_layers = 2
batch_size = 64
attention_size = 10
drop_prob = 0.2
num_epochs = 100
lr = 1e-3
random_sample_k = 5
max_seq_len = 15


def translate(dataset, random_sample_sentences, sample_in_indices, encoder, decoder, device):
    for i in range(random_sample_k):
        # sample_in_indices: [1, max_len]
        # enc_outputs: [max_len, 1, hidden_size]
        # enc_hidden: [num_layers, 1, hidden_size]
        enc_outputs, enc_hidden = encoder(sample_in_indices[i], encoder.init_hidden(1, device=device))
        # dec_input: [1, 1]
        dec_input = decoder.init_input(1, device=device)
        # dec_hidden: [num_layers, 1, hidden_size]
        dec_hidden = decoder.init_hidden(enc_hidden)
        answer = []
        for _ in range(max_seq_len):
            # dec_output: [1, out_vocab_size]  dec_hidden: [num_layers, 1, hidden_size]
            dec_output, dec_hidden = decoder(dec_input, dec_hidden, enc_outputs)
            # pred_token: [1, 1]
            pred_token = torch.argmax(dec_output, dim=1).view(1, 1)
            answer.append(pred_token.item())
            if pred_token.item() == 2:  # 2: EOS_index
                break
            else:
                dec_input = pred_token
        print('>', random_sample_sentences[i][0])
        print('=', random_sample_sentences[i][1])
        print('<', dataset.convert_index_to_token(dataset.out_lang, answer))


def main():
    data_set = F2EDataSet(max_length=max_seq_len)
    loader = DataLoader(data_set, batch_size=batch_size, shuffle=True)
    encoder = Encoder(data_set.in_lang.token_n, embed_size=embed_size, hidden_size=hidden_size, num_layers=num_layers,
                      drop_prob=drop_prob).to(device)
    decoder = Decoder(vocab_size=data_set.out_lang.token_n, embed_size=embed_size, hidden_size=hidden_size,
                      num_layers=num_layers, attention_size=attention_size, drop_prob=drop_prob).to(device)
    enc_optimizer = optim.Adam(encoder.parameters(), lr=lr)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=lr)
    criteon = nn.CrossEntropyLoss(reduction='none').to(device)
    random_sample_sentences = data_set.random_sample(k=random_sample_k)
    sample_in_indices = []
    for in_sentence, out_sentence in random_sample_sentences:
        sample_in_indices.append(data_set.convert_token_to_index(data_set.in_lang, in_sentence))
    # sample_in_indices: shape[random_sample_k, max_len], dtype: int64
    sample_in_indices = torch.LongTensor(sample_in_indices).to(device)
    # sample_in_indices: [random_sample_k, 1, max_len]
    sample_in_indices = torch.unsqueeze(sample_in_indices, dim=1)
    for epoch in range(num_epochs):
        total_loss = 0
        encoder.train()
        decoder.train()
        for batch_idx, (in_seq, out_seq) in enumerate(loader):
            this_batch_size = in_seq.shape[0]
            # in_seq, out_seq shape: [batch_size, max_len], dtype = int64
            in_seq, out_seq = in_seq.to(device), out_seq.to(device)
            # enc_outputs of shape (seq_len, batch, num_directions * hidden_size)
            # enc_hidden of shape (num_layers * num_directions, batch, hidden_size)
            enc_outputs, enc_hidden = encoder(in_seq, encoder.init_hidden(this_batch_size, device=device))
            # 解码器在最初时间步的输入是BOS
            # dec_input: [batch_size, 1]
            dec_input = decoder.init_input(this_batch_size, device=device)
            # initialize hidden state of decoder
            # dec_hidden: [num_layers, batch_size, hidden_size]
            dec_hidden = decoder.init_hidden(enc_hidden)
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
            loss /= num_not_pad_tokens
            total_loss += loss
            enc_optimizer.zero_grad()
            dec_optimizer.zero_grad()
            loss.backward()
            enc_optimizer.step()
            dec_optimizer.step()
        decoder.eval()
        encoder.eval()
        print(f"epoch {epoch+1}, loss = {total_loss/data_set.__len__()}")
        if epoch % 10 == 0:
            translate(data_set, random_sample_sentences, sample_in_indices, encoder, decoder, device)
    translate(data_set, random_sample_sentences, sample_in_indices, encoder, decoder, device)


if __name__ == "__main__":
    main()
