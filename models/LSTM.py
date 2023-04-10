import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_layers = configs.num_layers
        self.num_hidden = configs.d_model
        self.lstm = nn.LSTM(input_size=configs.enc_in, hidden_size=configs.d_model, num_layers=configs.num_layers,
                            batch_first=True, dropout=configs.dropout)
        self.project = nn.Linear(configs.d_model, configs.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, D = x_enc.shape
        h_0 = torch.randn(self.num_layers, B, self.num_hidden).to(device='cuda')
        c_0 = torch.randn(self.num_layers, B, self.num_hidden).to(device='cuda')
        output, hidden = self.lstm(x_enc, (h_0, c_0))
        output = self.project(output)
        return output