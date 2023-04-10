import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MyAttention import WaveletConv1d, WaveletTransposeConv1d, MyAttentionLayer, MyAttention
from layers.Embed import DataEmbedding
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        grad = False
        self.w = nn.Linear(configs.enc_in, configs.d_model)
        self.w2 = nn.Linear(configs.dec_in, configs.d_model)
        self.conv_h = WaveletConv1d(configs.d_model, configs.d_model, wavelet=configs.wavelet, requires_grad=grad)
        self.conv_l = WaveletConv1d(configs.d_model, configs.d_model, wavelet=configs.wavelet, high=False,
                                    requires_grad=grad)
        self.iconv_h = WaveletTransposeConv1d(configs.d_model, configs.d_model, wavelet=configs.wavelet,
                                              requires_grad=grad)
        self.iconv_l = WaveletTransposeConv1d(configs.d_model, configs.d_model, wavelet=configs.wavelet, high=False,
                                              requires_grad=grad)

        self.attentionlayer_h = MyAttentionLayer(
            MyAttention(configs.seq_len // 2, configs.M, dropout=configs.dropout),
            configs.d_model, n_heads=configs.n_heads, wavelet=configs.wavelet
        )
        self.attentionlayer_l = MyAttentionLayer(
            MyAttention(configs.seq_len // 2, configs.M, dropout=configs.dropout),
            configs.d_model, n_heads=configs.n_heads, wavelet=configs.wavelet
        )
        self.cross_attention = MyAttentionLayer(
            MyAttention(configs.seq_len, configs.M, dropout=configs.dropout),
            configs.d_model, n_heads=configs.n_heads, wavelet=configs.wavelet
        )
        self.conv1 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)
        self.dropout = nn.Dropout(configs.dropout)
        self.activation = F.relu
        self.wh = nn.Linear(configs.seq_len // 2, self.pred_len // 2)
        self.wl = nn.Linear(configs.seq_len // 2, self.pred_len // 2)
        self.proj = nn.Linear(configs.d_model, configs.enc_in)
        self.norm1 = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        self.conv3 = nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_ff, kernel_size=1)
        self.conv4 = nn.Conv1d(in_channels=configs.d_ff, out_channels=configs.d_model, kernel_size=1)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        B, L, D = x_enc.shape
        x_enc = self.w(x_enc)
        high_x = self.conv_h(x_enc)
        low_x = self.conv_l(x_enc)

        high_x = self.attentionlayer_h(high_x, high_x, high_x)[0] + high_x
        low_x = self.attentionlayer_l(low_x, low_x, low_x)[0] + low_x
        x = self.iconv_l(low_x) + self.iconv_h(high_x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.norm1(self.dropout(self.conv2(y).transpose(-1, 1)) + x)

        x_dec = self.w2(x_dec)
        x_dec = self.cross_attention(x_dec, y, y)[0] + x_dec
        y = x_dec
        y = self.dropout(self.activation(self.conv3(y.transpose(-1, 1))))
        y = self.norm2(self.dropout(self.conv4(y).transpose(-1, 1)) + x_dec)
        return self.proj(y)[:, -self.pred_len:, :]



