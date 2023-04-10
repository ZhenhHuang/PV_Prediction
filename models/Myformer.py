import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Myformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer
from layers.MyAttention import MyAttentionLayer, MyAttention, WaveletConv1d, WaveletTransposeConv1d
from layers.Embed import DataEmbedding_wo_pos
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)

        grad = True
        self.conv_h = WaveletConv1d(configs.d_model, configs.d_model // 2, wavelet=configs.wavelet, requires_grad=grad)
        self.conv_l = WaveletConv1d(configs.d_model, configs.d_model // 2, wavelet=configs.wavelet, high=False,
                                    requires_grad=grad)
        self.iconv_h = WaveletTransposeConv1d(configs.d_model // 2, configs.d_model, wavelet=configs.wavelet,
                                              requires_grad=grad)
        self.iconv_l = WaveletTransposeConv1d(configs.d_model // 2, configs.d_model, wavelet=configs.wavelet,
                                              high=False,
                                              requires_grad=grad)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    MyAttentionLayer(
                        MyAttention(L_k=configs.seq_len // 2, M=configs.M // 2, dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        wavelet=configs.wavelet
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    MyAttentionLayer(
                        MyAttention(L_k=(configs.label_len + configs.pred_len) // 2, M=configs.M // 2,
                                    dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        wavelet=configs.wavelet
                    ),
                    MyAttentionLayer(
                        MyAttention(L_k=configs.seq_len // 2, M=configs.M, dropout=configs.dropout,
                                    output_attention=configs.output_attention),
                        d_model=configs.d_model,
                        n_heads=configs.n_heads,
                        wavelet=configs.wavelet
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out_h, enc_out_l = self.conv_h(enc_out), self.conv_l(enc_out)
        enc_out = torch.concat([enc_out_h, enc_out_l], dim=-1)

        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out_h, dec_out_l = self.conv_h(dec_out), self.conv_l(dec_out)
        dec_out = torch.concat([dec_out_h, dec_out_l], dim=-1)

        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        D = dec_out.shape[-1] // 2
        dec_out_h, dec_out_l = dec_out[:, :, :D], dec_out[:, :, D:]
        dec_out = self.iconv_l(dec_out_l) + self.iconv_h(dec_out_h)
        dec_out = self.projection(dec_out)
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


