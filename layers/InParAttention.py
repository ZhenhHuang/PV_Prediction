import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.utils import split_idx
from math import sqrt
import numpy as np


class IPConv(nn.Module):
    def __init__(self, in_channels, out_channels: list, kernel_size=[4, 6, 2], stride: list = [4, 2, 2], dilation=1):
        """
        kernel_size=[4, 6, 2], stride: list = [4, 2, 2]
        kernel_size=[4, 6, 2, 2], stride: list = [4, 2, 2, 2], dilation = seq_len // 2 + 1
        no padding for extra branch
        """
        super(IPConv, self).__init__()
        assert dilation >= 1, 'Required that dilation >= 1'
        assert stride[0] == stride[1] * stride[
            2], 'Required that sequence length should be same after two type Convolution'
        self.extra = False if dilation == 1 else True
        self.pad = [int((kernel_size[i] - stride[i]) // 2) for i in range(len(stride))]
        self.conv_1 = nn.Conv1d(in_channels, out_channels[0], kernel_size[0], stride[0])
        self.conv_21 = nn.Conv1d(in_channels, out_channels[1], kernel_size[1], stride[1])
        self.conv_22 = nn.Conv1d(out_channels[1], out_channels[1], kernel_size[2], stride[2])
        if self.extra:
            self.conv_3 = nn.Conv1d(in_channels, out_channels[1], kernel_size[3], stride[3], dilation=dilation)
            self.pad[-1] = 0

    def forward(self, x):
        x1 = self.__get_padding(x, self.pad[0])
        x1 = self.conv_1(x1.transpose(-1, 1)).transpose(-1, 1)
        # print(f"x1:{x1.shape}")
        x2 = self.__get_padding(x, self.pad[1])
        x2 = self.conv_21(x2.transpose(-1, 1)).transpose(-1, 1)
        x2 = self.__get_padding(x2, self.pad[2])
        x2 = self.conv_22(x2.transpose(-1, 1)).transpose(-1, 1)
        # print(f"x2:{x2.shape}")
        x = torch.concat([x1, x2], dim=-1)

        if self.extra:
            x3 = self.__get_padding(x, self.pad[3])
            x3 = self.conv_3(x3.transpose(-1, 1)).transpose(-1, 1)
            x = torch.concat([x, x3], dim=-1)

        return x

    def __get_padding(self, x, pad):
        # x: B, L, D
        if pad == 0:
            return x
        pad_front = x[:, 0:1, :].repeat(1, pad, 1)
        pad_end = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.concat([pad_front, x, pad_end], dim=1)
        return x


class InParAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, seq_len=None, d_keys=None, d_values=None,
                 conv_size=[4, 6, 2], conv_stride=[4, 2, 2], dilation=1):
        super(InParAttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        in_channels = n_heads * d_keys
        self.n_heads = n_heads
        self.attention = attention

        self.w_q = nn.Linear(d_model, n_heads * d_keys)
        self.w_k = nn.Linear(d_model, n_heads * d_keys)
        self.w_v = nn.Linear(d_model, n_heads * d_values)

        self.conv = IPConv(in_channels=in_channels, out_channels=[in_channels // 2, in_channels // 2],
                           kernel_size=conv_size, stride=conv_stride, dilation=dilation)

        """extra branch"""
        # self.conv = IPConv(in_channels=in_channels, out_channels=[in_channels - 2 * in_channels//3, in_channels//3],
        #                    kernel_size=[4, 6, 2, 2], stride=[4, 2, 2, 2], dilation=seq_len//2 + 1)

        self.proj = nn.Linear(n_heads * d_values, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.w_q(queries).view(B, L, H, -1)
        keys = self.conv(self.w_k(keys)).view(B, -1, H, D // H)
        values = self.conv(self.w_v(values)).view(B, -1, H, D // H)

        out, scores = self.attention(queries, keys, values, attn_mask)

        out = self.proj(out)

        return out, scores


class InParAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, d_model=512, heads=8, seq_len=None):
        super(InParAttention, self).__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)
        self.factor = factor
        self.w_v = nn.Linear(d_model // heads, 1, bias=False)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        u = int(self.factor * np.log(L))
        sample_index, rest_index = split_idx(L, H, u, overlap_mode='randomly')
        sample_index.sort(0)
        rest_index.sort(0)

        queries_main = queries[:,
                       sample_index,
                       torch.arange(H)[None, :],
                       :]  # B, L-u, H, D
        queries_rest = queries[:,
                       rest_index,
                       torch.arange(H)[None, :],
                       :]  # B, u, H, D

        out_main, score1 = self._freq_aware(queries_main, keys, values)
        out_rest, score2 = self._time_aware(queries_rest, keys, values)

        out = (values.sum(1) / L).unsqueeze(1).repeat(1, L, 1, 1).clone()
        out[:, sample_index, torch.arange(H)[None, :], :] = out_main
        out[:, rest_index, torch.arange(H)[None, :], :] = out_rest

        return out.contiguous().view(B, L, -1), (score1, score2)

    def _freq_aware(self, queries, keys, values):
        B, L, H, E = queries.shape

        queries_fft = torch.fft.rfft(queries, dim=1)
        keys_fft = torch.fft.rfft(keys, dim=1)
        values_fft = torch.fft.rfft(values, dim=1)

        qk = torch.einsum('blhd, bshd->bhls', queries_fft, keys_fft)  # B, H, u, S
        score = self.dropout(F.softmax(torch.abs(qk) / sqrt(E), dim=-1))
        score = torch.complex(score, torch.zeros_like(score))
        out = torch.einsum('bhls,bshd->blhd', score, values_fft).contiguous()
        out = torch.fft.irfft(out, n=L, dim=1).contiguous()
        return out, score

    def _time_aware(self, queries, keys, values):
        qk_add = queries.unsqueeze(2) + keys.unsqueeze(1)  # B, u, S, H, D
        score = self.w_v(torch.tanh(qk_add)).squeeze(-1).permute(0, 3, 1, 2).contiguous()  # B, H, u, S
        score = self.dropout(F.softmax(score, dim=-1))
        out = torch.einsum('bhls,bshd->blhd', score, values)
        return out, score