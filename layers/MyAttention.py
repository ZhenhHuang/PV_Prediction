import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt
from math import sqrt


class WaveletConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=None, high=True, wavelet='db4', requires_grad=False):
        super(WaveletConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.high = high
        self.wavelet = pywt.Wavelet(wavelet)
        self.weight = nn.Parameter(self._weights, requires_grad=requires_grad)

    def forward(self, x):
        # [B, L, D]
        x = F.conv1d(self._pad(x), self.weight, stride=2).transpose(-1, 1)
        return x

    def _pad(self, x):
        # p = k/2 - 1
        _, L, _ = x.shape
        pad = self.wavelet.dec_len // 2 - 1
        if pad == 0:
            return x.transpose(-1, 1)
        x = F.pad(x.transpose(-1, 1), (pad, pad))
        return x

    @property
    def _weights(self):
        # [out, in, kernel]
        if self.high:
            weight = self.wavelet.dec_hi
        else:
            weight = self.wavelet.dec_lo
        weight = torch.tensor(weight)[None, None, :].repeat(self.out_channels, self.in_channels, 1)
        return weight


class WaveletTransposeConv1d(nn.Module):
    def __init__(self, in_channels, out_channels=None, high=True, wavelet='db4', requires_grad=False):
        super(WaveletTransposeConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        self.high = high
        self.wavelet = pywt.Wavelet(wavelet)
        self.weight = nn.Parameter(self._weights, requires_grad=requires_grad)

    def forward(self, x):
        # [B, L, D]
        L = x.shape[1]
        x = F.conv1d(self._pad(x), self.weight, stride=1).transpose(-1, 1)
        return x[:, :2 * L, :]

    def _pad(self, x):
        # p = k/2 - 1
        B, L, D = x.shape
        zeros = torch.zeros(B, 2*L, D).to(x.device)
        zeros[:, ::2, :] = x
        pad = (self.wavelet.rec_len - 1) // 2 + 1
        if pad == 0:
            return zeros.transpose(-1, 1)
        zeros = F.pad(zeros.transpose(-1, 1), (pad, pad))
        return zeros

    @property
    def _weights(self):
        # [in, out, kernel]
        if self.high:
            weight = self.wavelet.rec_hi
        else:
            weight = self.wavelet.rec_lo
        weight = torch.tensor(weight)[None, None, :].repeat(self.out_channels, self.in_channels, 1)
        return weight


class MyAttention(nn.Module):
    def __init__(self, L_k, M, scale=None, output_attention=False, dropout=0.1):
        super(MyAttention, self).__init__()
        self.scale = scale
        self.output_attention = output_attention
        self.dropout = nn.Dropout(dropout)
        self.squeeze = nn.Linear(L_k, M)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, D = queries.shape
        _, S, _, _ = keys.shape
        scale = self.scale or 1. / sqrt(D)
        keys_squeeze = self.squeeze(keys.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        values_squeeze = self.squeeze(values.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        attn = torch.einsum("blhd, bshd->bhls", queries, keys_squeeze)
        attn = self.dropout(torch.softmax(attn * scale, dim=-1))
        global_context = torch.einsum("bhls,bshd->blhd", attn, values_squeeze)
        if self.output_attention:
            return global_context, attn
        else:
            return global_context, None


# class MyAttentionLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, wavelet='db4', d_keys=None, d_values=None):
#         super(MyAttentionLayer, self).__init__()
#
#         d_keys = d_keys or (d_model // n_heads)
#         d_values = d_values or (d_model // n_heads)
#
#         self.n_heads = n_heads
#         self.attention = attention
#         self.w_q = nn.Linear(d_model, d_keys * n_heads)
#         self.w_k = nn.Linear(d_model, d_keys * n_heads)
#         self.w_v = nn.Linear(d_model, d_values * n_heads)
#         self.conv_h = WaveletConv1d(d_model, d_model, wavelet=wavelet)
#         self.conv_l = WaveletConv1d(d_model, d_model, wavelet=wavelet, high=False)
#         self.iconv_h = WaveletTransposeConv1d(d_model, d_model, wavelet=wavelet)
#         self.iconv_l = WaveletTransposeConv1d(d_model, d_model, wavelet=wavelet, high=False)
#         self.proj = nn.Linear(d_values * n_heads, d_model)
#
#
#     def forward(self, queries, keys, values, attn_mask=None):
#         B, L, D = queries.shape
#         _, S, _ = keys.shape
#         H = self.n_heads
#
#         high_q = self.w_q(self.conv_h(queries)).reshape(B, L//2, H, -1)     # B, L//2, d * h
#         low_q = self.w_q(self.conv_l(queries)).reshape(B, L//2, H, -1)
#
#         high_k = self.w_k(self.conv_h(keys)).reshape(B, S//2, H, -1)     # B, S//2, d * h
#         low_k = self.w_k(self.conv_l(keys)).reshape(B, S//2, H, -1)
#
#         high_v = self.w_v(self.conv_h(values)).reshape(B, S//2, H, -1)     # B, S//2, d * h
#         low_v = self.w_v(self.conv_l(values)).reshape(B, S//2, H, -1)
#
#         out_h, attn_h = self.attention(high_q, high_k, high_v, attn_mask)
#         out_l, attn_l = self.attention(low_q, low_k, low_v, attn_mask)
#         out_h = self.proj(out_h.reshape(B, L//2, -1))   # B, L//2, D
#         out_l = self.proj(out_l.reshape(B, L//2, -1))
#         out = self.iconv_l(out_l) + self.iconv_h(out_h)
#
#         return out, (attn_h, attn_l)


class MyAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, wavelet='db4', d_keys=None, d_values=None):
        super(MyAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.n_heads = n_heads
        self.attention = attention
        self.w_q = nn.Linear(d_model, d_keys * n_heads)
        self.w_k = nn.Linear(d_model, d_keys * n_heads)
        self.w_v = nn.Linear(d_model, d_values * n_heads)
        self.conv = WaveletConv1d(d_model, d_model, wavelet=wavelet, requires_grad=True)
        self.pw = nn.Conv1d(2 * d_values * n_heads, d_values * n_heads, kernel_size=1)
        self.dw = nn.Conv1d(2 * d_values * n_heads, d_values * n_heads, kernel_size=5, groups=d_values * n_heads, padding=2)
        self.out_proj = nn.Linear(d_values * n_heads, d_model)

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, D = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        high_x = self.w_q(self.conv(queries))     # B, L//2, D
        pad = torch.mean(high_x, dim=1, keepdim=True).repeat(1, L//4, 1)
        high_x_pad = torch.concat([pad, high_x, pad], dim=1)

        queries = self.w_q(queries).reshape(B, L, H, -1)
        keys = self.w_k(keys).reshape(B, S, H, -1)
        values = self.w_v(values).reshape(B, S, H, -1)

        global_out, global_attn = self.attention(queries, keys, values, attn_mask)
        global_out = global_out.reshape(B, L, -1)

        out = torch.concat([global_out, high_x_pad], dim=-1)
        x1 = self.pw(out.transpose(-1, 1)).transpose(-1, 1)
        x2 = self.dw(out.transpose(-1, 1)).transpose(-1, 1)
        out = self.out_proj(x1 + x2)

        return out, global_attn













