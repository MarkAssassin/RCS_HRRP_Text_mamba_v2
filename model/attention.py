# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:38:27 2022

@author: DELL
"""
import math
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from torch.nn.parameter import Parameter
from torch.nn.init import constant_, xavier_normal_
from torch.nn.init import xavier_uniform_
import warnings

Tensor = torch.Tensor

def positional_encoding(X, num_features, dropout_p=0.0, max_len=512) -> Tensor: #这里的max_len不用改为101，因为512>101，如果维度大于512了再改
    r'''
        给输入加入位置编码 X对应d_in, num_featrues对应d_hid, 即num_features=d_hid=d_model=100
    参数：
        - num_features: 输入进来的维度
        - dropout_p: dropout的概率，当其为非零元素时执行dropout
        - max_len: 句子的最大长度，默认512

    形状：
        - 输入： [batch_size, seq_length, num_features]
        - 输出： [batch_size, seq_length, num_features]
    例子：
        >>> X = torch.randn((2,4,10))
        >>> X = positional_encoding(X, 10)
        >>> print(X.shape)
        >>> torch.Size([2, 4, 10])
    '''

    dropout = nn.Dropout(dropout_p)
    P = torch.zeros((1, max_len, num_features))  # 行
    X_ = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
        10000,
        torch.arange(0, num_features, 2,
                     dtype=torch.float32) / num_features)  # 2i/d_model    num_features -> d_model  i维度指d_model章的维度
    P[:, :, 0::2] = torch.sin(X_)  # 偶数列用sin 奇数列用cos
    P[:, :, 1::2] = torch.cos(X_)
    X = X + P[:, :X.shape[1], :].to(X.device)  # X是已经embedding P复制batchsize份 加到X上
    return dropout(X)

def _in_projection_packed(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        w: Tensor,
        b: Optional[Tensor] = None,
) -> List[Tensor]:
    r"""
    用一个大的权重参数矩阵进行线性变换
    参数:
        q, k, v: 对自注意来说，三者都是src；对于seq2seq模型，k和v是一致的tensor。
                 但它们的最后一维(num_features或者叫做embed_dim)都必须保持一致。
        w: 用以线性变换的大矩阵，按照q,k,v的顺序压在一个tensor里面。
        b: 用以线性变换的偏置，按照q,k,v的顺序压在一个tensor里面。
    形状:
        输入:
        - q: shape:`(..., E)`，E是词嵌入的维度（下面出现的E均为此意）。
        - k: shape:`(..., E)`
        - v: shape:`(..., E)`
        - w: shape:`(E * 3, E)`
        - b: shape:`E * 3`
        输出:
        - 输出列表 :`[q', k', v']`，q,k,v经过线性变换前后的形状都一致。
    """
    E = q.size(-1)
    # 若为自注意，则q = k = v = src，因此它们的引用变量都是src
    # 即k is v和q is k结果均为True
    # 若为seq2seq，k = v，因而k is v的结果是True
    if k is v:
        if q is k:
            # 自注意
            return nn.functional.linear(q, w, b).chunk(3, dim=-1)  # .chunk(3, dim=-1)  沿最后一维将tensor 分为3块
        # nn.functional.linear(input, weight, bias=None)   y=xAT+b   最后一维维度 ：3E
        else:
            # seq2seq模型
            w_q, w_kv = w.split([E, E * 2])
            if b is None:
                b_q = b_kv = None
            else:
                b_q, b_kv = b.split([E, E * 2])
            return (nn.functional.linear(q, w_q, b_q),) + nn.functional.linear(k, w_kv, b_kv).chunk(2, dim=-1)  # ？？+
    else:
        # k,v 不相等的情况
        w_q, w_k, w_v = w.chunk(3)
        if b is None:
            b_q = b_k = b_v = None
        else:
            b_q, b_k, b_v = b.chunk(3)
        return nn.functional.linear(q, w_q, b_q), nn.functional.linear(k, w_k, b_k), nn.functional.linear(v, w_v, b_v)

def _scaled_dot_product_attention(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attn_mask: Optional[Tensor] = None,
        dropout_p: float = 0.0,
) -> Tuple[Tensor, Tensor]:
    r'''
    在query, key, value上计算点积注意力，若有注意力遮盖则使用，并且应用一个概率为dropout_p的dropout
    参数：
        - q: shape:`(B, Nt, E)` B代表batch size， Nt是目标语言序列长度，E是嵌入后的特征维度(num_features或者叫做embed_dim)
        - key: shape:`(B, Ns, E)` Ns是源语言序列长度
        - value: shape:`(B, Ns, E)`与key形状一样
        - attn_mask: 要么是3D的tensor，形状为:`(B, Nt, Ns)`或者2D的tensor，形状如:`(Nt, Ns)`
        - Output: attention values: shape:`(B, Nt, E)`，与q的形状一致;attention weights: shape:`(B, Nt, Ns)`
    补充：该函数的输出长度为2，out[0]是自注意力机制(Scaled Dot-Product Attention)的最终输出(见Attention is all you need)中的图2左、公式（1）的最终输出
                          out[1]是图2左中的Q矩阵和K矩阵MatMul之后的结果
    例子：
        >>> q = torch.randn((2,3,6))
        torch.manual_seed(0)
        >>> k = torch.randn((2,4,6))
        torch.manual_seed(0)
        >>> v = torch.randn((2,4,6))
        >>> out = _scaled_dot_product_attention(q, k, v)
        >>> out[0].shape, out[1].shape
        >>> torch.Size([2, 3, 6]) torch.Size([2, 3, 4])
    '''
    B, Nt, E = q.shape
    q = q / math.sqrt(E)  # math.sqrt(E) 即根号dk
    # (B, Nt, E) x (B, E, Ns) -> (B, Nt, Ns)
    attn = torch.bmm(q, k.transpose(-2,
                                    -1))  # k.transpose（-2，-1）是将k进行维度交换,将其维度变为torch.Size([2, 6, 4]);torch.bmm用来计算两个tensor的矩阵乘法,tensor矩阵的第一维好像只能是2？？？
    # torch.bmm 矩阵乘法 两个矩阵要求都为3维的 第一维维度相同 后两维要满足矩阵乘法
    if attn_mask is not None:  # ？？？注意力遮盖
        attn += attn_mask
    # attn意味着目标序列的每个词对源语言序列做注意力
    attn = nn.functional.softmax(attn, dim=-1)
    if dropout_p > 0.0:  # mask 用dropout实现
        attn = nn.functional.dropout(attn, p=dropout_p)
    # (B, Nt, Ns) x (B, Ns, E) -> (B, Nt, E)
    output = torch.bmm(attn, v)
    return output, attn


def multi_head_attention_forward(  # Multi-Head Attention模块内部的网络架构
        query: Tensor,
        key: Tensor,
        value: Tensor,
        num_heads: int,
        in_proj_weight: Tensor,
        in_proj_bias: Optional[Tensor],
        dropout_p: float,
        out_proj_weight: Tensor,
        out_proj_bias: Optional[Tensor],
        training: bool = True,
        key_padding_mask: Optional[Tensor] = None,  # ？？
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
        use_separate_proj_weight: bool = False,
        q_proj_weight: Optional[Tensor] = None,
        k_proj_weight: Optional[Tensor] = None,
        v_proj_weight: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r'''
    形状：
        输入：
        - query：`(L, N, E)`
        - key: `(S, N, E)`        # (seq_len, batch, embed_dim)
        - value: `(S, N, E)`
        - key_padding_mask: `(N, S)`
        - attn_mask: `(L, S)` or `(N * num_heads, L, S)`
        输出：
        - attn_output:`(L, N, E)`
        - attn_output_weights:`(N, L, S)`
    '''

    tgt_len, bsz, embed_dim = query.shape
    src_len, _, _ = key.shape
    head_dim = embed_dim // num_heads  # 每一个 head 的维度
    q, k, v = _in_projection_packed(query, key, value, in_proj_weight, in_proj_bias)  # 投影获得新的q,k,v

    if attn_mask is not None:
        if attn_mask.dtype == torch.uint8:
            warnings.warn("Byte tensor for attn_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
            attn_mask = attn_mask.to(torch.bool)
        else:
            assert attn_mask.is_floating_point() or attn_mask.dtype == torch.bool, f"Only float, byte, and bool types are supported for attn_mask, not {attn_mask.dtype}"
            # 判断attn_mask是否是浮点类型或者bool类型
            # assert 检查程序 不符合条件终止程序

        if attn_mask.dim() == 2:  # 判断attn_mask的维度
            correct_2d_size = (tgt_len, src_len)
            if attn_mask.shape != correct_2d_size:
                raise RuntimeError(
                    f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
            attn_mask = attn_mask.unsqueeze(0)  # 若维度为2 在第一维添加一维
        elif attn_mask.dim() == 3:
            correct_3d_size = (bsz * num_heads, tgt_len, src_len)
            if attn_mask.shape != correct_3d_size:
                raise RuntimeError(
                    f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
        else:
            raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

    if key_padding_mask is not None and key_padding_mask.dtype == torch.uint8:  # ？？？
        warnings.warn(
            "Byte tensor for key_padding_mask in nn.MultiheadAttention is deprecated. Use bool tensor instead.")
        key_padding_mask = key_padding_mask.to(torch.bool)  # 利用torch.bool 完成mask

    # reshape q,k,v将Batch放在第一维以适合点积注意力
    # 同时为多头机制，将不同的头拼在一起组成一层  ？？？
    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0,
                                                                          1)  # q.contiguous()  强制拷贝一份tensor 使它的布局和从头开始创建一样
    k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if key_padding_mask is not None:
        assert key_padding_mask.shape == (bsz, src_len), \
            f"expecting key_padding_mask shape of {(bsz, src_len)}, but got {key_padding_mask.shape}"
        key_padding_mask = key_padding_mask.view(bsz, 1, 1, src_len). \
            expand(-1, num_heads, -1, -1).reshape(bsz * num_heads, 1, src_len)  # ？？？？
        if attn_mask is None:
            attn_mask = key_padding_mask
        elif attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.logical_or(key_padding_mask)  # 逻辑或
        else:
            attn_mask = attn_mask.masked_fill(key_padding_mask,
                                              float("-inf"))  # 用-inf 填充 key_padding_mask 值为1对应的attn_mask中的元素
    # 若attn_mask值是布尔值，则将mask转换为float
    if attn_mask is not None and attn_mask.dtype == torch.bool:
        new_attn_mask = torch.zeros_like(attn_mask, dtype=torch.float)
        new_attn_mask.masked_fill_(attn_mask, float("-inf"))
        attn_mask = new_attn_mask

    # 若training为True时才应用dropout
    if not training:
        dropout_p = 0.0
    attn_output, attn_output_weights = _scaled_dot_product_attention(q, k, v, attn_mask, dropout_p)
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    attn_output = nn.functional.linear(attn_output, out_proj_weight, out_proj_bias)
    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None

class MultiheadAttention(nn.Module):
    r'''
    参数：
        embed_dim: 词嵌入的维度
        num_heads: 平行头的数量
        batch_first: 若`True`，则为(batch, seq, feature)，若为`False`，则为(seq, batch, feature)

    例子：
        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)  #embed_dim必须能被num_heads整除！
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    '''

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, kdim=None, vdim=None,
                 batch_first=False) -> None:
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim)))
            self.register_parameter('in_proj_weight', None)
        else:
            self.in_proj_weight = Parameter(torch.empty((3 * embed_dim, embed_dim)))
            self.register_parameter('q_proj_weight', None)
            self.register_parameter('k_proj_weight', None)
            self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)  # 初始化
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]  # N,S,E ->S,N,E

        if not self._qkv_same_embed_dim:  # 不是自注意力的情况下
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:  # 自注意力
            attn_output, attn_output_weights = multi_head_attention_forward(
                query, key, value, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)
        if self.batch_first:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class TransformerEncoderLayer(nn.Module):
    r'''
    参数：
        d_model: 词嵌入的维度
        nhead: 多头注意力中平行头的数目
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)
    例子：
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)  #注意，d_model必须能被nhead整除！
        >>> src = torch.randn((32, 10, 512))
        >>> out = encoder_layer(src)
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        src2 = self.self_attn(src, src, src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[
            0]  # 这里[0]说明只取MultiheadAttention的最终的输出矩阵，所以后面的例子设置的输入矩阵的维度顺序又变正常了(batchsize、序列长度、词嵌入维度)
        src = src + self.dropout1(src2)  # 残差连接
        src = self.norm1(src)  # 层归一化
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))  # 全连接(Feed-Forward)
        src = src + self.dropout(src2)  # 残差连接
        src = self.norm2(src)  # 层归一化
        return src  # 编码器层输出

class TransformerEncoder(nn.Module):
    r'''
    参数：
        encoder_layer
        num_layers： encoder_layer的层数
        norm: 归一化的选择

    例子：
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.randn((10, 32, 512))  #seq_length batch d_model
        >>> out = transformer_encoder(src)
    '''

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layer = encoder_layer
        self.num_layers = num_layers
        self.norm = norm

    #        self.fc=nn.Linear(10,20)   #原始特征维度 →embedd_dim

    def forward(self, src: Tensor, mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        #        output = positional_encoding(src, src.shape[-1])

        #        output=self.fc(src)
        output = src
        for _ in range(self.num_layers):
            output = self.layer(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            output = self.norm(output)

        return output

class TransformerModel(nn.Module):
    def __init__(self, d_model:int, use_fc=False):
        super(TransformerModel, self).__init__()
        self.d_model = d_model
        self.use_fc = use_fc
        # 注意，这里将数据维度变化包括维度转换都放在了网络中进行!!!最好这样，因为这样能训练上！
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=4, stride=1, padding=1)
        self.bn = nn.BatchNorm1d(1)
        self.transformer_encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layer=self.transformer_encoder_layer,num_layers=1)
        self.fc = nn.Linear(100 * 1, 4)

    def forward(self, x): # (64,1,128)
        # x = torch.unsqueeze(x, dim=1)
        x = self.conv(x) # (64,1,100)
        x = self.bn(x)
        x = F.relu(x)
        # x = torch.squeeze(x)
        # x = x.transpose(1, 2)
        x = x + positional_encoding(x, self.d_model) # (batch, seq, feture)
        x = self.transformer_encoder(x)
        x = x.view(x.shape[0], -1)
        if self.use_fc:
            x = self.fc(x)
        return x

class TransformerEncoderLayer_Double(nn.Module):
    r'''
    参数：
        d_model: 词嵌入的维度
        nhead: 多头注意力中平行头的数目
        dim_feedforward: 全连接层的神经元的数目，又称经过此层输入的维度（Default = 2048）
        dropout: dropout的概率
        activation: 两个线性层中间的激活函数，默认relu或gelu
        lay_norm_eps: layer normalization中的微小量，防止分母为0
        batch_first: 若`True`，则为(batch, seq, feture)，若为`False`，则为(seq, batch, feature)
    例子：
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)  #注意，d_model必须能被nhead整除！
        >>> src = torch.randn((32, 10, 512))
        >>> out = encoder_layer(src)
    '''

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.mish,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer_Double, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_q: Tensor, x_kv: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """
        x_q1=self.norm1(x_q)
        x_k=self.norm1(x_kv)
        x_v=self.norm1(x_kv)

        x=self.self_attn(x_q1,x_k,x_v,attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0] # 这里[0]说明只取MultiheadAttention的最终的输出矩阵，所以后面的例子设置的输入矩阵的维度顺序又变正常了(batchsize、序列长度、词嵌入维度)

        x=x_q1+self.dropout1(x) # 残差连接
        x=self.norm2(x) #层归一化
        x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))  # 全连接
        x=x+self.dropout2(x2) # 残差连接
        x=self.norm2(x) # 层归一化

        return x

class TransformerEncoderLayer_Treble(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.mish,
                 layer_norm_eps=1e-5, batch_first=False) -> None:
        super(TransformerEncoderLayer_Treble, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.activation = activation
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x_q: Tensor, x_k: Tensor, x_v: Tensor, src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        x_q1=self.norm1(x_q)
        x_k=self.norm1(x_k)
        x_v=self.norm1(x_v)

        x=self.self_attn(x_q1,x_k,x_v,attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0] # 这里[0]说明只取MultiheadAttention的最终的输出矩阵，所以后面的例子设置的输入矩阵的维度顺序又变正常了(batchsize、序列长度、词嵌入维度)
        x=x_q1+self.dropout1(x) # 残差连接
        x=self.norm2(x) #层归一化
        # x2 = self.linear2(self.dropout(self.activation(self.linear1(x))))  # 全连接
        # x=x+self.dropout2(x2) # 残差连接
        # x=self.norm2(x) # 层归一化

        return x


if __name__ == '__main__':
    x = torch.zeros((64, 1, 101))
    model = TransformerModel(d_model=100, use_fc=True)
    output = model(x)
    print(output.shape)