# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Groupfree 3D Multi-head Attention Module"""

import warnings
import mindspore as ms
import mindspore.numpy as np
from mindspore.nn import Dense
from mindspore.common.initializer import XavierUniform
from mindspore.common.initializer import initializer

from mindspore import Parameter
from mindspore.nn import Cell
import mindspore.ops as ops


class MultiheadAttention(Cell):
    r"""Allows the model to jointly attend to information
    from different representation subspaces.
    See reference: Attention Is All You Need

    .. math::
        \text{MultiHead}(Q, K, V) = \text{Concat}(head_1,\dots,head_h)W^O
        \text{where} head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)

    Args:
        embed_dim: total dimension of the model.
        num_heads: parallel attention heads.
        dropout: a Dropout layer on attn_output_weights. Default: 0.0.
        bias: add bias as module parameter. Default: True.
        add_bias_kv: add bias to the key and value sequences at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        kdim: total number of features in key. Default: None.
        vdim: total number of features in key. Default: None.

        Note: if kdim and vdim are None, they will be set to embed_dim such that
        query, key, and value have the same number of features.

    Examples::

        >>> multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        >>> attn_output, attn_output_weights = multihead_attn(query, key, value)
    """

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None,
                 vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.in_proj_weight = Parameter(ms.numpy.empty((3 * embed_dim, embed_dim)))

        if self._qkv_same_embed_dim is False:
            self.q_proj_weight = Parameter(ms.Tensor((embed_dim, embed_dim)))
            self.k_proj_weight = Parameter(ms.Tensor((embed_dim, self.kdim)))
            self.v_proj_weight = Parameter(ms.Tensor((embed_dim, self.vdim)))

        if bias:
            self.in_proj_bias = Parameter(ms.numpy.empty(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = Dense(embed_dim, embed_dim, has_bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(ms.numpy.empty(1, 1, embed_dim))
            self.bias_v = Parameter(ms.numpy.empty(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            self.in_proj_weight = initializer(XavierUniform(), self.in_proj_weight.shape)
        else:
            self.q_proj_weight = initializer(XavierUniform(), self.q_proj_weight.shape)
            self.k_proj_weight = initializer(XavierUniform(), self.k_proj_weight.shape)
            self.v_proj_weight = initializer(XavierUniform(), self.v_proj_weight.shape)

        if self.in_proj_bias is not None:
            self.in_proj_bias = initializer(0., self.in_proj_bias.shape)
            self.out_proj.bias = initializer(0., self.out_proj.bias.shape)
        if self.bias_k is not None:
            self.bias_k = initializer('he_normal', self.bias_k.shape)
        if self.bias_v is not None:
            self.bias_v = initializer('he_normal', self.bias_v.shape)

    def construct(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).

    Shape:
        - Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.

        - Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
        """
        if hasattr(self, '_qkv_same_embed_dim') and self._qkv_same_embed_dim is False:
            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight)
        else:
            if not hasattr(self, '_qkv_same_embed_dim'):
                warnings.warn('A new version of MultiheadAttention module has been implemented. \
                    Please re-train your model with the new module',
                              UserWarning)

            return multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

def linear(input, weight, bias=None):
    return ops.matmul(input, weight.T) + bias

def unwrap_optional(x):
    assert x is not None, "Unwrapping null optional"
    return x

def multi_head_attention_forward(query, 
                                 key,
                                 value,
                                 embed_dim_to_check,
                                 num_heads,
                                 in_proj_weight,
                                 in_proj_bias,
                                 bias_k,
                                 bias_v,
                                 add_zero_attn,
                                 dropout_p,
                                 out_proj_weight,
                                 out_proj_bias,
                                 training=True,
                                 key_padding_mask=None,
                                 need_weights=True,
                                 attn_mask=None,
                                 use_separate_proj_weight=False,
                                 q_proj_weight=None,
                                 k_proj_weight=None,
                                 v_proj_weight=None,
                                 static_k=None,
                                 static_v=None,
                                 ):
    r"""
    Args:
        query, key, value: map a query and a set of key-value pairs to an output.
            See "Attention Is All You Need" for more details.
        embed_dim_to_check: total dimension of the model.
        num_heads: parallel attention heads.
        in_proj_weight, in_proj_bias: input projection weight and bias.
        bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
        add_zero_attn: add a new batch of zeros to the key and
                       value sequences at dim=1.
        dropout_p: probability of an element to be zeroed.
        out_proj_weight, out_proj_bias: the output projection weight and bias.
        training: apply dropout if is ``True``.
        key_padding_mask: if provided, specified padding elements in the key will
            be ignored by the attention. This is an binary mask. When the value is True,
            the corresponding value on the attention layer will be filled with -inf.
        need_weights: output attn_output_weights.
        attn_mask: mask that prevents attention to certain positions. This is an additive mask
            (i.e. the values will be added to the attention layer).
        use_separate_proj_weight: the function accept the proj. weights for query, key,
            and value in differnt forms. If false, in_proj_weight will be used, which is
            a combination of q_proj_weight, k_proj_weight, v_proj_weight.
        q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
        static_k, static_v: static key and value used for attention operators.

    Shape:
        Inputs:
        - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
          the embedding dimension.
        - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
          the embedding dimension.
        - key_padding_mask: :math:`(N, S)`, ByteTensor, where N is the batch size, S is the source sequence length.
        - attn_mask: :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
        - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
          N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
        Outputs:
        - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
          E is the embedding dimension.
        - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
          L is the target sequence length, S is the source sequence length.
    """

    eq = ops.Equal()
    cat = ops.Concat()
    zeros = ops.Zeros()
    qkv_same = np.array_equal(query, key) and np.array_equal(key, value)
    kv_same = np.array_equal(key, value)

    tgt_len, bsz, embed_dim = query.shape
    assert embed_dim == embed_dim_to_check
    assert list(query.shape) == [tgt_len, bsz, embed_dim]
    assert key.shape == value.shape

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if use_separate_proj_weight is not True:
        if qkv_same:
            # self-attention
            split = ops.Split(-1, 3)
            q, k, v = split(linear(query, in_proj_weight, in_proj_bias))

        elif kv_same:
            # encoder-decoder attention
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            if key is None:
                assert value is None
                k = None
                v = None
            else:

                # This is inline in_proj function with in_proj_weight and in_proj_bias
                _b = in_proj_bias
                _start = embed_dim
                _end = None
                _w = in_proj_weight[_start:, :]
                if _b is not None:
                    _b = _b[_start:]
                split = ops.Split(-1, 2)
                k, v = split(linear(key, _w, _b))

        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = 0
            _end = embed_dim
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            q = linear(query, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim
            _end = embed_dim * 2
            _w = in_proj_weight[_start:_end, :]
            if _b is not None:
                _b = _b[_start:_end]
            k = linear(key, _w, _b)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            _b = in_proj_bias
            _start = embed_dim * 2
            _end = None
            _w = in_proj_weight[_start:, :]
            if _b is not None:
                _b = _b[_start:]
            v = linear(value, _w, _b)
    else:
        q_proj_weight_non_opt = unwrap_optional(q_proj_weight)
        len1, len2 = q_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == query.shape[-1]

        k_proj_weight_non_opt = unwrap_optional(k_proj_weight)
        len1, len2 = k_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == key.shape[-1]

        v_proj_weight_non_opt = unwrap_optional(v_proj_weight)
        len1, len2 = v_proj_weight_non_opt.shape
        assert len1 == embed_dim and len2 == value.shape[-1]

        if in_proj_bias is not None:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias[0:embed_dim])
            k = linear(key, k_proj_weight_non_opt, in_proj_bias[embed_dim:(embed_dim * 2)])
            v = linear(value, v_proj_weight_non_opt, in_proj_bias[(embed_dim * 2):])
        else:
            q = linear(query, q_proj_weight_non_opt, in_proj_bias)
            k = linear(key, k_proj_weight_non_opt, in_proj_bias)
            v = linear(value, v_proj_weight_non_opt, in_proj_bias)
    q = q * scaling

    if bias_k is not None and bias_v is not None:
        if static_k is None and static_v is None:
            k = cat([k, bias_k.repeat(1, bsz, 1)])
            v = cat([v, bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = cat([attn_mask,
                                       zeros((attn_mask.shape[0], 1),
                                                   dtype=attn_mask.dtype,
                                                   device=attn_mask.device)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = cat(
                    [key_padding_mask, zeros((key_padding_mask.shape[0], 1),
                                                   dtype=key_padding_mask.dtype,
                                                   device=key_padding_mask.device)], dim=1)
        else:
            assert static_k is None, "bias cannot be added to static key."
            assert static_v is None, "bias cannot be added to static value."
    else:
        assert bias_k is None
        assert bias_v is None

    transpose = ops.Transpose()
    q = transpose(q.view((tgt_len, bsz * num_heads, head_dim)), (1, 0, 2))
    if k is not None:
        k = transpose(k.view((-1, bsz * num_heads, head_dim)), (1, 0, 2))
    if v is not None:
        v = transpose(v.view((-1, bsz * num_heads, head_dim)), (1, 0, 2))

    if static_k is not None:
        assert static_k.shape[0] == bsz * num_heads
        assert static_k.shape[2] == head_dim
        k = static_k

    if static_v is not None:
        assert static_v.shape[0] == bsz * num_heads
        assert static_v.shape[2] == head_dim
        v = static_v

    src_len = k.shape[1]

    if key_padding_mask is not None:
        assert key_padding_mask.shape[0] == bsz
        assert key_padding_mask.shape[1] == src_len

    if add_zero_attn:
        src_len += 1

        k = cat([k, zeros((k.shape[0], 1) + k.shape[2:], dtype=k.dtype, device=k.device)], dim=1)
        v = cat([v, zeros((v.shape[0], 1) + v.shape[2:], dtype=v.dtype, device=v.device)], dim=1)
        if attn_mask is not None:
            attn_mask = cat([attn_mask, zeros((attn_mask.shape[0], 1),
                                                          dtype=attn_mask.dtype,
                                                          device=attn_mask.device)], dim=1)
        if key_padding_mask is not None:
            key_padding_mask = cat(
                [key_padding_mask, zeros((key_padding_mask.shape[0], 1),
                                               dtype=key_padding_mask.dtype,
                                               device=key_padding_mask.device)], dim=1)

    bmm = ops.BatchMatMul()
    attn_output_weights = bmm(q, transpose(k, (0, 2, 1)))
    assert list(attn_output_weights.shape) == [bsz * num_heads, tgt_len, src_len]

    unsqueeze = ops.ExpandDims()
    if attn_mask is not None:
        attn_mask = unsqueeze(attn_mask, 0)
        attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
        attn_output_weights = attn_output_weights.masked_fill(
            unsqueeze(unsqueeze(key_padding_mask, 1), 2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view((bsz * num_heads, tgt_len, src_len))

    softmax = ops.Softmax(axis=-1)
    attn_output_weights = softmax(attn_output_weights)
    dropout = ms.nn.Dropout(dropout_p)
    dropout.set_train()
    attn_output_weights = dropout(attn_output_weights)

    transpose = ops.Transpose()
    attn_output = bmm(attn_output_weights, v)
    assert list(attn_output.shape) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = transpose(attn_output, (1, 0, 2)).view((tgt_len, bsz, embed_dim))
    attn_output = linear(attn_output, out_proj_weight, out_proj_bias)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view((bsz, num_heads, tgt_len, src_len))
        return attn_output, attn_output_weights.sum(axis=1) / num_heads
    else:
        return attn_output, None
