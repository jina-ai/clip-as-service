import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from torch.nn.functional import linear
from flash_attn.flash_attn_interface import flash_attn_unpadded_func


class MultiheadAttention(nn.MultiheadAttention):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            embed_dim,
            num_heads,
            dropout,
            bias,
            add_bias_kv,
            add_zero_attn,
            kdim,
            vdim,
            batch_first,
            device,
            dtype,
        )

    def attention(
        self,
        q,
        k,
        v,
        batch_size=1,
        seqlen=77,
        softmax_scale=None,
        attention_dropout=0.0,
        causal=False,
        cu_seqlens=None,
        max_s=None,
        need_weights=False,
    ):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            q,k,v: The tensor containing the query, key, and value. each of (B*S, H, D)
            key_padding_mask: a bool tensor of shape (B, S)

        """

        if cu_seqlens is None:
            max_s = seqlen
            cu_seqlens = torch.arange(
                0,
                (batch_size + 1) * seqlen,
                step=seqlen,
                dtype=torch.int32,
                device=q.device,
            )
            output = flash_attn_unpadded_func(
                q,
                k,
                v,
                cu_seqlens,
                cu_seqlens,
                max_s,
                max_s,
                attention_dropout,
                softmax_scale=softmax_scale,
                causal=causal,
            )

        return output

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[Tensor] = None,
        average_attn_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # set up shape vars
        seqlen, batch_size, embed_dim = query.shape

        # in-projection and rearrange `b s (h d) -> (b s) h d`
        q, k, v = linear(query, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = (
            q.transpose(0, 1)
            .contiguous()
            .view(batch_size * seqlen, self.num_heads, self.head_dim)
        )
        k = (
            k.transpose(0, 1)
            .contiguous()
            .view(batch_size * seqlen, self.num_heads, self.head_dim)
        )
        v = (
            v.transpose(0, 1)
            .contiguous()
            .view(batch_size * seqlen, self.num_heads, self.head_dim)
        )

        # flash attention (use causal mask)
        causal = attn_mask is not None
        attn_output = self.attention(q, k, v, batch_size, seqlen, causal=causal)

        # out-projection
        # `(b s) h d -> s b (h d)`
        attn_output = attn_output.contiguous().view(
            batch_size, seqlen, self.num_heads, self.head_dim
        )
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seqlen, batch_size, embed_dim)
        )
        attn_output = linear(attn_output, self.out_proj.weight, self.out_proj.bias)

        return attn_output, None
