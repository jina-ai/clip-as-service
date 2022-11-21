# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt


from collections import OrderedDict
from typing import Callable, Optional

import torch
from torch import nn
from open_clip.transformer import LayerNorm
from open_clip.transformer import ResidualAttentionBlock as _ResidualAttentionBlock
from open_clip.transformer import Transformer as _Transformer
from open_clip.transformer import VisionTransformer as _VisionTransformer
from open_clip.transformer import TextTransformer as _TextTransformer

# Use flash attention
try:
    from clip_server.model.flash_attention import MultiheadAttention

    FLASH_ATTENTION_AVAILABLE = True
except:
    FLASH_ATTENTION_AVAILABLE = False


class ResidualAttentionBlock(_ResidualAttentionBlock):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        # scale_attn: bool = False,
        # scale_fc: bool = False,
        dtype=torch.float32,
    ):
        super().__init__(
            d_model, n_head, mlp_ratio, ls_init_value, act_layer, norm_layer
        )
        head_dim = d_model // n_head
        flash_attention = head_dim % 8 == 0 and head_dim <= 128

        self.attn = (
            MultiheadAttention(d_model, n_head)
            if FLASH_ATTENTION_AVAILABLE
            and torch.cuda.is_available()
            and dtype in (torch.float16, torch.bfloat16)
            and flash_attention
            else nn.MultiheadAttention(d_model, n_head)
        )


class Transformer(_Transformer):
    def __init__(
        self,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float = 4.0,
        ls_init_value: float = None,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        dtype=torch.float32,
    ):
        super().__init__(
            width, layers, heads, mlp_ratio, ls_init_value, act_layer, norm_layer
        )

        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    width,
                    heads,
                    mlp_ratio,
                    ls_init_value=ls_init_value,
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    dtype=dtype,
                )
                for _ in range(layers)
            ]
        )


class VisionTransformer(_VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        width: int,
        layers: int,
        heads: int,
        mlp_ratio: float,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        dtype=torch.float32,
    ):
        super().__init__(
            image_size,
            patch_size,
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value,
            output_dim,
            act_layer,
            norm_layer,
        )
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dtype=dtype,
        )


class TextTransformer(_TextTransformer):
    def __init__(
        self,
        context_length: int = 77,
        vocab_size: int = 49408,
        width: int = 512,
        heads: int = 8,
        layers: int = 12,
        ls_init_value: float = None,
        output_dim: int = 512,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = LayerNorm,
        dtype=torch.float32,
    ):
        super().__init__(
            context_length,
            vocab_size,
            width,
            heads,
            layers,
            ls_init_value,
            output_dim,
            act_layer,
            norm_layer,
        )

        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dtype=dtype,
        )
        self.init_parameters()
