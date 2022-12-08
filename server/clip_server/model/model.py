""" CLIP Model

Adapted from https://github.com/mlfoundations/open_clip.

Originally MIT License, Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
Ludwig Schmidt
"""

import warnings
import torch
import numpy as np
from torch import nn
from dataclasses import dataclass
from typing import Tuple, Union, Optional
from copy import deepcopy
from clip_server.helper import __cast_dtype__
from open_clip.transformer import QuickGELU, LayerNorm, LayerNormFp32, Attention
from open_clip.timm_model import TimmModel
from open_clip.factory import _MODEL_CONFIGS
from open_clip.hf_model import PreTrainedTextEncoder
from open_clip.transformer import ResidualAttentionBlock as _ResidualAttentionBlock
from open_clip.transformer import Transformer as _Transformer
from open_clip.transformer import VisionTransformer as _VisionTransformer
from open_clip.transformer import TextTransformer as _TextTransformer
from open_clip.modified_resnet import ModifiedResNet as _ModifiedResNet
from open_clip.model import CustomTextCLIP as _CustomTextCLIP
from open_clip.model import CLIP as _CLIP

# Use flash attention
try:
    from clip_server.model.flash_attention import MultiheadAttention

    FLASH_ATTENTION_AVAILABLE = True
except:
    FLASH_ATTENTION_AVAILABLE = False


class ModifiedResNet(_ModifiedResNet):
    def forward(self, x):
        # To handle fp16 inference
        x = x.type(self.conv1.weight.dtype)
        return super().forward(x)


class ResidualAttentionBlock(_ResidualAttentionBlock):
    def __init__(
        self, width: int, heads: int, dtype: torch.dtype = torch.float32, **kwargs
    ):
        super().__init__(width, heads, **kwargs)
        head_dim = width // heads
        flash_attention = head_dim % 8 == 0 and head_dim <= 128

        self.attn = (
            MultiheadAttention(width, heads)
            if FLASH_ATTENTION_AVAILABLE
            and torch.cuda.is_available()
            and dtype in (torch.float16, torch.bfloat16)
            and flash_attention
            else nn.MultiheadAttention(width, heads)
        )


class Transformer(_Transformer):
    def __init__(self, layers: int, dtype: torch.dtype = torch.float32, **kwargs):
        super().__init__(layers=layers, **kwargs)
        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(dtype=dtype, **kwargs) for _ in range(layers)]
        )


class VisionTransformer(_VisionTransformer):
    def __init__(
        self,
        image_size: int,
        patch_size: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(image_size, patch_size, output_dim=output_dim, **kwargs)
        self.transformer = Transformer(dtype=dtype, **kwargs)

    def forward(self, x: torch.Tensor):
        dtype = self.transformer.get_cast_dtype()
        x = x.to(dtype)
        return super().forward(x)


class TextTransformer(_TextTransformer):
    def __init__(
        self,
        context_length: int,
        vocab_size: int,
        output_dim: int,
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(context_length, vocab_size, output_dim=output_dim, **kwargs)
        self.transformer = Transformer(dtype=dtype, **kwargs)
        self.init_parameters()


@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    ls_init_value: Optional[float] = None  # layer scale initial value
    timm_model_name: str = (
        None  # a valid model name overrides layers, width, patch_size
    )
    timm_model_pretrained: bool = (
        False  # use (imagenet) pretrained weights for named model
    )
    timm_pool: str = (
        'avg'  # feature pooling for timm model ('abs_attn', 'rot_attn', 'avg', '')
    )
    timm_proj: str = (
        'linear'  # linear projection for timm model output ('linear', 'mlp', '')
    )
    timm_proj_bias: bool = False  # enable bias final projection


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12
    ls_init_value: Optional[float] = None  # layer scale initial value
    hf_model_name: str = None
    hf_tokenizer_name: str = None
    proj: str = 'mlp'
    pooler_type: str = 'mean_pooler'


def _build_vision_tower(
    embed_dim: int,
    vision_cfg: CLIPVisionCfg,
    quick_gelu: bool = False,
    dtype: Optional[torch.dtype] = torch.float32,
):
    if isinstance(vision_cfg, dict):
        vision_cfg = CLIPVisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    # NOTE: timm models always use native GELU regardless of quick_gelu flag.
    act_layer = QuickGELU if quick_gelu else nn.GELU

    if vision_cfg.timm_model_name:
        visual = TimmModel(
            model_name=vision_cfg.timm_model_name,
            pretrained=vision_cfg.timm_model_pretrained,
            pool=vision_cfg.timm_pool,
            proj=vision_cfg.timm_proj,
            proj_bias=vision_cfg.timm_proj_bias,
            embed_dim=embed_dim,
            image_size=vision_cfg.image_size,
        )
        act_layer = (
            nn.GELU
        )  # so that text transformer doesn't use QuickGELU w/ timm models
    elif isinstance(vision_cfg.layers, (tuple, list)):
        vision_heads = vision_cfg.width * 32 // vision_cfg.head_width
        visual = ModifiedResNet(
            layers=vision_cfg.layers,
            output_dim=embed_dim,
            heads=vision_heads,
            image_size=vision_cfg.image_size,
            width=vision_cfg.width,
        )
    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = (
            LayerNormFp32 if dtype in (torch.float16, torch.bfloat16) else LayerNorm
        )
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dtype=dtype,
        )

    return visual


def _build_text_tower(
    embed_dim: int,
    text_cfg: CLIPTextCfg,
    quick_gelu: bool = False,
    dtype: Optional[torch.dtype] = torch.float32,
):
    if isinstance(text_cfg, dict):
        text_cfg = CLIPTextCfg(**text_cfg)

    if text_cfg.hf_model_name:
        text = PreTrainedTextEncoder(
            text_cfg.hf_model_name,
            output_dim=embed_dim,
            proj=text_cfg.proj,
            pooler_type=text_cfg.pooler_type,
        )
    else:
        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32 if dtype in (torch.float16, torch.bfloat16) else LayerNorm
        )

        text = TextTransformer(
            context_length=text_cfg.context_length,
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            heads=text_cfg.heads,
            layers=text_cfg.layers,
            ls_init_value=text_cfg.ls_init_value,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            dtype=dtype,
        )
    return text


class CustomTextCLIP(_CustomTextCLIP):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        super().__init__(embed_dim, vision_cfg, text_cfg, quick_gelu, dtype)
        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            dtype=dtype,
        )
        self.text = _build_text_tower(
            embed_dim=embed_dim, text_cfg=text_cfg, quick_gelu=quick_gelu, dtype=dtype
        )


class CLIP(_CLIP):
    def __init__(
        self,
        embed_dim: int,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        quick_gelu: bool = False,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        nn.Module.__init__(self)

        self.visual = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            dtype=dtype,
        )
        text = _build_text_tower(
            embed_dim=embed_dim, text_cfg=text_cfg, quick_gelu=quick_gelu, dtype=dtype
        )
        self.transformer = text.transformer
        self.vocab_size = text.vocab_size
        self.token_embedding = text.token_embedding
        self.positional_embedding = text.positional_embedding
        self.ln_final = text.ln_final
        self.text_projection = text.text_projection
        self.register_buffer('attn_mask', text.attn_mask, persistent=False)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [
                *[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]],
                "in_proj_bias",
                "bias_k",
                "bias_v",
            ]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def build_model_from_openai_state_dict(
    state_dict: dict,
    quick_gelu: bool = False,
    dtype: torch.dtype = torch.float16,
):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [
                k
                for k in state_dict.keys()
                if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
            ]
        )
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5
        )
        image_size = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2]
                    for k in state_dict
                    if k.startswith(f"visual.layer{b}")
                )
            )
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5
        )
        vision_patch_size = None
        assert (
            output_width**2 + 1
            == state_dict["visual.attnpool.positional_embedding"].shape[0]
        )
        image_size = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2]
            for k in state_dict
            if k.startswith(f"transformer.resblocks")
        )
    )

    vision_cfg = CLIPVisionCfg(
        layers=vision_layers,
        width=vision_width,
        patch_size=vision_patch_size,
        image_size=image_size,
    )
    text_cfg = CLIPTextCfg(
        context_length=context_length,
        vocab_size=vocab_size,
        width=transformer_width,
        heads=transformer_heads,
        layers=transformer_layers,
    )
    model = CLIP(
        embed_dim=embed_dim,
        vision_cfg=vision_cfg,
        text_cfg=text_cfg,
        quick_gelu=quick_gelu,  # OpenAI models were trained with QuickGELU
        dtype=dtype,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        state_dict.pop(key, None)

    convert_weights_to_fp16(model)
    model.load_state_dict(state_dict)

    return model.eval()


def load_openai_model(
    model_path: str,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    dtype: Optional[Union[str, torch.dtype]] = None,
    jit: bool = True,
):
    """Load a CLIP model

    Parameters
    ----------
    model_path : str
        The path to a model checkpoint containing the state_dict
    dtype: str
        Model precision, if None defaults to 'fp32' if device == 'cpu' else 'fp16'.
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.
    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if isinstance(dtype, str):
        dtype = __cast_dtype__.get(dtype, 'amp')
    elif dtype is None:
        dtype = (
            torch.float32 if device in ('cpu', torch.device('cpu')) else torch.float16
        )
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # Build a non-jit model from the OpenAI jitted model state dict
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict(), dtype=dtype
            )
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd, dtype=dtype)

        # model from OpenAI state dict is in manually cast fp16 mode, must be converted for AMP/fp32/bf16 use
        model = model.to(device)
        if dtype == torch.float32 or (
            isinstance(dtype, str) and dtype.startswith('amp')
        ):
            model.float()
        elif dtype == torch.bfloat16:
            convert_weights_to_lp(model, dtype=torch.bfloat16)

        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 (typically for CPU)
    if dtype == torch.float32:
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


def load_openclip_model(
    model_name: str,
    model_path: str,
    device: Union[str, torch.device] = 'cpu',
    jit: bool = False,
    force_quick_gelu: bool = False,
    force_custom_text: bool = False,
    pretrained_image: bool = False,
    dtype: Optional[Union[str, torch.dtype]] = None,
):
    if isinstance(dtype, str):
        dtype = __cast_dtype__.get(dtype)
    elif dtype is None:
        dtype = (
            torch.float32 if device in ('cpu', torch.device('cpu')) else torch.float16
        )

    model_name = model_name.replace(
        '/', '-'
    )  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    if pretrained_image:
        if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
            # pretrained weight loading for timm models set via vision_cfg
            model_cfg['vision_cfg']['timm_model_pretrained'] = True
        else:
            assert (
                False
            ), 'pretrained image towers currently only supported for timm models'

    custom_text = (
        model_cfg.pop('custom_text', False)
        or force_custom_text
        or ('hf_model_name' in model_cfg['text_cfg'])
    )

    if custom_text:
        model = CustomTextCLIP(**model_cfg, dtype=dtype)
    else:
        model = CLIP(**model_cfg, dtype=dtype)

    model.eval()
    model.load_state_dict(load_state_dict(model_path))
    model.to(device=device)

    if dtype in (torch.float16, torch.bfloat16):
        convert_weights_to_lp(model, dtype=dtype)

    if jit:
        model = torch.jit.script(model)

    return model
