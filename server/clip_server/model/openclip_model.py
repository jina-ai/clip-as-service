# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

from typing import TYPE_CHECKING
from copy import deepcopy
import warnings
import torch

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model

from open_clip.model import (
    CLIP,
    convert_weights_to_fp16,
    build_model_from_openai_state_dict,
)
from open_clip.factory import _MODEL_CONFIGS, load_state_dict, load_openai_model

if TYPE_CHECKING:
    import torch


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)

        if '::' in name:
            model_name, pretrained = name.split('::')
        else:  # older CaS version's name format
            model_name = name
            pretrained = 'openai'
        if model_name.endswith('-quickgelu'):
            model_name = model_name[:-10]
        model_url, md5sum = get_model_url_md5(name)
        model_path = download_model(model_url, md5sum=md5sum)
        if pretrained.lower() == 'openai':
            model = load_openai_model(model_path, device=device, jit=jit)
        else:
            if model_name in _MODEL_CONFIGS:
                model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
            else:
                raise RuntimeError(f'Model config for {model_name} not found.')

            model = CLIP(**model_cfg)
            model.load_state_dict(load_state_dict(model_path))
            model.to(device=torch.device(device))

            if device == 'cuda':
                convert_weights_to_fp16(model)
            if jit:
                model = torch.jit.script(model)
        self._model_name = model_name
        self._model = model

    @property
    def model_name(self):
        if self._model_name == 'ViT-L/14@336px':
            return 'ViT-L-14-336'
        return self._model_name.replace('/', '-')

    def encode_text(
        self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor', **kwargs
    ):
        return self._model.encode_text(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor'):
        return self._model.encode_image(pixel_values)
