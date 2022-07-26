# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

from typing import TYPE_CHECKING
from copy import deepcopy
import torch

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model

from open_clip.model import (
    CLIP,
    convert_weights_to_fp16,
)
from open_clip.factory import _MODEL_CONFIGS, load_state_dict, load_openai_model

if TYPE_CHECKING:
    import torch


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)

        if '::' in name:
            model_name, pretrained = name.split('::')
        else:
            # default pretrained model is from openai
            model_name = name
            pretrained = 'openai'

        self._model_name = model_name

        model_url, md5sum = get_model_url_md5(name)
        model_path = download_model(model_url, md5sum=md5sum)
        if pretrained.lower() == 'openai':
            self._model = load_openai_model(model_path, device=device, jit=jit)
        else:
            if model_name in _MODEL_CONFIGS:
                model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
            else:
                raise RuntimeError(f'Model config for {model_name} not found.')

            self._model = CLIP(**model_cfg)

            state_dict = load_state_dict(model_path)
            self._model.load_state_dict(state_dict, strict=True)

            if str(device) == 'cuda':
                convert_weights_to_fp16(self._model)
            if jit:
                self._model = torch.jit.script(self._model)

            self._model.to(device=torch.device(device))
            self._model.eval()

    @property
    def model_name(self):
        if self._model_name == 'ViT-L/14@336px':
            return 'ViT-L-14-336'
        return self._model_name.replace('/', '-')

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.encode_text(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor'):
        return self._model.encode_image(pixel_values)
