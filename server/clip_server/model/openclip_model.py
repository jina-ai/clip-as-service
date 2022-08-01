# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt


from copy import deepcopy

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model
from clip_server.model.model import CLIP, convert_weights_to_fp16

from open_clip.openai import load_openai_model
from open_clip.factory import _MODEL_CONFIGS

import torch


def _load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def _load_model(
    model_name: str,
    model_path: str,
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
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

    model = CLIP(**model_cfg)
    model.eval()

    model.load_state_dict(_load_state_dict(model_path))

    if str(device).startswith('cuda'):
        convert_weights_to_fp16(model)

    model.to(device=device)

    if jit:
        model = torch.jit.script(model)

    return model


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)

        if '::' in name:
            model_name, pretrained = name.split('::')
        else:
            model_name = name
            pretrained = 'openai'

        self._model_name = model_name

        model_url, md5sum = get_model_url_md5(name)
        model_path = download_model(model_url, md5sum=md5sum)

        if pretrained == 'openai':
            self._model = load_openai_model(model_path, device=device, jit=jit)
        else:
            self._model = _load_model(
                self._model_name, model_path=model_path, device=device, jit=jit
            )

    @staticmethod
    def get_model_name(name: str):
        if '::' in name:
            model_name, pretrained = name.split('::')
        else:
            model_name = name
        if model_name == 'ViT-L/14@336px':
            return 'ViT-L-14-336'
        return model_name.replace('/', '-')

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.encode_text(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        return self._model.encode_image(pixel_values)
