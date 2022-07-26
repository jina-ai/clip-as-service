# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

from typing import TYPE_CHECKING

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model
import open_clip
from open_clip.openai import load_openai_model

if TYPE_CHECKING:
    import torch


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)

        model_url, md5sum = get_model_url_md5(name)
        if model_url:
            model_path = download_model(model_url, md5sum=md5sum)
            self._model = load_openai_model(model_path, device=device, jit=jit)
            self._model_name = name.split('::')[0]
        else:
            model_name, pretrained = name.split('::')
            self._model = open_clip.create_model(
                model_name, pretrained=pretrained, device=device, jit=jit
            )
            self._model_name = model_name

    @property
    def model_name(self):
        if self._model_name == 'ViT-L/14@336px':
            return 'ViT-L-14-336'
        elif self._model_name.endswith('-quickgelu'):
            return self._model_name[:-10]
        return self._model_name.replace('/', '-')

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.encode_text(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        return self._model.encode_image(pixel_values)
