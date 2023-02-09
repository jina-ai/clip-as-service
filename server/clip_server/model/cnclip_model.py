# Originally from https://github.com/OFA-Sys/Chinese-CLIP. MIT License.

import torch

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import _VISUAL_MODEL_IMAGE_SIZE
from cn_clip.clip import load_from_name

_CNCLIP_MODEL_MAPS = {
    'CN-CLIP/ViT-B-16': 'ViT-B-16',
    'CN-CLIP/ViT-L-14': 'ViT-L-14',
    'CN-CLIP/ViT-L-14-336': 'ViT-L-14-336',
    'CN-CLIP/ViT-H-14': 'ViT-H-14',
    'CN-CLIP/RN50': 'RN50',
}


class CNClipModel(CLIPModel):
    def __init__(
        self,
        name: str,
        device: str = 'cpu',
        jit: bool = False,
        dtype: str = None,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self._name = _CNCLIP_MODEL_MAPS[name]

        self._model, self._preprocess = load_from_name(
            _CNCLIP_MODEL_MAPS[name], device=device
        )
        self._model.eval()

    @staticmethod
    def get_model_name(name: str):
        return _CNCLIP_MODEL_MAPS[name]

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.encode_text(input_ids).detach()

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        return self._model.encode_image(pixel_values).detach()

    @property
    def model_name(self):
        return self.__class__.get_model_name(self._name)

    @property
    def image_size(self):
        return _VISUAL_MODEL_IMAGE_SIZE.get(self._name, None)
