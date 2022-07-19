import open_clip
import torch

from clip_server.model.clip_model import CLIPModel


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str, jit: bool):
        super().__init__(name, device, jit)
        name, pretrained = name.split('::')
        self._model = open_clip.create_model(
            name, pretrained=pretrained, device=device, jit=jit
        )

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        return self._model.encode_text(input_ids, **kwargs)

    def encode_image(self, pixel_values: torch.Tensor, **kwargs):
        return self._model.encode_image(pixel_values, **kwargs)
