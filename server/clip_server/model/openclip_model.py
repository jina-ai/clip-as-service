import open_clip

from .clip_model import CLIPModel


class OpenCLIPModel(CLIPModel):
    def __init__(self, name, device, jit):
        super().__init__(name, device, jit)
        name, pretrained = name.split('::')
        self._model = open_clip.create_model(
            name, pretrained=pretrained, device=device, jit=jit
        )
