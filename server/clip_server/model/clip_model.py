from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
    _VISUAL_MODEL_IMAGE_SIZE,
)


class CLIPModel:
    def __new__(cls, name: str, **kwargs):
        if cls is CLIPModel:
            if name in _OPENCLIP_MODELS:
                from clip_server.model.openclip_model import OpenCLIPModel

                instance = super().__new__(OpenCLIPModel)
            elif name in _MULTILINGUALCLIP_MODELS:
                from clip_server.model.mclip_model import MultilingualCLIPModel

                instance = super().__new__(MultilingualCLIPModel)
            else:
                raise ValueError(f'The CLIP model name=`{name}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, name: str, **kwargs):
        self._name = name

    @property
    def model_name(self):
        return self._name

    @property
    def image_size(self):
        return _VISUAL_MODEL_IMAGE_SIZE.get(self.model_name, None)
