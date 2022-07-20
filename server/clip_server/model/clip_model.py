from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
)


class CLIPModel:
    def __new__(cls, name: str):
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
