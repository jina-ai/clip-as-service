from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
    _VISUAL_MODEL_IMAGE_SIZE,
)


class BaseCLIPModel:
    def __init__(self, name: str, **kwargs):
        super().__init__()
        self._name = name

    @staticmethod
    def get_model_name(name: str):
        return name

    @property
    def model_name(self):
        return self.__class__.get_model_name(self._name)

    @property
    def image_size(self):
        return _VISUAL_MODEL_IMAGE_SIZE.get(self.model_name, None)


class CLIPModel(BaseCLIPModel):
    def __new__(cls, name: str, **kwargs):
        if cls is CLIPModel:
            if name in _OPENCLIP_MODELS:
                from clip_server.model.openclip_model import OpenCLIPModel

                instance = super().__new__(OpenCLIPModel)
            elif name in _MULTILINGUALCLIP_MODELS:
                from clip_server.model.mclip_model import MultilingualCLIPModel

                instance = super().__new__(MultilingualCLIPModel)
            else:
                raise ValueError(
                    'CLIP model {} not found; below is a list of all available models:\n{}'.format(
                        name,
                        ''.join(
                            [
                                '\t- {}\n'.format(i)
                                for i in list(_OPENCLIP_MODELS.keys())
                                + list(_MULTILINGUALCLIP_MODELS.keys())
                            ]
                        ),
                    )
                )
        else:
            instance = super().__new__(cls)
        return instance
