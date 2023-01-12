# Originally from https://github.com/modelscope/modelscope. Apache License 2.0, Copyright 2022-2023 Alibaba ModelScope.

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model
from modelscope.models import Model

class DamoModel(CLIPModel):
    def __init__(
        self,
        name: str,
        device: str = 'cpu',
        jit: bool = False,
        dtype: str = None,
        **kwargs
    ):
        super().__init__()
        self._name = name

        self._model = Model.from_pretrained(name)

    @staticmethod
    def get_model_name(name: str):
        return name

    @property
    def model_name(self):
        return self.__class__.get_model_name(self._name)

    @property
    def image_size(self):
        return _VISUAL_MODEL_IMAGE_SIZE.get(self._name, None)
