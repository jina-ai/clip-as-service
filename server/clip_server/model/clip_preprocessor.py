from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
)


class CLIPPreprocessor:
    def __new__(cls, model):
        if cls is CLIPPreprocessor:
            if model._model_name in _OPENCLIP_MODELS:
                from clip_server.model.openclip_preprocessor import OpenCLIPPreprocessor

                instance = super().__new__(OpenCLIPPreprocessor)
            elif model._model_name in _MULTILINGUALCLIP_MODELS:
                from clip_server.model.mclip_preprocessor import (
                    MultilingualCLIPPreprocessor,
                )

                instance = super().__new__(MultilingualCLIPPreprocessor)
            else:
                raise ValueError(
                    f'The CLIP preprocessor name=`{model._model_name}` is not supported.'
                )
        else:
            instance = super().__new__(cls)
        return instance