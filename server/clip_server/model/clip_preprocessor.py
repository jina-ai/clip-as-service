from PIL import Image

from .pretrained_models import _OPENCLIP_MODELS, _MULTILINGUALCLIP_MODELS


class CLIPPreprocessor:
    def __new__(cls, model):
        """Create a CLIP feature extractor instance."""
        if cls is CLIPPreprocessor:
            if model._model_name in _OPENCLIP_MODELS:
                from .openclip_preprocessor import OpenCLIPPreprocessor

                instance = super().__new__(OpenCLIPPreprocessor)
            elif model._model_name in _MULTILINGUALCLIP_MODELS:
                from .mclip_preprocessor import MultilingualCLIPPreprocessor

                instance = super().__new__(MultilingualCLIPPreprocessor)
            else:
                raise ValueError(
                    f'The CLIP preprocessor name=`{model._model_name}` is not supported.'
                )
        else:
            instance = super().__new__(cls)
        return instance

    def tokenize(self, texts, **kwargs):
        return self._tokenizer(texts, **kwargs)

    def transform(self, images, **kwargs):
        return self._vision_preprocessor(Image.fromarray(images), **kwargs)
