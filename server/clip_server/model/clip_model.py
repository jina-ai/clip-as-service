from .pretrained_models import _OPENCLIP_MODELS, _MULTILINGUALCLIP_MODELS


class CLIPModel:
    def __new__(cls, name, device, jit):
        """Create a CLIP model instance."""
        if cls is CLIPModel:
            if name in _OPENCLIP_MODELS:
                from .openclip_model import OpenCLIPModel

                instance = super().__new__(OpenCLIPModel)
            elif name in _MULTILINGUALCLIP_MODELS:
                from .mclip_model import MultilingualCLIPModel

                instance = super().__new__(MultilingualCLIPModel)
            else:
                raise ValueError(f'The CLIP model name=`{name}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, name, device, jit):
        self._model_name = name
        self._device = device
        self._jit = jit

    def encode_text(self, input_ids, attention_mask, **kwargs):
        return self._model.encode_text(input_ids, **kwargs)

    def encode_image(self, pixel_values, **kwargs):
        return self._model.encode_image(pixel_values, **kwargs)
