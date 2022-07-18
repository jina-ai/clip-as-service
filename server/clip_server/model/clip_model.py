from clip_server.model.pretrained_models import _OPENAI_MODELS, _OPENCLIP_MODELS


class CLIPModel:
    def __new__(cls, name: str = 'ViT-B/32', *args, **kwargs):
        """Create a CLIP model instance."""
        if cls is CLIPModel:
            if name in _OPENAI_MODELS:
                from clip_server.model.openai_model import OpenAIClipModel

                instance = super().__new__(OpenAIClipModel)
            elif name in _OPENCLIP_MODELS:
                from clip_server.model.openclip_model import OpenClipModel

                instance = super().__new__(OpenClipModel)
            else:
                raise ValueError(f'The CLIP model name=`{name}` is not supported.')
        else:
            instance = super().__new__(cls)
        return instance

    def __init__(self, *args, **kwargs):
        self.x = 1
        ...

    def encode_text(self, input_ids, attention_mask, **kwargs):
        ...

    def encode_image(self, pixel_values, **kwargs):
        ...


# class CLIPPreprocessor:
#
#         def __new__(cls, *args, name: str = 'ViT-B/32', **kwargs):
#             """Create a CLIP feature extractor instance."""
#             if cls is CLIPFeatureExtractor:
#                 if name in []:
#                     ...
#                 else:
#                     raise ValueError(f'The CLIP feature extractor name=`{name}` is not supported.')
#             else:
#                 instance = super().__new__(cls)
#             return instance
#
#         def tokenize(self, texts: List[str], **kwargs):
#             ...
#
#         def transform(self, images, **kwargs):
#             ...
