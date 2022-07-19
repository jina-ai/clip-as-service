import open_clip

from .clip_preprocessor import CLIPPreprocessor


class OpenCLIPPreprocessor(CLIPPreprocessor):
    def __init__(self, model):
        self._tokenizer = open_clip.tokenize
        self._vision_preprocessor = open_clip.image_transform(
            model._model.visual.image_size, is_train=False
        )
