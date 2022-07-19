import open_clip
import transformers

from .clip_preprocessor import CLIPPreprocessor


class MultilingualCLIPPreprocessor(CLIPPreprocessor):
    def __init__(self, model):
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(model._mclip_path)
        self._vision_preprocessor = open_clip.image_transform(
            model._model.visual.image_size, is_train=False
        )
