import numpy
import open_clip
import torch
from PIL import Image
from typing import List, Tuple, Dict

from clip_server.model.clip_preprocessor import CLIPPreprocessor


class OpenCLIPPreprocessor(CLIPPreprocessor):
    def __init__(self, model):
        self._tokenizer = open_clip.tokenize
        self._vision_preprocessor = open_clip.image_transform(
            model._model.visual.image_size, is_train=False
        )
        self._device = model._device

    def tokenize(self, texts: List[str], **kwargs):
        return self._tokenizer(texts, **kwargs)

    def transform(self, images: numpy.ndarray, **kwargs):
        return self._vision_preprocessor(Image.fromarray(images), **kwargs)

    def preproc_image(
        self, da: 'DocumentArray', return_np: bool = False, **kwargs
    ) -> Tuple['DocumentArray', Dict]:
        tensors_batch = []
        for d in da:
            content = d.content
            if d.blob:
                d.convert_blob_to_image_tensor()
            elif d.tensor is None and d.uri:
                # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                d.load_uri_to_image_tensor()
            tensors_batch.append(self.transform(d.tensor, **kwargs).detach())
            # recover doc content
            d.content = content
        tensors_batch = torch.stack(tensors_batch).type(torch.float32)
        if return_np:
            tensors_batch = tensors_batch.cpu().numpy()
        else:
            tensors_batch = tensors_batch.to(self._device)
        return da, {'pixel_values': tensors_batch}

    def preproc_text(
        self, da: 'DocumentArray', return_np: bool = False, **kwargs
    ) -> Tuple['DocumentArray', Dict]:
        inputs = self.tokenize(da.texts, **kwargs)
        inputs = {'input_ids': inputs, 'attention_mask': inputs}
        inputs['input_ids'] = inputs['input_ids'].detach()
        if return_np:
            inputs['input_ids'] = inputs['input_ids'].cpu().numpy().astype(np.int32)
            inputs['attention_mask'] = (
                inputs['attention_mask'].cpu().numpy().astype(np.int32)
            )
        else:
            inputs['input_ids'] = inputs['input_ids'].to(self._device)
            inputs['attention_mask'] = inputs['attention_mask'].to(self._device)
        da[:, 'mime_type'] = 'text'
        return da, inputs
