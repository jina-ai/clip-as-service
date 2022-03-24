import io
import os
from typing import TYPE_CHECKING, List, Sequence

import torch
from PIL import Image
from jina import Executor, requests

from clip_server.model import clip
from clip_server.model.clip_onnx import CLIPOnnxModel

if TYPE_CHECKING:
    from docarray import DocumentArray, Document

_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
}


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        providers: Sequence = (
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ),
        **kwargs
    ):
        super().__init__(**kwargs)
        self._preprocess = clip._transform(_SIZE[name])
        self._model = CLIPOnnxModel(name)
        self._model.start_sessions(providers=providers)

    def _preproc_image(self, d: 'Document'):
        d.tensor = self._preprocess(Image.open(io.BytesIO(d.blob))).cpu().numpy()
        return d

    def _preproc_text(self, da: 'DocumentArray') -> List[str]:
        texts = da.texts
        da.tensors = clip.tokenize(da.texts).cpu().numpy()
        da[:, 'mime_type'] = 'text'
        return texts

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = docs.find({'blob': {'$exists': True}})
        _txt_da = docs.find({'text': {'$exists': True}})

        with torch.inference_mode():
            # for image
            if _img_da:
                _img_da.apply(self._preproc_image)
                _img_da.embeddings = self._model.encode_image(_img_da.tensors)

            # for text
            if _txt_da:
                texts = self._preproc_text(_txt_da)
                _txt_da.embeddings = self._model.encode_text(_txt_da.tensors)
                _txt_da.texts = texts

        # drop tensors
        docs.tensors = None
