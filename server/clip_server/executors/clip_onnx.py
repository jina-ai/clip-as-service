import io
import os
from typing import TYPE_CHECKING, List, Sequence, Tuple

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
        num_worker_preprocess: int = 4,
        minibatch_size: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._preprocess = clip._transform(_SIZE[name])
        self._model = CLIPOnnxModel(name)
        self._num_worker_preprocess = num_worker_preprocess
        self._minibatch_size = minibatch_size
        self._model.start_sessions(providers=providers)

    def _preproc_image(self, da: 'DocumentArray') -> 'DocumentArray':
        for d in da:
            d.tensor = self._preprocess(Image.open(io.BytesIO(d.blob)))
        da.tensors = da.tensors.cpu().numpy()
        return da

    def _preproc_text(self, da: 'DocumentArray') -> Tuple['DocumentArray', List[str]]:
        texts = da.texts
        da.tensors = clip.tokenize(texts).cpu().numpy()
        da[:, 'mime_type'] = 'text'
        return da, texts

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = docs.find({'blob': {'$exists': True}})
        _txt_da = docs.find({'text': {'$exists': True}})

        # for image
        if _img_da:
            for minibatch in _img_da.map_batch(
                self._preproc_image,
                batch_size=self._minibatch_size,
                num_worker=self._num_worker_preprocess,
            ):
                minibatch.embeddings = (
                    self._model.encode_image(minibatch.tensors).cpu().numpy()
                )

        # for text
        if _txt_da:
            for minibatch, _texts in _txt_da.map_batch(
                self._preproc_text,
                batch_size=self._minibatch_size,
                num_worker=self._num_worker_preprocess,
            ):
                minibatch.embeddings = (
                    self._model.encode_text(minibatch.tensors).cpu().numpy()
                )
                minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
