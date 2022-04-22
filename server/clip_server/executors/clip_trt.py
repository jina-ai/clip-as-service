from multiprocessing.pool import ThreadPool
from typing import List, Tuple
import numpy as np

from jina import Executor, requests, DocumentArray
from jina.logging.logger import JinaLogger

from clip_server.model import clip
from clip_server.model.clip_trt import CLIPTensorRTModel

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
        device: str = 'cuda',
        num_worker_preprocess: int = 4,
        minibatch_size: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self._preprocess_blob = clip._transform_blob(_SIZE[name])
        self._preprocess_tensor = clip._transform_ndarray(_SIZE[name])
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._minibatch_size = minibatch_size
        self._device = device

        import torch

        assert (
            torch.cuda.is_available()
        ), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"

        assert self._device.startswith('cuda'), (
            f'can not perform inference on {self._device}'
            f' with Nvidia TensorRT as backend'
        )

        self._model = CLIPTensorRTModel(
            name, max_batch_size=minibatch_size, image_resolution=_SIZE[name]
        )
        self._model.start_contexts()

    def _preproc_image(self, da: 'DocumentArray') -> 'DocumentArray':
        for d in da:
            if d.tensor is not None:
                d.tensor = self._preprocess_tensor(d.tensor)
            else:
                if not d.blob and d.uri:
                    # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                    d.load_uri_to_blob()
                d.tensor = self._preprocess_blob(d.blob)
        da.tensors = da.tensors.detach().cpu().numpy().astype(np.float32)
        return da

    def _preproc_text(self, da: 'DocumentArray') -> Tuple['DocumentArray', List[str]]:
        texts = da.texts
        da.tensors = clip.tokenize(texts).detach().cpu().numpy().astype(np.int64)
        da[:, 'mime_type'] = 'text'
        return da, texts

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs:
            if d.text:
                _txt_da.append(d)
            elif (d.blob is not None) or (d.tensor is not None):
                _img_da.append(d)
            elif d.uri:
                _img_da.append(d)
            else:
                self.logger.warning(
                    f'The content of document {d.id} is empty, cannot be processed'
                )

        # for image
        if _img_da:
            for minibatch in _img_da.map_batch(
                self._preproc_image, batch_size=self._minibatch_size, pool=self._pool
            ):
                minibatch.embeddings = self._model.encode_image(minibatch.tensors)

        # for text
        if _txt_da:
            for minibatch, _texts in _txt_da.map_batch(
                self._preproc_text, batch_size=self._minibatch_size, pool=self._pool
            ):
                minibatch.embeddings = self._model.encode_text(minibatch.tensors)
                minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
