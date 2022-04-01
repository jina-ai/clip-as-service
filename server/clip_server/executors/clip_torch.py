import io
from multiprocessing.pool import ThreadPool, Pool
from typing import Optional, List, Tuple

import torch
from PIL import Image
from jina import Executor, requests, DocumentArray

from clip_server.model import clip


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        jit: bool = False,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 64,
        pool_backend: str = 'thread',
        **kwargs
    ):
        super().__init__(**kwargs)
        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self._minibatch_size = minibatch_size
        self._model, self._preprocess = clip.load(name, device=self._device, jit=jit)
        if pool_backend == 'thread':
            self._pool = ThreadPool(processes=num_worker_preprocess)
        else:
            self._pool = Pool(processes=num_worker_preprocess)

    def _preproc_image(self, da: 'DocumentArray') -> 'DocumentArray':
        for d in da:
            if not d.blob and d.uri:
                # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                d.load_uri_to_blob()
            d.tensor = self._preprocess(Image.open(io.BytesIO(d.blob)))
        da.tensors = da.tensors.to(self._device)
        return da

    def _preproc_text(self, da: 'DocumentArray') -> Tuple['DocumentArray', List[str]]:
        texts = da.texts
        da.tensors = clip.tokenize(texts).to(self._device)
        da[:, 'mime_type'] = 'text'
        return da, texts

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = docs.find({'blob': {'$exists': True}})
        _txt_da = docs.find({'text': {'$exists': True}})

        with torch.inference_mode():
            # for image
            if _img_da:
                for minibatch in _img_da.map_batch(
                    self._preproc_image,
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_image(minibatch.tensors).cpu().numpy()
                    )

            # for text
            if _txt_da:
                for minibatch, _texts in _txt_da.map_batch(
                    self._preproc_text,
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_text(minibatch.tensors).cpu().numpy()
                    )
                    minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
