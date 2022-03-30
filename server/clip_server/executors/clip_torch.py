import io
from typing import TYPE_CHECKING, Optional, List, Tuple

import torch
from PIL import Image
from clip_server.model import clip
from jina import Executor, requests

if TYPE_CHECKING:
    from docarray import DocumentArray


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        jit: bool = False,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self._num_worker_preprocess = num_worker_preprocess
        self._minibatch_size = minibatch_size
        self._model, self._preprocess = clip.load(name, device=self._device, jit=jit)

    def _preproc_image(self, da: 'DocumentArray') -> 'DocumentArray':
        for d in da:
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
