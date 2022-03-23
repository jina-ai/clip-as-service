import io
import os
from typing import TYPE_CHECKING, Optional, List

import torch
from PIL import Image
from jina import Executor, requests

from clip_server.model import clip

if TYPE_CHECKING:
    from docarray import DocumentArray, Document


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        jit: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self._model, self._preprocess = clip.load(name, device=self._device, jit=jit)

    def _preproc_image(self, d: 'Document'):
        d.tensor = self._preprocess(Image.open(io.BytesIO(d.blob))).to(self._device)
        return d

    def _preproc_text(self, da: 'DocumentArray') -> List[str]:
        texts = da.texts
        da.tensors = clip.tokenize(da.texts).to(self._device)
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
                _img_da.embeddings = (
                    self._model.encode_image(_img_da.tensors).cpu().numpy()
                )

            # for text
            if _txt_da:
                texts = self._preproc_text(_txt_da)
                _txt_da.embeddings = (
                    self._model.encode_text(_txt_da.tensors).cpu().numpy()
                )
                _txt_da.texts = texts

        # drop tensors
        docs.tensors = None


if __name__ == '__main__':
    os.environ['JINA_LOG_LEVEL'] = 'DEBUG'
    CLIPEncoder.serve(port=12345)
