import io
import os
from time import perf_counter
from typing import TYPE_CHECKING, Optional, List

import torch
from PIL import Image
from jina import Executor, requests
from torch.utils.data import Dataset, DataLoader

from clip_server.model import clip

if TYPE_CHECKING:
    from docarray import DocumentArray, Document


class DocumentDataset(Dataset):
    def __init__(self, device):
        self.device = device
        self.da = None

    def set_da(self, da: 'DocumentArray'):
        self.da = da

    def __len__(self):
        return len(self.da)

    def __getitem__(self, idx):
        text = self.da[idx].text
        tensor = clip.tokenize(text)
        return tensor[0]


class CLIPEncoder(Executor):
    def __init__(
            self,
            name: str = 'ViT-B/32',
            device: Optional[str] = None,
            jit: bool = False,
            batch_size: int = 1000,
            **kwargs
    ):
        super().__init__(**kwargs)
        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device
        self._model, self._preprocess = clip.load(name, device=self._device, jit=jit)
        self.batch_size = batch_size

        self.dataset = DocumentDataset(self._device)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers=48, persistent_workers=True)

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
                self.dataset.set_da(_txt_da)

                st = perf_counter()
                for i, batch in enumerate(self.dataloader):
                    print('gpu waiting for workers', perf_counter() - st); st = perf_counter()
                    _txt_da[i * self.batch_size:(i + 1) * self.batch_size].embeddings = (
                        self._model.encode_text(batch.to(self._device)).cpu().numpy()
                    )
                    print('gpu encoding', perf_counter() - st); st = perf_counter()

                    st = perf_counter()

        # drop tensors
        docs.tensors = None
