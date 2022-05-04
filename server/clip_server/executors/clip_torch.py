import os
import warnings
from functools import partial

from multiprocessing.pool import ThreadPool
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
from clip_server.model import clip
from clip_server.executors.helper import split_img_txt_da, preproc_image, preproc_text

from jina import Executor, requests, DocumentArray


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        jit: bool = False,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        if not self._device.startswith('cuda') and (
            'OMP_NUM_THREADS' not in os.environ
            and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads {num_threads} may result in '
                    f'sub-optimal performance.'
                )

            # NOTE: make sure to set the threads right after the torch import,
            # and `torch.set_num_threads` always take precedence over environment variables `OMP_NUM_THREADS`.
            # For more details, please see https://pytorch.org/docs/stable/generated/torch.set_num_threads.html
            # FIXME: This hack would harm the performance in K8S deployment.
            torch.set_num_threads(max(num_threads, 1))
            torch.set_num_interop_threads(1)

        self._minibatch_size = minibatch_size
        self._model, self._preprocess_tensor = clip.load(
            name, device=self._device, jit=jit
        )

        self._pool = ThreadPool(processes=num_worker_preprocess)

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        import torch

        _source = parameters.get('source', 'matches')
        _get = lambda d: getattr(d, _source)

        for d in docs:
            _img_da = DocumentArray()
            _txt_da = DocumentArray()
            split_img_txt_da(d, _img_da, _txt_da)

            for c in _get(d):
                split_img_txt_da(c, _img_da, _txt_da)

            if len(_img_da) != 1 and len(_txt_da) != 1:
                raise ValueError(
                    f'`d.{_source}` must be all in same modality, either all images or all text'
                )
            elif not _img_da or not _txt_da:
                raise ValueError(
                    f'`d` and `d.{_source}` must be in different modality, one is image one is text'
                )
            elif len(_get(d)) <= 1:
                raise ValueError(
                    f'`d.{_source}` must have more than one Documents to do ranking'
                )
            else:
                _img_da = await self.encode(_img_da)
                _txt_da = await self.encode(_txt_da)
                _img_da.embeddings = torch.from_numpy(_img_da.embeddings)
                _txt_da.embeddings = torch.from_numpy(_txt_da.embeddings)

                # normalized features
                image_features = _img_da.embeddings / _img_da.embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features = _txt_da.embeddings / _txt_da.embeddings.norm(
                    dim=-1, keepdim=True
                )

                # cosine similarity as logits
                logit_scale = self._model.logit_scale.exp()
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                if len(_img_da) == 1:
                    probs = (
                        logits_per_image.softmax(dim=-1)
                        .cpu()
                        .detach()
                        .numpy()
                        .squeeze()
                    )
                elif len(_txt_da) == 1:
                    probs = (
                        logits_per_text.softmax(dim=-1).cpu().detach().numpy().squeeze()
                    )

                _img_da.embeddings = None
                _txt_da.embeddings = None

                for c, v in zip(_get(d), probs):
                    c.scores['clip_score'].value = v
                setattr(
                    d,
                    _source,
                    sorted(
                        _get(d),
                        key=lambda _m: _m.scores['clip_score'].value,
                        reverse=True,
                    ),
                )

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs:
            split_img_txt_da(d, _img_da, _txt_da)

        with torch.inference_mode():
            # for image
            if _img_da:
                for minibatch in _img_da.map_batch(
                    partial(
                        preproc_image,
                        preprocess_fn=self._preprocess_tensor,
                        device=self._device,
                        return_np=False,
                    ),
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_image(minibatch.tensors)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )

            # for text
            if _txt_da:
                for minibatch, _texts in _txt_da.map_batch(
                    partial(preproc_text, device=self._device, return_np=False),
                    batch_size=self._minibatch_size,
                    pool=self._pool,
                ):
                    minibatch.embeddings = (
                        self._model.encode_text(minibatch.tensors)
                        .cpu()
                        .numpy()
                        .astype(np.float32)
                    )
                    minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
        return docs
