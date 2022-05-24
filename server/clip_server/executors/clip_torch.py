import os
import warnings
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict

import numpy as np
import torch
from clip_server.executors.helper import (
    split_img_txt_da,
    preproc_image,
    preproc_text,
    set_rank,
)
from clip_server.model import clip
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

        await self.encode(docs['@r,m'])

        set_rank(docs)

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs:
            split_img_txt_da(d, _img_da, _txt_da)

        with torch.inference_mode():
            # for image
            if _img_da:
                for minibatch, _contents in _img_da.map_batch(
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

                    # recover original content
                    try:
                        _ = iter(_contents)
                        for _d, _ct in zip(minibatch, _contents):
                            _d.content = _ct
                    except TypeError:
                        pass

            # for text
            if _txt_da:
                for minibatch, _contents in _txt_da.map_batch(
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

                    # recover original content
                    try:
                        _ = iter(_contents)
                        for _d, _ct in zip(minibatch, _contents):
                            _d.content = _ct
                    except TypeError:
                        pass

        # drop tensors
        docs.tensors = None
        return docs
