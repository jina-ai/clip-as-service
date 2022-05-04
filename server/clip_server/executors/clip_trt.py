from multiprocessing.pool import ThreadPool
from functools import partial
import numpy as np
from jina import Executor, requests, DocumentArray
from jina.logging.logger import JinaLogger

from clip_server.model import clip
from clip_server.model.clip_trt import CLIPTensorRTModel
from clip_server.executors.helper import split_img_txt_da, preproc_image, preproc_text


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

        self._preprocess_tensor = clip._transform_ndarray(clip.MODEL_SIZE[name])
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._minibatch_size = minibatch_size
        self._device = device

        import torch

        assert self._device.startswith('cuda'), (
            f'can not perform inference on {self._device}'
            f' with Nvidia TensorRT as backend'
        )

        assert (
            torch.cuda.is_available()
        ), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"

        self._model = CLIPTensorRTModel(name)

        self._model.start_engines()

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs:
            split_img_txt_da(d, _img_da, _txt_da)

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
                    .detach()
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
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
