import warnings
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Dict, Optional

import numpy as np
from clip_server.executors.helper import (
    preproc_image,
    preproc_text,
    set_rank,
    split_img_txt_da,
)
from clip_server.model import clip
from clip_server.model.clip_trt import CLIPTensorRTModel
from clip_server.model.tokenization import Tokenizer
from jina import DocumentArray, Executor, requests
from opentelemetry.trace import NoOpTracer, Span


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: str = 'cuda',
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        **kwargs,
    ):
        """
        :param name: The name of the model to be used. Default 'ViT-B-32::openai'. A list of available models can be
            found at https://clip-as-service.jina.ai/user-guides/server/#model-support
        :param device: 'cpu' or 'cuda'. Default is 'cuda' since TensorRT is only supported on CUDA.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param access_paths: The access paths to traverse on the input documents to get the images and texts to be
            processed. Visit https://docarray.jina.ai/fundamentals/documentarray/access-elements for more details.
        """
        super().__init__(**kwargs)

        self._num_worker_preprocess = num_worker_preprocess
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._minibatch_size = minibatch_size
        self._access_paths = access_paths
        if 'traversal_paths' in kwargs:
            warnings.warn(
                f'`traversal_paths` is deprecated. Use `access_paths` instead.'
            )
            self._access_paths = kwargs['traversal_paths']

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

        self._tokenizer = Tokenizer(name)
        self._image_transform = clip._transform_ndarray(self._model.image_size)

        if not self.tracer:
            self.tracer = NoOpTracer()

    def _preproc_images(self, docs: 'DocumentArray', drop_image_content: bool):
        with self.monitor(
            name='preprocess_images_seconds',
            documentation='images preprocess time in seconds',
        ):
            with self.tracer.start_as_current_span('preprocess_images'):
                return preproc_image(
                    docs,
                    preprocess_fn=self._image_transform,
                    device=self._device,
                    return_np=False,
                    drop_image_content=drop_image_content,
                )

    def _preproc_texts(self, docs: 'DocumentArray'):
        with self.monitor(
            name='preprocess_texts_seconds',
            documentation='texts preprocess time in seconds',
        ):
            with self.tracer.start_as_current_span('preprocess_images'):
                return preproc_text(
                    docs,
                    tokenizer=self._tokenizer,
                    device=self._device,
                    return_np=False,
                )

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        _drop_image_content = parameters.get('drop_image_content', False)
        await self.encode(docs['@r,m'], drop_image_content=_drop_image_content)

        set_rank(docs)

    @requests
    async def encode(
        self,
        docs: 'DocumentArray',
        tracing_context=None,
        parameters: Dict = {},
        **kwargs,
    ):
        with self.tracer.start_as_current_span(
            'encode', context=tracing_context
        ) as span:
            span.set_attribute('device', self._device)
            span.set_attribute('runtime', 'tensorrt')
            access_paths = parameters.get('access_paths', self._access_paths)
            if 'traversal_paths' in parameters:
                warnings.warn(
                    f'`traversal_paths` is deprecated. Use `access_paths` instead.'
                )
                access_paths = parameters['traversal_paths']
            _drop_image_content = parameters.get('drop_image_content', False)

            _img_da = DocumentArray()
            _txt_da = DocumentArray()
            for d in docs[access_paths]:
                split_img_txt_da(d, _img_da, _txt_da)

            with self.tracer.start_as_current_span('inference') as inference_span:
                inference_span.set_attribute('drop_image_content', _drop_image_content)
                inference_span.set_attribute('minibatch_size', self._minibatch_size)
                inference_span.set_attribute('has_img_da', True if _img_da else False)
                inference_span.set_attribute('has_txt_da', True if _txt_da else False)
                # for image
                if _img_da:
                    with self.tracer.start_as_current_span(
                        'img_minibatch_encoding'
                    ) as img_encode_span:
                        for minibatch, batch_data in _img_da.map_batch(
                            partial(
                                self._preproc_images,
                                drop_image_content=_drop_image_content,
                            ),
                            batch_size=self._minibatch_size,
                            pool=self._pool,
                        ):
                            with self.monitor(
                                name='encode_images_seconds',
                                documentation='images encode time in seconds',
                            ):
                                minibatch.embeddings = (
                                    self._model.encode_image(batch_data)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )

                # for text
                if _txt_da:
                    with self.tracer.start_as_current_span(
                        'txt_minibatch_encoding'
                    ) as txt_encode_span:
                        for minibatch, batch_data in _txt_da.map_batch(
                            self._preproc_texts,
                            batch_size=self._minibatch_size,
                            pool=self._pool,
                        ):
                            with self.monitor(
                                name='encode_texts_seconds',
                                documentation='texts encode time in seconds',
                            ):
                                minibatch.embeddings = (
                                    self._model.encode_text(batch_data)
                                    .detach()
                                    .cpu()
                                    .numpy()
                                    .astype(np.float32)
                                )

        return docs
