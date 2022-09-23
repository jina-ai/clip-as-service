import os
import warnings
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict
from functools import partial

import onnxruntime as ort
from clip_server.executors.helper import (
    split_img_txt_da,
    preproc_image,
    preproc_text,
    set_rank,
)
from clip_server.model import clip
from clip_server.model.clip_onnx import CLIPOnnxModel
from clip_server.model.tokenization import Tokenizer
from jina import Executor, requests, DocumentArray


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B-32::openai',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        access_paths: str = '@r',
        model_path: Optional[str] = None,
        **kwargs,
    ):
        """
        :param name: The name of the model to be used. Default 'ViT-B-32::openai'. A list of available models can be
            found at https://clip-as-service.jina.ai/user-guides/server/#model-support
        :param device: 'cpu' or 'cuda'. Default is None, which auto-detects the device.
        :param num_worker_preprocess: The number of CPU workers to preprocess images and texts. Default is 4.
        :param minibatch_size: The size of the minibatch for preprocessing and encoding. Default is 32. Reduce this
            number if you encounter OOM errors.
        :param access_paths: The access paths to traverse on the input documents to get the images and texts to be
            processed. Visit https://docarray.jina.ai/fundamentals/documentarray/access-elements for more details.
        :param model_path: The path to the model to be used. If not specified, the model will be downloaded or loaded
            from the local cache. Visit https://clip-as-service.jina.ai/user-guides/server/#use-custom-model-for-onnx
            to learn how to finetune custom models.
        """
        super().__init__(**kwargs)

        self._minibatch_size = minibatch_size
        self._access_paths = access_paths
        if 'traversal_paths' in kwargs:
            warnings.warn(
                f'`traversal_paths` is deprecated. Use `access_paths` instead.'
            )
            self._access_paths = kwargs['traversal_paths']

        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._model = CLIPOnnxModel(name, model_path)
        self._tokenizer = Tokenizer(name)

        self._image_transform = clip._transform_ndarray(self._model.image_size)

        import torch

        if not device:
            self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self._device = device

        # define the priority order for the execution providers
        providers = ['CPUExecutionProvider']

        # prefer CUDA Execution Provider over CPU Execution Provider
        if self._device.startswith('cuda'):
            providers.insert(0, 'CUDAExecutionProvider')

        sess_options = ort.SessionOptions()

        # Enables all available optimizations including layout optimizations
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        if not self._device.startswith('cuda') and (
            'OMP_NUM_THREADS' not in os.environ
            and hasattr(self.runtime_args, 'replicas')
        ):
            replicas = getattr(self.runtime_args, 'replicas', 1)
            num_threads = max(1, torch.get_num_threads() * 2 // replicas)
            if num_threads < 2:
                warnings.warn(
                    f'Too many replicas ({replicas}) vs too few threads {num_threads} may result in '
                    f'sub-optimal performance.'
                )

            # Run the operators in the graph in parallel (not support the CUDA Execution Provider)
            sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

            # The number of threads used to parallelize the execution of the graph (across nodes)
            sess_options.inter_op_num_threads = 1
            sess_options.intra_op_num_threads = max(num_threads, 1)

        self._model.start_sessions(sess_options=sess_options, providers=providers)

    def _preproc_images(self, docs: 'DocumentArray', drop_image_content: bool):
        with self.monitor(
            name='preprocess_images_seconds',
            documentation='images preprocess time in seconds',
        ):
            return preproc_image(
                docs,
                preprocess_fn=self._image_transform,
                return_np=True,
                drop_image_content=drop_image_content,
            )

    def _preproc_texts(self, docs: 'DocumentArray'):
        with self.monitor(
            name='preprocess_texts_seconds',
            documentation='texts preprocess time in seconds',
        ):
            return preproc_text(docs, tokenizer=self._tokenizer, return_np=True)

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        _drop_image_content = parameters.get('drop_image_content', False)
        await self.encode(docs['@r,m'], drop_image_content=_drop_image_content)

        set_rank(docs)

    @requests
    async def encode(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):
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

        # for image
        if _img_da:
            for minibatch, batch_data in _img_da.map_batch(
                partial(self._preproc_images, drop_image_content=_drop_image_content),
                batch_size=self._minibatch_size,
                pool=self._pool,
            ):
                with self.monitor(
                    name='encode_images_seconds',
                    documentation='images encode time in seconds',
                ):
                    minibatch.embeddings = self._model.encode_image(batch_data)

        # for text
        if _txt_da:
            for minibatch, batch_data in _txt_da.map_batch(
                self._preproc_texts,
                batch_size=self._minibatch_size,
                pool=self._pool,
            ):
                with self.monitor(
                    name='encode_texts_seconds',
                    documentation='texts encode time in seconds',
                ):
                    minibatch.embeddings = self._model.encode_text(batch_data)

        return docs
