import os
import warnings
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict

import onnxruntime as ort
from clip_server.executors.helper import (
    split_img_txt_da,
    preproc_image,
    preproc_text,
    set_rank,
)
from clip_server.model import clip
from clip_server.model.clip_onnx import CLIPOnnxModel
from jina import Executor, requests, DocumentArray


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 32,
        traversal_paths: str = '@r',
        model_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._minibatch_size = minibatch_size
        self._traversal_paths = traversal_paths

        self._preprocess_tensor = clip._transform_ndarray(clip.MODEL_SIZE[name])
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._model = CLIPOnnxModel(name, model_path)

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

    def _preproc_images(self, docs: 'DocumentArray'):
        with self.monitor(
            name='preprocess_images_seconds',
            documentation='images preprocess time in seconds',
        ):
            return preproc_image(
                docs, preprocess_fn=self._preprocess_tensor, return_np=True
            )

    def _preproc_texts(self, docs: 'DocumentArray'):
        with self.monitor(
            name='preprocess_texts_seconds',
            documentation='texts preprocess time in seconds',
        ):
            return preproc_text(docs, return_np=True)

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
        await self.encode(docs['@r,m'])

        set_rank(docs)

    @requests
    async def encode(self, docs: 'DocumentArray', parameters: Dict = {}, **kwargs):

        traversal_paths = parameters.get('traversal_paths', self._traversal_paths)
        minibatch_size = parameters.get('minibatch_size', self._minibatch_size)

        _img_da = DocumentArray()
        _txt_da = DocumentArray()
        for d in docs[traversal_paths]:
            split_img_txt_da(d, _img_da, _txt_da)

        # for image
        if _img_da:
            for minibatch, batch_data in _img_da.map_batch(
                self._preproc_images,
                batch_size=minibatch_size,
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
                batch_size=minibatch_size,
                pool=self._pool,
            ):
                with self.monitor(
                    name='encode_texts_seconds',
                    documentation='texts encode time in seconds',
                ):
                    minibatch.embeddings = self._model.encode_text(batch_data)

        return docs
