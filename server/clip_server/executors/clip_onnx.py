import os
import warnings
from functools import partial
from multiprocessing.pool import ThreadPool
from typing import Optional, Dict
import numpy as np
import onnxruntime as ort

from jina import Executor, requests, DocumentArray

from clip_server.model import clip
from clip_server.model.clip_onnx import CLIPOnnxModel
from clip_server.executors.helper import split_img_txt_da, preproc_image, preproc_text


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Optional[str] = None,
        num_worker_preprocess: int = 4,
        minibatch_size: int = 16,
        logit_scale: float = 4.60,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._preprocess_tensor = clip._transform_ndarray(clip.MODEL_SIZE[name])
        self._pool = ThreadPool(processes=num_worker_preprocess)

        self._minibatch_size = minibatch_size

        self._model = CLIPOnnxModel(name)
        self._logit_scale = logit_scale

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
            # TODO: support tensorrt
            # providers.insert(0, 'TensorrtExecutionProvider')

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
            num_threads = max(1, torch.get_num_threads() // replicas)
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

    @requests(on='/rank')
    async def rank(self, docs: 'DocumentArray', parameters: Dict, **kwargs):
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

                # normalized features
                image_features = _img_da.embeddings / np.linalg.norm(
                    _img_da.embeddings, axis=1, keepdims=True
                )
                text_features = _txt_da.embeddings / np.linalg.norm(
                    _txt_da.embeddings, axis=1, keepdims=True
                )

                # cosine similarity as logits
                logit_scale = np.exp(self._logit_scale)
                logits_per_image = logit_scale * np.matmul(
                    image_features, text_features.T
                )
                logits_per_text = logits_per_image.T

                def numpy_softmax(z):
                    s = np.max(z, axis=1)
                    s = s[:, np.newaxis]
                    e_x = np.exp(z - s)
                    div = np.sum(e_x, axis=1)
                    div = div[:, np.newaxis]  # dito
                    return e_x / div

                if len(_img_da) == 1:
                    probs = numpy_softmax(logits_per_image)[0]
                elif len(_txt_da) == 1:
                    probs = numpy_softmax(logits_per_text)[0]

                # drop embeddings
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

        # for image
        if _img_da:
            for minibatch in _img_da.map_batch(
                partial(
                    preproc_image, preprocess_fn=self._preprocess_tensor, return_np=True
                ),
                batch_size=self._minibatch_size,
                pool=self._pool,
            ):
                minibatch.embeddings = self._model.encode_image(minibatch.tensors)

        # for text
        if _txt_da:
            for minibatch, _texts in _txt_da.map_batch(
                partial(preproc_text, return_np=True),
                batch_size=self._minibatch_size,
                pool=self._pool,
            ):
                minibatch.embeddings = self._model.encode_text(minibatch.tensors)
                minibatch.texts = _texts

        # drop tensors
        docs.tensors = None

        return docs
