import io
from multiprocessing.pool import ThreadPool, Pool
from typing import List, Sequence, Tuple

from PIL import Image
from jina import Executor, requests, DocumentArray

from clip_server.model import clip
from clip_server.model.clip_onnx import CLIPOnnxModel

_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
}


class CLIPEncoder(Executor):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        providers: Sequence = (
            'TensorrtExecutionProvider',
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ),
        num_worker_preprocess: int = 4,
        minibatch_size: int = 64,
        pool_backend: str = 'thread',
        **kwargs
    ):
        super().__init__(**kwargs)
        self._preprocess = clip._transform(_SIZE[name])
        self._model = CLIPOnnxModel(name)
        if pool_backend == 'thread':
            self._pool = ThreadPool(processes=num_worker_preprocess)
        else:
            self._pool = Pool(processes=num_worker_preprocess)
        self._minibatch_size = minibatch_size
        self._model.start_sessions(providers=providers)

    def _preproc_image(self, da: 'DocumentArray') -> 'DocumentArray':
        for d in da:
            if not d.blob and d.uri:
                # in case user uses HTTP protocol and send data via curl not using .blob (base64), but in .uri
                d.load_uri_to_blob()
            d.tensor = self._preprocess(Image.open(io.BytesIO(d.blob)))
        da.tensors = da.tensors.cpu().numpy()
        return da

    def _preproc_text(self, da: 'DocumentArray') -> Tuple['DocumentArray', List[str]]:
        texts = da.texts
        da.tensors = clip.tokenize(texts).cpu().numpy()
        da[:, 'mime_type'] = 'text'
        return da, texts

    @requests
    async def encode(self, docs: 'DocumentArray', **kwargs):
        _img_da = docs.find({'blob': {'$exists': True}})
        _txt_da = docs.find({'text': {'$exists': True}})

        # for image
        if _img_da:
            for minibatch in _img_da.map_batch(
                self._preproc_image, batch_size=self._minibatch_size, pool=self._pool
            ):
                minibatch.embeddings = self._model.encode_image(minibatch.tensors)

        # for text
        if _txt_da:
            for minibatch, _texts in _txt_da.map_batch(
                self._preproc_text, batch_size=self._minibatch_size, pool=self._pool
            ):
                minibatch.embeddings = self._model.encode_text(minibatch.tensors)
                minibatch.texts = _texts

        # drop tensors
        docs.tensors = None
