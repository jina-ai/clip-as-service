import os

import numpy as np
import torch.cuda

from clip_server.model.pretrained_models import (
    download_model,
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
)
from clip_server.model.clip_model import BaseCLIPModel
from clip_server.model.clip_onnx import _MODELS, _S3_BUCKET, _S3_BUCKET_V2


class CLIPNebullvmModel(BaseCLIPModel):
    def __init__(self, name: str, model_path: str = None):
        super().__init__(name)
        if name in _MODELS:
            if not model_path:
                cache_dir = os.path.expanduser(
                    f'~/.cache/clip/{name.replace("/", "-").replace("::", "-")}'
                )
                textual_model_name, textual_model_md5 = _MODELS[name][0]
                self._textual_path = download_model(
                    url=_S3_BUCKET_V2 + textual_model_name,
                    target_folder=cache_dir,
                    md5sum=textual_model_md5,
                    with_resume=True,
                )
                visual_model_name, visual_model_md5 = _MODELS[name][1]
                self._visual_path = download_model(
                    url=_S3_BUCKET_V2 + visual_model_name,
                    target_folder=cache_dir,
                    md5sum=visual_model_md5,
                    with_resume=True,
                )
            else:
                if os.path.isdir(model_path):
                    self._textual_path = os.path.join(model_path,
                                                      'textual.onnx')
                    self._visual_path = os.path.join(model_path, 'visual.onnx')
                    if not os.path.isfile(
                            self._textual_path) or not os.path.isfile(
                            self._visual_path
                    ):
                        raise RuntimeError(
                            f'The given model path {model_path} does not contain `textual.onnx` and `visual.onnx`'
                        )
                else:
                    raise RuntimeError(
                        f'The given model path {model_path} should be a folder containing both '
                        f'`textual.onnx` and `visual.onnx`.'
                    )
        else:
            raise RuntimeError(
                'CLIP model {} not found or not supports ONNX backend; below is a list of all available models:\n{}'.format(
                    name,
                    ''.join(
                        ['\t- {}\n'.format(i) for i in list(_MODELS.keys())]),
                )
            )

    def optimize_models(
        self,
        **kwargs,
    ):
        from nebullvm.api.functions import optimize_model

        general_kwargs = {}
        general_kwargs.update(kwargs)

        dynamic_info = {
            "inputs": [
                {0: 'batch', 1: 'num_channels', 2: 'pixel_size', 3: 'pixel_size'}
            ],
            "outputs": [{0: 'batch'}],
        }

        self._visual_model = optimize_model(
            self._visual_path,
            input_data=[
                (
                    (
                        np.random.randn(1, 3, self.pixel_size, self.pixel_size).astype(
                            np.float32
                        ),
                    ),
                    0,
                )
            ],
            dynamic_info=dynamic_info,
            **general_kwargs,
        )

        dynamic_info = {
            "inputs": [
                {0: 'batch', 1: 'num_tokens'},
            ],
            "outputs": [
                {0: 'batch'},
            ],
        }

        self._textual_model = optimize_model(
            self._textual_path,
            input_data=[((np.random.randint(0, 100, (1, 77)),), 0)],
            dynamic_info=dynamic_info,
            **general_kwargs,
        )

    @staticmethod
    def get_model_name(name: str):
        if name in _OPENCLIP_MODELS:
            from clip_server.model.openclip_model import OpenCLIPModel

            return OpenCLIPModel.get_model_name(name)
        elif name in _MULTILINGUALCLIP_MODELS:
            from clip_server.model.mclip_model import MultilingualCLIPModel

            return MultilingualCLIPModel.get_model_name(name)

        return name

    def encode_image(self, onnx_image):
        (visual_output,) = self._visual_model(onnx_image)
        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_model(onnx_text)
        return textual_output


class EnvRunner:
    def __init__(self, device: str, num_threads: int = None):
        self.device = device
        self.cuda_str = None
        self.rm_cuda_flag = False
        self.num_threads = num_threads

    def __enter__(self):
        if self.device == "cpu" and torch.cuda.is_available():
            self.cuda_str = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            self.rm_cuda_flag = self.cuda_str is None
        if self.num_threads is not None:
            os.environ["NEBULLVM_THREADS_PER_MODEL"] = f"{self.num_threads}"

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cuda_str is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_str
        elif self.rm_cuda_flag:
            os.environ.pop("CUDA_VISIBLE_DEVICES")
        if self.num_threads is not None:
            os.environ.pop("NEBULLVM_THREADS_PER_MODEL")
