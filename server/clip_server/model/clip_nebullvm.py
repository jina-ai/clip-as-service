import os
from pathlib import Path

import torch.cuda

from .clip import _download, available_models

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/'
_MODELS = {
    'RN50': ('RN50/textual.onnx', 'RN50/visual.onnx'),
    'RN101': ('RN101/textual.onnx', 'RN101/visual.onnx'),
    'RN50x4': ('RN50x4/textual.onnx', 'RN50x4/visual.onnx'),
    'RN50x16': ('RN50x16/textual.onnx', 'RN50x16/visual.onnx'),
    'RN50x64': ('RN50x64/textual.onnx', 'RN50x64/visual.onnx'),
    'ViT-B/32': ('ViT-B-32/textual.onnx', 'ViT-B-32/visual.onnx'),
    'ViT-B/16': ('ViT-B-16/textual.onnx', 'ViT-B-16/visual.onnx'),
    'ViT-L/14': ('ViT-L-14/textual.onnx', 'ViT-L-14/visual.onnx'),
    'ViT-L/14@336px': ('ViT-L-14@336px/textual.onnx', 'ViT-L-14@336px/visual.onnx'),
}


class CLIPNebullvmModel:
    def __init__(self, name: str = None, pixel_size: int = 224):
        self.pixel_size = pixel_size
        if name in _MODELS:
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')
            self._textual_path = _download(_S3_BUCKET + _MODELS[name][0], cache_dir)
            self._visual_path = _download(_S3_BUCKET + _MODELS[name][1], cache_dir)
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

    def optimize_models(
        self,
        **kwargs,
    ):
        from nebullvm.api.frontend.onnx import optimize_onnx_model

        save_dir = os.path.expanduser("~/.cache/clip/nebullvm")
        Path(save_dir).mkdir(exist_ok=True)
        visual_save_dir = os.path.join(save_dir, "visual")
        Path(visual_save_dir).mkdir(exist_ok=True)
        text_save_dir = os.path.join(save_dir, "text")
        Path(text_save_dir).mkdir(exist_ok=True)
        general_kwargs = {
            "batch_size": 1,
        }
        general_kwargs.update(kwargs)
        self._visual_model = optimize_onnx_model(
            self._visual_path,
            input_sizes=[(3, self.pixel_size, self.pixel_size)],
            save_dir=visual_save_dir,
            ignore_compilers=["tvm"],
            **general_kwargs,
        )

        self._textual_model = optimize_onnx_model(
            self._textual_path,
            input_sizes=[(77,)],
            save_dir=text_save_dir,
            input_types=["int"],
            ignore_compilers=["tvm"],
            **general_kwargs,
        )

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
