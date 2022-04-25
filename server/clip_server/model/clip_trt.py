import os
import torch

try:
    import tensorrt as trt
    from tensorrt.tensorrt import Logger, Runtime

    from .trt_utils import load_engine
except ImportError:
    raise ImportError(
        "It seems that TensorRT is not yet installed. "
        "It is required when you declare TensorRT backend."
        "Please find installation instruction on "
        "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    )

from .clip import _download, available_models

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/tensorrt/'
_MODELS = {
    'RN50': ('RN50/textual.trt', 'RN50/visual.trt'),
    'RN101': ('RN101/textual.trt', 'RN101/visual.trt'),
    'RN50x4': ('RN50x4/textual.trt', 'RN50x4/visual.trt'),
    'RN50x16': ('RN50x16/textual.trt', 'RN50x16/visual.trt'),
    'RN50x64': ('RN50x64/textual.trt', 'RN50x64/visual.trt'),
    'ViT-B/32': ('ViT-B-32/textual.trt', 'ViT-B-32/visual.trt'),
    'ViT-B/16': ('ViT-B-16/textual.trt', 'ViT-B-16/visual.trt'),
    'ViT-L/14': ('ViT-L-14/textual.trt', 'ViT-L-14/visual.trt'),
}


class CLIPTensorRTModel:
    def __init__(
        self,
        name: str = None,
    ):
        if name in _MODELS:
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')
            self._textual_path = _download(_S3_BUCKET + _MODELS[name][0], cache_dir)
            self._visual_path = _download(_S3_BUCKET + _MODELS[name][1], cache_dir)
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        runtime: Runtime = trt.Runtime(trt_logger)

        self._textual_engine = load_engine(runtime, self._textual_path)
        self._visual_engine = load_engine(runtime, self._visual_path)

    def encode_image(self, onnx_image):
        (visual_output,) = self._visual_engine({'input': onnx_image})

        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_engine({'input': onnx_text})

        return textual_output
