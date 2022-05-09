import os

try:
    import tensorrt as trt
    from tensorrt.tensorrt import Logger, Runtime

    from clip_server.model.trt_utils import load_engine, build_engine, save_engine
except ImportError:
    raise ImportError(
        "It seems that TensorRT is not yet installed. "
        "It is required when you declare TensorRT backend."
        "Please find installation instruction on "
        "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    )

from .clip import _download, MODEL_SIZE

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/tensorrt/'
_MODELS = {
    'RN50': ('RN50/textual.trt', 'RN50/visual.trt'),
    'RN101': ('RN101/textual.trt', 'RN101/visual.trt'),
    'RN50x4': ('RN50x4/textual.trt', 'RN50x4/visual.trt'),
    # 'RN50x16': ('RN50x16/textual.trt', 'RN50x16/visual.trt'),
    # 'RN50x64': ('RN50x64/textual.trt', 'RN50x64/visual.trt'),
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
                f'Model {name} not found or not supports Nvidia TensorRT backend; available models = {list(_MODELS.keys())}'
            )
        self._name = name

    def start_engines(self):
        import torch

        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        runtime: Runtime = trt.Runtime(trt_logger)
        compute_capacity = torch.cuda.get_device_capability()

        if compute_capacity != (8, 6):
            print(
                f'The engine plan file is generated on an incompatible device, expecting compute {compute_capacity} '
                'got compute 8.6, will rebuild the TensorRT engine.'
            )
            from clip_server.model.clip_onnx import CLIPOnnxModel

            onnx_model = CLIPOnnxModel(self._name)

            visual_engine = build_engine(
                runtime=runtime,
                onnx_file_path=onnx_model._visual_path,
                logger=trt_logger,
                min_shape=(1, 3, MODEL_SIZE[self._name], MODEL_SIZE[self._name]),
                optimal_shape=(
                    768,
                    3,
                    MODEL_SIZE[self._name],
                    MODEL_SIZE[self._name],
                ),
                max_shape=(
                    1024,
                    3,
                    MODEL_SIZE[self._name],
                    MODEL_SIZE[self._name],
                ),
                workspace_size=10000 * 1024 * 1024,
                fp16=False,
                int8=False,
            )

            save_engine(visual_engine, self._visual_path)

            text_engine = build_engine(
                runtime=runtime,
                onnx_file_path=onnx_model._textual_path,
                logger=trt_logger,
                min_shape=(1, 77),
                optimal_shape=(768, 77),
                max_shape=(1024, 77),
                workspace_size=10000 * 1024 * 1024,
                fp16=False,
                int8=False,
            )
            save_engine(text_engine, self._textual_path)

        self._textual_engine = load_engine(runtime, self._textual_path)
        self._visual_engine = load_engine(runtime, self._visual_path)

    def encode_image(self, onnx_image):
        (visual_output,) = self._visual_engine({'input': onnx_image})

        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_engine({'input': onnx_text})

        return textual_output
