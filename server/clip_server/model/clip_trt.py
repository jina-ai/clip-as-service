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

from clip_server.model.clip import _download, MODEL_SIZE
from clip_server.model.clip_onnx import _MODELS as ONNX_MODELS

_MODELS = {
    'RN50': (
        ('RN50/textual.trt', 'e2b9bcbd32c43c3007e8d1b0c5cf88a0'),
        ('RN50/visual.trt', '62a91786f5e69750409ea31dee4d1d06'),
    ),
    'RN101': (
        ('RN101/textual.trt', 'b78d708c7cce2eb5066c46911ded0c3a'),
        ('RN101/visual.trt', '2be9a9c00a4f963a7379a26cd2d1a643'),
    ),
    'RN50x4': (
        ('RN50x4/textual.trt', 'd6872573f70e3362a6f706c82997a2ce'),
        ('RN50x4/visual.trt', 'b56106204de38d28fa01a60ffb5f3124'),
    ),
    # 'RN50x16'
    # 'RN50x64'
    'ViT-B/32': (
        ('ViT-B-32/textual.trt', 'fda48ae2bb0b3e8c402e102d3b5b8344'),
        ('ViT-B-32/visual.trt', '9c9ad16efe0e01c768fefe978135e3e8'),
    ),
    'ViT-B/16': (
        ('ViT-B-16/textual.trt', '74ee85b7cfbc3d9dda6fe927cc56d163'),
        ('ViT-B-16/visual.trt', 'be623fcbdaa512a77d6becea5188e72e'),
    ),
    'ViT-L/14': (
        ('ViT-L-14/textual.trt', 'a90ca6422f5e948f4f6e9fafd06e2d76'),
        ('ViT-L-14/visual.trt', 'e32cfcdb04d98693bb05ee2ba330cc93'),
    ),
    # 'ViT-L/14@336px'
}


class CLIPTensorRTModel:
    def __init__(
        self,
        name: str = None,
    ):
        if name in _MODELS:
            self._name = name
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')

            self._textual_path = os.path.join(
                cache_dir,
                _MODELS[name][0][0].replace('.', f'.{ONNX_MODELS[name][0][1]}.'),
            )
            self._visual_path = os.path.join(
                cache_dir,
                _MODELS[name][1][0].replace('.', f'.{ONNX_MODELS[name][1][1]}.'),
            )

            if not os.path.exists(self._textual_path) or not os.path.exists(
                self._visual_path
            ):
                from clip_server.model.clip_onnx import CLIPOnnxModel

                trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
                runtime: Runtime = trt.Runtime(trt_logger)
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
        else:
            raise RuntimeError(
                f'Model {name} not found or not supports Nvidia TensorRT backend; available models = {list(_MODELS.keys())}'
            )

    def start_engines(self):
        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        runtime: Runtime = trt.Runtime(trt_logger)
        self._textual_engine = load_engine(runtime, self._textual_path)
        self._visual_engine = load_engine(runtime, self._visual_path)

    def encode_image(self, onnx_image):
        (visual_output,) = self._visual_engine({'input': onnx_image})
        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_engine(onnx_text)
        return textual_output
