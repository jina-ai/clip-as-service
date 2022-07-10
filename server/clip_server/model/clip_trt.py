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

from clip_server.model.clip import MODEL_SIZE
from clip_server.model.clip_onnx import _MODELS as ONNX_MODELS

_MODELS = [
    'RN50',
    'RN101',
    'RN50x4',
    # 'RN50x16',
    # 'RN50x64',
    'ViT-B/32',
    'ViT-B/16',
    # 'ViT-L/14',
    # 'ViT-L/14@336px',
]


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
                f'textual.{ONNX_MODELS[name][0][1]}.trt',
            )
            self._visual_path = os.path.join(
                cache_dir,
                f'visual.{ONNX_MODELS[name][1][1]}.trt',
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
        (visual_output,) = self._visual_engine(onnx_image)
        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_engine(onnx_text)
        return textual_output
