import os
from typing import Dict

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
from clip_server.model.pretrained_models import (
    _OPENCLIP_MODELS,
    _MULTILINGUALCLIP_MODELS,
)
from clip_server.model.clip_model import BaseCLIPModel
from clip_server.model.clip_onnx import _MODELS as ONNX_MODELS

_MODELS = [
    'RN50::openai',
    'RN50::yfcc15m',
    'RN50::cc12m',
    'RN101::openai',
    'RN101::yfcc15m',
    'RN50x4::openai',
    'ViT-B-32::openai',
    'ViT-B-32::laion2b_e16',
    'ViT-B-32::laion400m_e31',
    'ViT-B-32::laion400m_e32',
    'ViT-B-16::openai',
    'ViT-B-16::laion400m_e31',
    'ViT-B-16::laion400m_e32',
    # older version name format
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


class CLIPTensorRTModel(BaseCLIPModel):
    def __init__(
        self,
        name: str,
    ):
        super().__init__(name)

        if name in _MODELS:
            cache_dir = os.path.expanduser(
                f'~/.cache/clip/{name.replace("/", "-").replace("::", "-")}'
            )

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
                onnx_model = CLIPOnnxModel(name)

                visual_engine = build_engine(
                    runtime=runtime,
                    onnx_file_path=onnx_model._visual_path,
                    logger=trt_logger,
                    min_shape=(1, 3, onnx_model.image_size, onnx_model.image_size),
                    optimal_shape=(
                        768,
                        3,
                        onnx_model.image_size,
                        onnx_model.image_size,
                    ),
                    max_shape=(
                        1024,
                        3,
                        onnx_model.image_size,
                        onnx_model.image_size,
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
                'CLIP model {} not found or not supports Nvidia TensorRT backend; below is a list of all available models:\n{}'.format(
                    name,
                    ''.join(['\t- {}\n'.format(i) for i in list(_MODELS.keys())]),
                )
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

    def start_engines(self):
        trt_logger: Logger = trt.Logger(trt.Logger.ERROR)
        runtime: Runtime = trt.Runtime(trt_logger)
        self._textual_engine = load_engine(runtime, self._textual_path)
        self._visual_engine = load_engine(runtime, self._visual_path)

    def encode_image(self, image_input: Dict):
        (visual_output,) = self._visual_engine(image_input)
        return visual_output

    def encode_text(self, text_input: Dict):
        (textual_output,) = self._textual_engine(text_input)
        return textual_output
