import os
import torch

try:
    import tensorrt as trt
    from tensorrt.tensorrt import Logger, Runtime

    from .trt_utils import build_engine, infer_tensorrt
except ImportError:
    raise ImportError(
        "It seems that TensorRT is not yet installed. "
        "It is required when you declare TensorRT backend."
        "Please find installation instruction on "
        "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
    )

from .clip import _download, available_models
from .clip_onnx import _S3_BUCKET, _MODELS


class CLIPTensorRTModel:
    def __init__(
        self,
        name: str = None,
        max_batch_size: int = 1024,
        max_seq_len: int = 77,
        image_resolution: int = 224,
        image_channel: int = 3,
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

        self._textual_engine = build_engine(
            runtime=runtime,
            onnx_file_path=self._textual_path,
            logger=trt_logger,
            min_shape=(1, max_seq_len),
            optimal_shape=(max_batch_size, max_seq_len),
            max_shape=(max_batch_size, max_seq_len),
            workspace_size=10000 * 1024 * 1024,
            fp16=True,
            int8=False,
        )

        self._visual_engine = build_engine(
            runtime=runtime,
            onnx_file_path=self._visual_path,
            logger=trt_logger,
            min_shape=(1, image_channel, image_resolution, image_resolution),
            optimal_shape=(
                max_batch_size,
                image_channel,
                image_resolution,
                image_resolution,
            ),
            max_shape=(
                max_batch_size,
                image_channel,
                image_resolution,
                image_resolution,
            ),
            workspace_size=10000 * 1024 * 1024,
            fp16=True,
            int8=False,
        )

    def start_contexts(
        self,
        **kwargs,
    ):
        self._textual_context = self._textual_engine.create_execution_context()
        self._textual_context.set_optimization_profile_async(
            profile_index=0, stream_handle=torch.cuda.current_stream().cuda_stream
        )

        self._visual_context = self._visual_engine.create_execution_context()
        self._visual_context.set_optimization_profile_async(
            profile_index=0, stream_handle=torch.cuda.current_stream().cuda_stream
        )

    def encode_image(self, onnx_image):
        (visual_output,) = infer_tensorrt(
            context=self._visual_context,
            host_inputs={'input': torch.tensor(onnx_image)},
            input_binding_idxs=[0],
            output_binding_idxs=[1],
        )

        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = infer_tensorrt(
            context=self._textual_context,
            host_inputs={'input': torch.tensor(onnx_text)},
            input_binding_idxs=[0],
            output_binding_idxs=[1],
        )
        return textual_output
