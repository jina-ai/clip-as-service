import os
import onnx
import onnxruntime as ort
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
}


class CLIPOnnxModel:
    def __init__(self, name: str = None):
        if name in _MODELS:
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')
            self._textual_path = _download(_S3_BUCKET + _MODELS[name][0], cache_dir)
            self._visual_path = _download(_S3_BUCKET + _MODELS[name][1], cache_dir)
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

    def start_sessions(
        self,
        **kwargs,
    ):
        self._visual_session = ort.InferenceSession(self._visual_path, **kwargs)
        self._visual_session.disable_fallback()

        self._textual_session = ort.InferenceSession(self._textual_path, **kwargs)
        self._textual_session.disable_fallback()

    def encode_image(self, onnx_image):
        onnx_input_image = {self._visual_session.get_inputs()[0].name: onnx_image}
        (visual_output,) = self._visual_session.run(None, onnx_input_image)
        return visual_output

    def encode_text(self, onnx_text):
        onnx_input_text = {self._textual_session.get_inputs()[0].name: onnx_text}
        (textual_output,) = self._textual_session.run(None, onnx_input_text)
        return textual_output


def convert_float_to_float16(model_path: str, output_model_path: str):
    from onnxmltools.utils.float16_converter import (
        convert_float_to_float16_model_path,
    )

    new_onnx_model = convert_float_to_float16_model_path(model_path)

    onnx.save(new_onnx_model, output_model_path)

    # Alternate approach
    # from onnx import load_model
    # from onnxruntime.transformers import optimizer, onnx_model
    #
    # # optimized_model = optimizer.optimize_model(model_path, model_type='bert')
    #
    # model = load_model(model_path)
    # optimized_model = onnx_model.OnnxModel(model)
    #
    # if hasattr(optimized_model, 'convert_float32_to_float16'):
    #     optimized_model.convert_float_to_float16()
    # else:
    #     optimized_model.convert_model_float32_to_float16()
    #
    # self._textual_path = f'{self._textual_path[:-5]}_optimized.onnx'
    # optimized_model.save_model_to_file(output_model_path)


def quantize(model_path: str, output_model_path: str):
    """
    Quantize the weights of the model from float32 to in8 to allow very efficient inference on modern CPU
    Uses unsigned ints for activation values, signed ints for weights, per
    https://onnxruntime.ai/docs/performance/quantization.html#data-type-selection
    it is faster on most CPU architectures
    Args:
        onnx_model_path: Path to location the exported ONNX model is stored
    Returns: The Path generated for the quantized
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantize_dynamic(
        model_input=model_path,
        model_output=output_model_path,
        per_channel=True,
        reduce_range=True,  # should be the same as per_channel
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
        optimize_model=True,
        op_types_to_quantize=["MatMul", "Attention", "Mul", "Add"],
        extra_options={"WeightSymmetric": False, "MatMulConstBOnly": True},
    )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
