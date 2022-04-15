import os
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
    def __init__(self, name: str = None, use_float16: bool = False):
        if name in _MODELS:
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')
            self._textual_path = _download(_S3_BUCKET + _MODELS[name][0], cache_dir)
            self._visual_path = _download(_S3_BUCKET + _MODELS[name][1], cache_dir)
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )
        if use_float16:
            import onnx
            from onnxmltools.utils.float16_converter import (
                convert_float_to_float16_model_path,
            )

            new_onnx_model = convert_float_to_float16_model_path(self._textual_path)

            self._textual_path = f'{self._textual_path[:-5]}_optimized.onnx'
            onnx.save(new_onnx_model, self._textual_path)

            # from onnx import load_model
            # from onnxruntime.transformers import optimizer, onnx_model
            #
            # # optimized_model = optimizer.optimize_model(self._textual_path, model_type='bert')
            #
            # model = load_model(self._textual_path)
            # optimized_model = onnx_model.OnnxModel(model)
            #
            # if hasattr(optimized_model, 'convert_float32_to_float16'):
            #     optimized_model.convert_float_to_float16()
            # else:
            #     optimized_model.convert_model_float32_to_float16()
            #
            # self._textual_path = f'{self._textual_path[:-5]}_optimized.onnx'
            # optimized_model.save_model_to_file(self._textual_path)

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
