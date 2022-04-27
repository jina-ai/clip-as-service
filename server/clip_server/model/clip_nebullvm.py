import os
from pathlib import Path

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
