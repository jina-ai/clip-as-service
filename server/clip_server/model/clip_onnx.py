import os

from clip_server.model.clip import _download, available_models

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/'
_S3_BUCKET_V2 = 'https://clip-as-service.s3.us-east-2.amazonaws.com/modelsV2/onnx/'
_MODELS = {
    'RN50': (
        {'file': 'RN50/textual.onnx', 'md5': ''},
        {'file': 'RN50/visual.onnx', 'md5': ''},
    ),
    'RN101': (
        {'file': 'RN101/textual.onnx', 'md5': ''},
        {'file': 'RN101/visual.onnx', 'md5': ''},
    ),
    'RN50x4': (
        {'file': 'RN50x4/textual.onnx', 'md5': ''},
        {'file': 'RN50x4/visual.onnx', 'md5': ''},
    ),
    'RN50x16': (
        {'file': 'RN50x16/textual.onnx', 'md5': ''},
        {'file': 'RN50x16/visual.onnx', 'md5': ''},
    ),
    'RN50x64': (
        {'file': 'RN50x64/textual.onnx', 'md5': ''},
        {'file': 'RN50x64/visual.onnx', 'md5': ''},
    ),
    'ViT-B/32': (
        {'file': 'ViT-B-32/textual.onnx', 'md5': ''},
        {'file': 'ViT-B-32/visual.onnx', 'md5': ''},
    ),
    'ViT-B/16': (
        {'file': 'ViT-B-16/textual.onnx', 'md5': ''},
        {'file': 'ViT-B-16/visual.onnx', 'md5': ''},
    ),
    'ViT-L/14': (
        {'file': 'ViT-L-14/textual.onnx', 'md5': ''},
        {'file': 'ViT-L-14/visual.onnx', 'md5': ''},
    ),
    'ViT-L/14@336px': (
        {'file': 'ViT-L-14@336px/textual.onnx', 'md5': ''},
        {'file': 'ViT-L-14@336px/visual.onnx', 'md5': ''},
    ),
}


class CLIPOnnxModel:
    def __init__(self, name: str = None, model_path: str = None):
        if name in _MODELS:
            if not model_path:
                cache_dir = os.path.expanduser(
                    f'~/.cache/clip/v2/{name.replace("/", "-")}'
                )
                self._textual_path = _download(
                    _S3_BUCKET_V2 + _MODELS[name][0]['file'],
                    _MODELS[name][0]['md5'],
                    cache_dir,
                    with_resume=True,
                )
                self._visual_path = _download(
                    _S3_BUCKET_V2 + _MODELS[name][1]['file'],
                    _MODELS[name][1]['md5'],
                    cache_dir,
                    with_resume=True,
                )
            else:
                if os.path.isdir(model_path):
                    self._textual_path = os.path.join(model_path, 'textual.onnx')
                    self._visual_path = os.path.join(model_path, 'visual.onnx')
                    if not os.path.isfile(self._textual_path) or not os.path.isfile(
                        self._visual_path
                    ):
                        raise RuntimeError(
                            f'{model_path} does not contain `textual.onnx` and `visual.onnx`'
                        )
                else:
                    raise RuntimeError(f'{model_path} is not a directory')
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

    def start_sessions(
        self,
        **kwargs,
    ):
        import onnxruntime as ort

        self._visual_session = ort.InferenceSession(self._visual_path, **kwargs)
        self._visual_session.disable_fallback()

        self._textual_session = ort.InferenceSession(self._textual_path, **kwargs)
        self._textual_session.disable_fallback()

    def encode_image(self, onnx_image):
        onnx_input_image = {self._visual_session.get_inputs()[0].name: onnx_image}
        (visual_output,) = self._visual_session.run(None, onnx_input_image)
        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_session.run(None, onnx_text)
        return textual_output
