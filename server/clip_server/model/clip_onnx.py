import os

from clip_server.model.clip import _download, available_models

_S3_BUCKET = (
    'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/'  # Deprecated
)
_S3_BUCKET_V2 = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models-436c69702d61732d53657276696365/onnx/'
_MODELS = {
    'RN50': (
        ('RN50/textual.onnx', '722418bfe47a1f5c79d1f44884bb3103'),
        ('RN50/visual.onnx', '5761475db01c3abb68a5a805662dcd10'),
    ),
    'RN101': (
        ('RN101/textual.onnx', '2d9efb7d184c0d68a369024cedfa97af'),
        ('RN101/visual.onnx', '0297ebc773af312faab54f8b5a622d71'),
    ),
    'RN50x4': (
        ('RN50x4/textual.onnx', 'd9d63d3fe35fb14d4affaa2c4e284005'),
        ('RN50x4/visual.onnx', '16afe1e35b85ad862e8bbdb12265c9cb'),
    ),
    'RN50x16': (
        ('RN50x16/textual.onnx', '1525785494ff5307cadc6bfa56db6274'),
        ('RN50x16/visual.onnx', '2a293d9c3582f8abe29c9999e47d1091'),
    ),
    'RN50x64': (
        ('RN50x64/textual.onnx', '3ae8ade74578eb7a77506c11bfbfaf2c'),
        ('RN50x64/visual.onnx', '1341f10b50b3aca6d2d5d13982cabcfc'),
    ),
    'ViT-B/32': (
        ('ViT-B-32/textual.onnx', 'bd6d7871e8bb95f3cc83aff3398d7390'),
        ('ViT-B-32/visual.onnx', '88c6f38e522269d6c04a85df18e6370c'),
    ),
    'ViT-B/16': (
        ('ViT-B-16/textual.onnx', '6f0976629a446f95c0c8767658f12ebe'),
        ('ViT-B-16/visual.onnx', 'd5c03bfeef1abbd9bede54a8f6e1eaad'),
    ),
    'ViT-L/14': (
        ('ViT-L-14/textual.onnx', '325380b31af4837c2e0d9aba2fad8e1b'),
        ('ViT-L-14/visual.onnx', '53f5b319d3dc5d42572adea884e31056'),
    ),
    'ViT-L/14@336px': (
        ('ViT-L-14@336px/textual.onnx', '78fab479f136403eed0db46f3e9e7ed2'),
        ('ViT-L-14@336px/visual.onnx', 'f3b1f5d55ca08d43d749e11f7e4ba27e'),
    ),
}


class CLIPOnnxModel:
    def __init__(self, name: str = None, model_path: str = None):
        if name in _MODELS:
            if not model_path:
                cache_dir = os.path.expanduser(
                    f'~/.cache/clip/{name.replace("/", "-")}'
                )
                textual_model_name, textual_model_md5 = _MODELS[name][0]
                self._textual_path = _download(
                    url=_S3_BUCKET_V2 + textual_model_name,
                    target_folder=cache_dir,
                    md5sum=textual_model_md5,
                    with_resume=True,
                )
                visual_model_name, visual_model_md5 = _MODELS[name][1]
                self._visual_path = _download(
                    url=_S3_BUCKET_V2 + visual_model_name,
                    target_folder=cache_dir,
                    md5sum=visual_model_md5,
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
                            f'The given model path {model_path} does not contain `textual.onnx` and `visual.onnx`'
                        )
                else:
                    raise RuntimeError(
                        f'The given model path {model_path} is not a valid directory'
                    )
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
        (visual_output,) = self._visual_session.run(None, onnx_image)
        return visual_output

    def encode_text(self, onnx_text):
        (textual_output,) = self._textual_session.run(None, onnx_text)
        return textual_output
