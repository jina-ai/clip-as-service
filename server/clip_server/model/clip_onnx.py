import os

from clip_server.model.clip import _download, available_models

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/onnx/'
_S3_BUCKET_V2 = 'https://clip-as-service.s3.us-east-2.amazonaws.com/modelsV2/onnx/'
_MODELS = {
    'RN50': (
        {'file': 'RN50/textual.onnx', 'md5': '4a9ec971acaec34803bf998fb76eeb0d'},
        {'file': 'RN50/visual.onnx', 'md5': 'cfb219a1425a9695e3932b352b871cea'},
    ),
    'RN101': (
        {'file': 'RN101/textual.onnx', 'md5': '27bb683bd3d7ca981de41392267b0c67'},
        {'file': 'RN101/visual.onnx', 'md5': '2c4eb325dfe666c77bcf367ca6ea8a1f'},
    ),
    'RN50x4': (
        {'file': 'RN50x4/textual.onnx', 'md5': '94867427a2fc56277360b9264d7175cf'},
        {'file': 'RN50x4/visual.onnx', 'md5': '82dcd804e26e49212c7c9dd8ef03ec99'},
    ),
    'RN50x16': (
        {'file': 'RN50x16/textual.onnx', 'md5': 'e30f3b7663abea8da0ea6308b6537379'},
        {'file': 'RN50x16/visual.onnx', 'md5': 'e2230d03206d440d78c2124e9509e82c'},
    ),
    'RN50x64': (
        {'file': 'RN50x64/textual.onnx', 'md5': 'feac53596fc81aba14172228190ab01b'},
        {'file': 'RN50x64/visual.onnx', 'md5': '18e44a88d4d5ee060b60e0b0c002c84d'},
    ),
    'ViT-B/32': (
        {'file': 'ViT-B-32/textual.onnx', 'md5': 'c0a36f7e31f36beab096d5b83ad7f5dc'},
        {'file': 'ViT-B-32/visual.onnx', 'md5': '2357f80ccdce3264c37ab0b8587d9d82'},
    ),
    'ViT-B/16': (
        {'file': 'ViT-B-16/textual.onnx', 'md5': '699bc2b3a36f7fc645d694008967e11d'},
        {'file': 'ViT-B-16/visual.onnx', 'md5': '3316d2e54acc20999357ec8552e72daa'},
    ),
    'ViT-L/14': (
        {'file': 'ViT-L-14/textual.onnx', 'md5': '535a086fcaebd6018b778ff10f9dd938'},
        {'file': 'ViT-L-14/visual.onnx', 'md5': 'bf851771e29fcc512a2276f4b5fe3660'},
    ),
    'ViT-L/14@336px': (
        {
            'file': 'ViT-L-14@336px/textual.onnx',
            'md5': 'd87232b942bb399c902e5feec0761909',
        },
        {
            'file': 'ViT-L-14@336px/visual.onnx',
            'md5': 'fdcb444d3a696b0261838decd3635c29',
        },
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
