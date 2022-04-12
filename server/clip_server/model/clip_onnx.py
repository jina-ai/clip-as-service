import os

# It needs to be done before importing onnxruntime
os.environ['OMP_WAIT_POLICY'] = 'PASIVE'

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
    def __init__(
        self,
        name: str = None,
    ):
        if name in _MODELS:
            cache_dir = os.path.expanduser(f'~/.cache/clip/{name.replace("/", "-")}')
            self._textual_path = _download(_S3_BUCKET + _MODELS[name][0], cache_dir)
            self._visual_path = _download(_S3_BUCKET + _MODELS[name][1], cache_dir)
        else:
            raise RuntimeError(
                f'Model {name} not found; available models = {available_models()}'
            )

        self.sess_options = ort.SessionOptions()

        # Enables all available optimizations including layout optimizations
        self.sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        # Set the operators in the graph run in parallel
        self.sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL

        # Control the number of threads used to parallelize the execution of the graph (across nodes)
        self.sess_options.inter_op_num_threads = 1
        self.sess_options.intra_op_num_threads = int(
            os.environ.get('OMP_NUM_THREADS', '1')
        )

    def start_sessions(
        self,
        **kwargs,
    ):
        self._visual_session = ort.InferenceSession(
            self._visual_path, sess_options=self.sess_options, **kwargs
        )
        self._textual_session = ort.InferenceSession(
            self._textual_path, sess_options=self.sess_options, **kwargs
        )

    def encode_image(self, onnx_image):
        onnx_input_image = {self._visual_session.get_inputs()[0].name: onnx_image}
        (visual_output,) = self._visual_session.run(None, onnx_input_image)
        return visual_output

    def encode_text(self, onnx_text):
        onnx_input_text = {self._textual_session.get_inputs()[0].name: onnx_text}
        (textual_output,) = self._textual_session.run(None, onnx_input_text)
        return textual_output
