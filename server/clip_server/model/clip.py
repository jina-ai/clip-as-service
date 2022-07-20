# Originally from https://github.com/openai/CLIP. MIT License, Copyright (c) 2021 OpenAI

import io
import os
import hashlib
import shutil
import urllib
import warnings
from typing import Union, List

import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from clip_server.model.model import build_model
from clip_server.model.simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

__all__ = ['available_models', 'load', 'tokenize']
_tokenizer = _Tokenizer()

_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/torch/'
_MODELS = {
    'RN50': ('RN50.pt', '9140964eaaf9f68c95aa8df6ca13777c'),
    'RN101': ('RN101.pt', 'fa9d5f64ebf152bc56a18db245071014'),
    'RN50x4': ('RN50x4.pt', '03830990bc768e82f7fb684cde7e5654'),
    'RN50x16': ('RN50x16.pt', '83d63878a818c65d0fb417e5fab1e8fe'),
    'RN50x64': ('RN50x64.pt', 'a6631a0de003c4075d286140fc6dd637'),
    'ViT-B/32': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'ViT-B/16': ('ViT-B-16.pt', '44c3d804ecac03d9545ac1a3adbca3a6'),
    'ViT-L/14': ('ViT-L-14.pt', '096db1af569b284eb76b3881534822d9'),
    'ViT-L/14@336px': ('ViT-L-14-336px.pt', 'b311058cae50cb10fbfa2a44231c9473'),
}

MODEL_SIZE = {
    'RN50': 224,
    'RN101': 224,
    'RN50x4': 288,
    'RN50x16': 384,
    'RN50x64': 448,
    'ViT-B/32': 224,
    'ViT-B/16': 224,
    'ViT-L/14': 224,
    'ViT-L/14@336px': 336,
}


def md5file(filename: str):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def _download(
    url: str,
    target_folder: str,
    md5sum: str = None,
    with_resume: bool = True,
    max_attempts: int = 3,
) -> str:
    os.makedirs(target_folder, exist_ok=True)
    filename = os.path.basename(url)

    download_target = os.path.join(target_folder, filename)

    if os.path.exists(download_target):
        if not os.path.isfile(download_target):
            raise FileExistsError(f'{download_target} exists and is not a regular file')

        actual_md5sum = md5file(download_target)
        if (not md5sum) or actual_md5sum == md5sum:
            return download_target

    from rich.progress import (
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    progress = Progress(
        TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )

    with progress:
        task = progress.add_task('download', filename=url, start=False)

        for _ in range(max_attempts):
            tmp_file_path = download_target + '.part'
            resume_byte_pos = (
                os.path.getsize(tmp_file_path) if os.path.exists(tmp_file_path) else 0
            )

            try:
                # resolve the 403 error by passing a valid user-agent
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                total_bytes = int(
                    urllib.request.urlopen(req).info().get('Content-Length', -1)
                )
                mode = 'ab' if (with_resume and resume_byte_pos) else 'wb'

                with open(tmp_file_path, mode) as output:
                    progress.update(task, total=total_bytes)
                    progress.start_task(task)

                    if resume_byte_pos and with_resume:
                        progress.update(task, advance=resume_byte_pos)
                        req.headers['Range'] = f'bytes={resume_byte_pos}-'

                    with urllib.request.urlopen(req) as source:
                        while True:
                            buffer = source.read(8192)
                            if not buffer:
                                break

                            output.write(buffer)
                            progress.update(task, advance=len(buffer))

                actual_md5 = md5file(tmp_file_path)
                if (md5sum and actual_md5 == md5sum) or (not md5sum):
                    shutil.move(tmp_file_path, download_target)
                    return download_target
                else:
                    os.remove(tmp_file_path)
                    raise RuntimeError(
                        f'MD5 mismatch: expected {md5sum}, got {actual_md5}'
                    )

            except Exception as ex:
                progress.console.print(
                    f'Failed to download {url} with {ex!r} at the {_}th attempt'
                )
                progress.reset(task)

        raise RuntimeError(
            f'Failed to download {url} within retry limit {max_attempts}'
        )


def _convert_image_to_rgb(image):
    return image.convert('RGB')


def _blob2image(blob):
    return Image.open(io.BytesIO(blob))


def _transform_blob(n_px):
    return Compose(
        [
            _blob2image,
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            _convert_image_to_rgb,
            ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def _transform_ndarray(n_px):
    return Compose(
        [
            ToTensor(),
            Resize(n_px, interpolation=BICUBIC),
            CenterCrop(n_px),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )


def available_models() -> List[str]:
    '''Returns the names of available CLIP models'''
    return list(_MODELS.keys())


def load(
    name: str,
    device: Union[str, torch.device] = 'cuda' if torch.cuda.is_available() else 'cpu',
    jit: bool = False,
    download_root: str = None,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses '~/.cache/clip/'

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        model_name, model_md5 = _MODELS[name]
        model_path = _download(
            url=_S3_BUCKET + model_name,
            target_folder=download_root or os.path.expanduser('~/.cache/clip'),
            md5sum=model_md5,
            with_resume=True,
        )
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(
            f'Model {name} not found; available models = {available_models()}'
        )

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else 'cpu').eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f'File {model_path} is not a JIT archive. Loading as a state dict instead'
            )
            jit = False
        state_dict = torch.load(model_path, map_location='cpu')

    if not jit:
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == 'cpu':
            model.float()
        return (
            model,
            _transform_ndarray(model.visual.input_resolution),
        )

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes('prim::Constant')
        if 'Device' in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, 'graph') else []
        except RuntimeError:
            graphs = []

        if hasattr(module, 'forward1'):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes('prim::Constant'):
                if 'value' in node.attributeNames() and str(node['value']).startswith(
                    'cuda'
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == 'cpu':
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode('aten::to').inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, 'graph') else []
            except RuntimeError:
                graphs = []

            if hasattr(module, 'forward1'):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes('aten::to'):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()['value'] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return (
        model,
        _transform_ndarray(model.input_resolution.item()),
    )


def tokenize(
    texts: Union[str, List[str]], context_length: int = 77, truncate: bool = True
) -> dict:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A dict of tokenized representations of the input strings and their corresponding attention masks with both
        shape = [batch size, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder['<|startoftext|>']
    eot_token = _tokenizer.encoder['<|endoftext|>']
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]

    input_ids = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    attention_mask = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f'Input {texts[i]} is too long for context length {context_length}'
                )
        input_ids[i, : len(tokens)] = torch.tensor(tokens)
        attention_mask[i, : len(tokens)] = 1

    return {'input_ids': input_ids, 'attention_mask': attention_mask}
