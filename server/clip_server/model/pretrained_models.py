import os
import hashlib
import shutil
import urllib
import warnings
import logging
from typing import Union, List
from copy import deepcopy
from open_clip.model import (
    CLIP,
    convert_weights_to_fp16,
    build_model_from_openai_state_dict,
)
from open_clip.openai import load_openai_model
from open_clip.factory import _MODEL_CONFIGS, load_state_dict
from open_clip.pretrained import get_pretrained_url
import torch


_OPENCLIP_S3_BUCKET = 'https://clip-as-service.s3.us-east-2.amazonaws.com/models/torch'
_OPENCLIP_MODELS = {
    'RN50::openai': ('RN50.pt', '9140964eaaf9f68c95aa8df6ca13777c'),
    'RN50::yfcc15m': (),
    'RN50::cc12m': (),
    'RN50-quickgelu::openai': (),
    'RN50-quickgelu::yfcc15m': (),
    'RN50-quickgelu::cc12m': (),
    'RN101::openai': ('RN101.pt', 'fa9d5f64ebf152bc56a18db245071014'),
    'RN101::yfcc15m': (),
    'RN101-quickgelu::openai': (),
    'RN101-quickgelu::yfcc15m': (),
    'RN50x4::openai': ('RN50x4.pt', '03830990bc768e82f7fb684cde7e5654'),
    'RN50x16::openai': ('RN50x16.pt', '83d63878a818c65d0fb417e5fab1e8fe'),
    'RN50x64::openai': ('RN50x64.pt', 'a6631a0de003c4075d286140fc6dd637'),
    'ViT-B-32::openai': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'ViT-B-32::laion2b_e16': (),
    'ViT-B-32::laion400m_e31': (),
    'ViT-B-32::laion400m_e32': (),
    'ViT-B-32-quickgelu::openai': (),
    'ViT-B-32-quickgelu::laion400m_e31': (),
    'ViT-B-32-quickgelu::laion400m_e32': (),
    'ViT-B-16::openai': ('ViT-B-16.pt', '44c3d804ecac03d9545ac1a3adbca3a6'),
    'ViT-B-16::laion400m_e31': (),
    'ViT-B-16::laion400m_e32': (),
    'ViT-B-16-plus-240::laion400m_e31': (),
    'ViT-B-16-plus-240::laion400m_e32': (),
    'ViT-L-14::openai': ('ViT-L-14.pt', '096db1af569b284eb76b3881534822d9'),
    'ViT-L-14-336::openai': ('ViT-L-14-336px.pt', 'b311058cae50cb10fbfa2a44231c9473'),
    'RN50': (
        'RN50.pt',
        '9140964eaaf9f68c95aa8df6ca13777c',
    ),  # older version name format
    'RN101': ('RN101.pt', 'fa9d5f64ebf152bc56a18db245071014'),
    'RN50x4': ('RN50x4.pt', '03830990bc768e82f7fb684cde7e5654'),
    'RN50x16': ('RN50x16.pt', '83d63878a818c65d0fb417e5fab1e8fe'),
    'RN50x64': ('RN50x64.pt', 'a6631a0de003c4075d286140fc6dd637'),
    'ViT-B/32': ('ViT-B-32.pt', '3ba34e387b24dfe590eeb1ae6a8a122b'),
    'ViT-B/16': ('ViT-B-16.pt', '44c3d804ecac03d9545ac1a3adbca3a6'),
    'ViT-L/14': ('ViT-L-14.pt', '096db1af569b284eb76b3881534822d9'),
    'ViT-L/14@336px': ('ViT-L-14-336px.pt', 'b311058cae50cb10fbfa2a44231c9473'),
}

_MULTILINGUALCLIP_MODELS = {
    'M-CLIP/XLM-Roberta-Large-Vit-B-32': (),
    'M-CLIP/XLM-Roberta-Large-Vit-L-14': (),
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus': (),
    'M-CLIP/LABSE-Vit-L-14': (),
}


def md5file(filename: str):
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    return hash_md5.hexdigest()


def create_model(
    model_name: str,
    pretrained: str = '',
    precision: str = 'fp32',
    device: torch.device = torch.device('cpu'),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model_name = model_name.replace(
        '/', '-'
    )  # for callers using old naming with / in ViT names

    if pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if model_name in _MODEL_CONFIGS:
            logging.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert (
                    False
                ), 'pretrained image towers currently only supported for timm models'

        model = CLIP(**model_cfg)

        if pretrained:
            checkpoint_path = ''
            url, md5 = get_model_url_md5(model_name, pretrained)
            if url:
                checkpoint_path = download_model(url, md5sum=md5)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                model.load_state_dict(load_state_dict(checkpoint_path))
            else:
                logging.warning(
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                )
                raise RuntimeError(
                    f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                )

        model.to(device=device)
        if precision == "fp16":
            assert device.type != 'cpu'
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model


def load_openai_model(
    name: str,
    device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu",
    jit=True,
):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict
    device : Union[str, torch.device]
        The device to put the loaded model
    jit : bool
        Whether to load the optimized JIT model (default) or more hackable non-JIT model.

    Returns
    -------
    model : torch.nn.Module
        The CLIP model
    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if get_model_url_md5(name, 'openai'):
        url, md5 = get_model_url_md5(name, 'openai')
        model_path = download_model(url, md5sum=md5)
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found")

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(
                f"File {model_path} is not a JIT archive. Loading as a state dict instead"
            )
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        try:
            model = build_model_from_openai_state_dict(
                state_dict or model.state_dict()
            ).to(device)
        except KeyError:
            sd = {k[7:]: v for k, v in state_dict["state_dict"].items()}
            model = build_model_from_openai_state_dict(sd).to(device)

        if str(device) == "cpu":
            model.float()
        return model

    # patch the device names
    device_holder = torch.jit.trace(
        lambda: torch.ones([]).to(torch.device(device)), example_inputs=[]
    )
    device_node = [
        n
        for n in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(n)
    ][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith(
                    "cuda"
                ):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(
            lambda: torch.ones([]).float(), example_inputs=[]
        )
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [
                        1,
                        2,
                    ]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)
        model.float()

    # ensure image_size attr available at consistent location for both jit and non-jit
    model.visual.image_size = model.input_resolution.item()
    return model


def get_model_url_md5(model: str, pretrained: str):
    full_name = model + '::' + pretrained
    if full_name not in _OPENCLIP_MODELS:
        return ''
    model_pretrained = _OPENCLIP_MODELS[full_name]
    if len(model_pretrained) == 0:  # not on s3
        return get_pretrained_url(model, pretrained), None
    else:
        return (_OPENCLIP_S3_BUCKET + '/' + model_pretrained[0], model_pretrained[1])


def download_model(
    url: str,
    target_folder: str = os.path.expanduser("~/.cache/clip"),
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
        " \n",  # divide this bar from Flow's bar
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
