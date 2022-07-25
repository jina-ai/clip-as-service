# Originally from https://github.com/mlfoundations/open_clip.
#
# Copyright (c) 2012-2021 Gabriel Ilharco, Mitchell Wortsman,
# Nicholas Carlini, Rohan Taori, Achal Dave, Vaishaal Shankar,
# John Miller, Hongseok Namkoong, Hannaneh Hajishirzi, Ali Farhadi,
# Ludwig Schmidt

from typing import TYPE_CHECKING
from copy import deepcopy
import warnings
import torch

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import get_model_url_md5, download_model

from open_clip.model import (
    CLIP,
    convert_weights_to_fp16,
    build_model_from_openai_state_dict,
)
from open_clip.factory import _MODEL_CONFIGS, load_state_dict

if TYPE_CHECKING:
    import torch


class OpenCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)

        if '::' in name:
            model_name, pretrained = name.split('::')
        else:  # older CaS version's name format
            model_name = name
            pretrained = 'openai'

        precision = 'fp32' if device == 'cpu' else 'fp16'  # fp16 for cuda to save VRAM
        model_url, md5sum = get_model_url_md5(name)
        if model_url:
            model_path = download_model(model_url, md5sum=md5sum)
            if pretrained.lower() == 'openai':
                try:
                    # loading JIT archive
                    model = torch.jit.load(
                        model_path, map_location=device if jit else "cpu"
                    ).eval()
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
                else:
                    # patch the device names
                    device_holder = torch.jit.trace(
                        lambda: torch.ones([]).to(torch.device(device)),
                        example_inputs=[],
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
                                if "value" in node.attributeNames() and str(
                                    node["value"]
                                ).startswith("cuda"):
                                    node.copyAttributes(device_node)

                    model.apply(patch_device)
                    patch_device(model.encode_image)
                    patch_device(model.encode_text)

                    # patch dtype to float32 on CPU
                    if device == "cpu":
                        float_holder = torch.jit.trace(
                            lambda: torch.ones([]).float(), example_inputs=[]
                        )
                        float_input = list(
                            float_holder.graph.findNode("aten::to").inputs()
                        )[1]
                        float_node = float_input.node()

                        def patch_float(module):
                            try:
                                graphs = (
                                    [module.graph] if hasattr(module, "graph") else []
                                )
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
                if precision == "fp32":
                    model = model.float()
            else:
                if model_name in _MODEL_CONFIGS:
                    model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
                else:
                    raise RuntimeError(f'Model config for {model_name} not found.')

                model = CLIP(**model_cfg)

                if pretrained:
                    if model_path:
                        model.load_state_dict(load_state_dict(model_path))
                    else:
                        raise RuntimeError(
                            f'Pretrained weights ({pretrained}) not found for model {model_name}.'
                        )

                model.to(device=torch.device(device))
                if precision == "fp16":
                    convert_weights_to_fp16(model)
                if jit:
                    model = torch.jit.script(model)

            self._model = model
            self._model_name = model_name
        else:
            raise RuntimeError(f'Model ({name}) is not on s3 server.')

    @property
    def model_name(self):
        if self._model_name == 'ViT-L/14@336px':
            return 'ViT-L-14-336'
        elif self._model_name.endswith('-quickgelu'):
            return self._model_name[:-10]
        return self._model_name.replace('/', '-')

    def encode_text(self, input_ids: 'torch.Tensor', **kwargs):
        return self._model.encode_text(input_ids)

    def encode_image(self, pixel_values: 'torch.Tensor', **kwargs):
        return self._model.encode_image(pixel_values)
