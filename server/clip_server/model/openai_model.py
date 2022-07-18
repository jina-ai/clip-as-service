import os

import clip
import torch
from torch import nn
import warnings
from typing import Union, List

from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import (
    _OPENAI_MODELS,
    _OPENAI_S3_BUCKET,
    download_model,
)
from clip_server.model.model import build_model


def available_models() -> List[str]:
    '''Returns the names of available CLIP models'''
    return list(_OPENAI_MODELS.keys())


class OpenAIClipModel(CLIPModel, nn.Module):
    def __init__(
        self,
        name: str = 'ViT-B/32',
        device: Union[str, torch.device] = 'cuda'
        if torch.cuda.is_available()
        else 'cpu',
        jit: bool = False,
    ):
        print(f'===> Creating a OpenAIClipModel instance: {name}')
        if name in available_models():
            model_name, model_md5 = _OPENAI_MODELS[name]
            model_path = download_model(
                url=_OPENAI_S3_BUCKET + model_name,
                target_folder=os.path.expanduser('~/.cache/clip'),
                md5sum=model_md5,
                with_resume=True,
            )
        elif os.path.isfile(name):
            model_path = name
        else:
            raise RuntimeError(
                f'Model `{name}` not found; available models = {available_models()} in {self.__class__.__name__}'
            )

        try:
            # loading JIT archive
            model = torch.jit.load(
                model_path, map_location=device if jit else 'cpu'
            ).eval()
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
            # return model

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
                    if 'value' in node.attributeNames() and str(
                        node['value']
                    ).startswith('cuda'):
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

        # tokenizer = clip.load(name=name)
        # preprocessor = clip.load(name=name)

        # return model

    def encode_text(self, input_ids, attention_mask, **kwargs):
        ...

    def encode_image(self, pixel_values, **kwargs):
        ...
