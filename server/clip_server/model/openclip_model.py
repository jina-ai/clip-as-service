import torch
from torch import nn
from typing import List
from clip_server.model.clip_model import CLIPModel
from clip_server.model.pretrained_models import _OPENCLIP_MODELS


def available_models() -> List[str]:
    '''Returns the names of available CLIP models'''
    return list(_OPENCLIP_MODELS.keys())


class OpenClipModel(CLIPModel, nn.Module):
    def __init__(self, name: str = 'ViT-B-32::opeanai', **kwargs):
        print(f'===> Open Clip Model: name={name}')
