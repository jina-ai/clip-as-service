import pytest
from clip_server.model.clip_model import CLIPModel
from clip_server.model.clip_onnx import CLIPOnnxModel
from clip_server.model.openclip_model import OpenCLIPModel
from clip_server.model.mclip_model import MultilingualCLIPModel


@pytest.mark.parametrize(
    'name, model_cls',
    [
        ('ViT-L/14@336px', OpenCLIPModel),
        ('RN50::openai', OpenCLIPModel),
        ('xlm-roberta-large-ViT-H-14::frozen_laion5b_s13b_b90k', OpenCLIPModel),
        ('M-CLIP/LABSE-Vit-L-14', MultilingualCLIPModel),
    ],
)
def test_torch_model(name, model_cls):
    model = CLIPModel(name)
    assert model.__class__ == model_cls


@pytest.mark.parametrize(
    'name',
    [
        'RN50::openai',
        'ViT-H-14::laion2b-s32b-b79k',
        'M-CLIP/LABSE-Vit-L-14',
    ],
)
def test_onnx_model(name):
    CLIPOnnxModel(name)
