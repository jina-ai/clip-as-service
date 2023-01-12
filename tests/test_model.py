import pytest
from clip_server.model.clip_model import CLIPModel
from clip_server.model.clip_onnx import CLIPOnnxModel
from clip_server.model.openclip_model import OpenCLIPModel
from clip_server.model.mclip_model import MultilingualCLIPModel
from clip_server.model.cnclip_model import CNClipModel


@pytest.mark.parametrize(
    'name, model_cls',
    [
        ('ViT-L/14@336px', OpenCLIPModel),
        ('RN50::openai', OpenCLIPModel),
        ('roberta-ViT-B-32::laion2b-s12b-b32k', OpenCLIPModel),
        ('M-CLIP/LABSE-Vit-L-14', MultilingualCLIPModel),
        ('ViT-B-16', CNClipModel),
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


@pytest.mark.gpu
@pytest.mark.parametrize(
    'name',
    ['ViT-H-14::laion2b-s32b-b79k'],
)
def test_large_onnx_model_fp16(name):
    from clip_server.executors.clip_onnx import CLIPEncoder

    CLIPEncoder(name, dtype='fp16')
