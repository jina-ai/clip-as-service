import pytest
from clip_server.model.clip_model import CLIPModel
from clip_server.model.openclip_model import OpenCLIPModel
from clip_server.model.mclip_model import MultilingualCLIPModel


@pytest.mark.parametrize(
    'name, model_cls',
    [
        ('ViT-L/14@336px', OpenCLIPModel),
        ('RN101::openai', OpenCLIPModel),
        ('M-CLIP/XLM-Roberta-Large-Vit-B-32', MultilingualCLIPModel),
    ],
)
def test_model_name(name, model_cls):
    model = CLIPModel(name)
    assert model.__class__ == model_cls
