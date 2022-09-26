# Originally from https://github.com/FreddeFrallan/Multilingual-CLIP. MIT License, Copyright (c) 2022 Multilingual-CLIP

import transformers
import torch

from clip_server.model.clip_model import CLIPModel
from clip_server.model.openclip_model import OpenCLIPModel

_CLIP_MODEL_MAPS = {
    'm-clip/xlm-roberta-large-vit-b-32': 'vit-b-32::openai',
    'm-clip/xlm-roberta-large-vit-l-14': 'vit-l-14::openai',
    'm-clip/xlm-roberta-large-vit-b-16plus': 'vit-b-16-plus-240::laion400m_e31',
    'm-clip/labse-vit-l-14': 'vit-l-14::openai',
}

_MCLIP_MODEL_MAPS = {
    'm-clip/xlm-roberta-large-vit-b-32': 'M-CLIP/XLM-Roberta-Large-Vit-B-32',
    'm-clip/xlm-roberta-large-vit-l-14': 'M-CLIP/XLM-Roberta-Large-Vit-L-14',
    'm-clip/xlm-roberta-large-vit-b-16plus': 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus',
    'm-clip/labse-vit-l-14': 'M-CLIP/LABSE-Vit-L-14',
}


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = 'm-clip'

    def __init__(
        self,
        modelBase: str = 'xlm-roberta-large',
        transformerDimSize: int = 1024,
        imageDimSize: int = 768,
        **kwargs
    ):
        self.transformerDimensions = transformerDimSize
        self.numDims = imageDimSize
        self.modelBase = modelBase
        super().__init__(**kwargs)


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase)
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs):
        embs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )[0]
        embs = (embs * attention_mask.unsqueeze(2)).sum(dim=1) / attention_mask.sum(
            dim=1
        )[:, None]
        return self.LinearTransformation(embs)


class MultilingualCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str = 'cpu', jit: bool = False, **kwargs):
        super().__init__(name, **kwargs)
        self._mclip_model = MultilingualCLIP.from_pretrained(
            _MCLIP_MODEL_MAPS[name.lower()]
        )
        self._mclip_model.to(device=device)
        self._mclip_model.eval()
        self._model = OpenCLIPModel(
            _CLIP_MODEL_MAPS[name.lower()], device=device, jit=jit
        )

    @staticmethod
    def get_model_name(name: str):
        return _CLIP_MODEL_MAPS[name].split('::')[0]

    def encode_text(
        self, input_ids: 'torch.Tensor', attention_mask: 'torch.Tensor', **kwargs
    ):
        return self._mclip_model(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs
        )

    def encode_image(self, pixel_values: torch.Tensor):
        return self._model.encode_image(pixel_values)
