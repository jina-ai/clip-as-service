import transformers
import torch
import open_clip

from clip_server.model.clip_model import CLIPModel

corresponding_clip_models = {
    'M-CLIP/XLM-Roberta-Large-Vit-B-32': ('ViT-B-32', 'openai'),
    'M-CLIP/XLM-Roberta-Large-Vi-L-14': ('ViT-L-14', 'openai'),
    'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus': ('ViT-B-16-plus-240', 'laion400m_e31'),
    'M-CLIP/LABSE-Vit-L-14': ('ViT-L-14', 'openai'),
}


class MultilingualCLIPModel(CLIPModel):
    def __init__(self, name: str, device: str, jit: bool):
        super().__init__(name, device, jit)
        self._mclip_model = MultilingualCLIP.from_pretrained(name)
        clip_name, clip_pretrained = corresponding_clip_models[name]
        self._model = open_clip.create_model(
            clip_name, pretrained=clip_pretrained, device=device, jit=jit
        )

    def encode_text(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs
    ):
        return self._mclip_model.encode_text(
            dict({"input_ids": input_ids, "attention_mask": attention_mask}, **kwargs)
        )

    def encode_image(self, pixel_values: torch.Tensor, **kwargs):
        return self._model.encode_image(pixel_values, **kwargs)


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, **kwargs):
        self.transformerDimensions = "xlm-roberta-large"
        self.numDims = 1024
        self.modelBase = 768
        super().__init__(**kwargs)


class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase)
        self.LinearTransformation = torch.nn.Linear(
            in_features=config.transformerDimensions, out_features=config.numDims
        )

    def encode_text(self, txt_tok):
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok["attention_mask"]
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)