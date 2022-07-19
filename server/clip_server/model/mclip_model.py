import transformers
import torch
import open_clip

from .clip_model import CLIPModel


class MultilingualCLIPModel(CLIPModel):
    def __init__(self, name, device, jit):
        super().__init__(name, device, jit)
        model_name, pretrained = name.split("::")
        self._mclip_path = "{1}/{0}".format(model_name, pretrained)
        self._mclip_model = MultilingualCLIP.from_pretrained(self._mclip_path)
        corresponding_clip_models = {
            'XLM-Roberta-Large-Vit-B-32': ('ViT-B-32', 'openai'),
            'XLM-Roberta-Large-Vi-L-14': ('ViT-L-14', 'openai'),
            'XLM-Roberta-Large-Vit-B-16Plus': ('ViT-B-16-plus-240', 'laion400m_e31'),
        }
        clip_name, clip_pretrained = corresponding_clip_models[model_name]
        self._model = open_clip.create_model(
            clip_name, pretrained=clip_pretrained, device=device, jit=jit
        )

    def encode_text(self, input_ids, attention_mask, **kwargs):
        return self._mclip_model.encode_text(
            {"input_ids": input_ids, "attention_mask": attention_mask}
        )


class MCLIPConfig(transformers.PretrainedConfig):
    model_type = "M-CLIP"

    def __init__(self, **kwargs):
        self.transformerDimensions = 'xlm-roberta-large'
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
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)
