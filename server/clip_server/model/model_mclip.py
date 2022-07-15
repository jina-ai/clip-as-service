import clip_server.model.common_model as CommonModel
from clip_server.model.mclip import MultilingualCLIP
import transformers
import open_clip


class MultilingualCLIPModel(CommonModel.CLIPModel):
    @classmethod
    def load(cls, model_name, device, jit):
        model = cls(model_name, device, jit)
        name, pretrained = model_name.split("::")
        model_path = "{1}/{0}".format(name, pretrained)
        model._mclip_model = MultilingualCLIP.from_pretrained(model_path)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
        corresponding_clip_models = {
            'XLM-Roberta-Large-Vit-B-32': ('ViT-B-32', 'openai'),
            'XLM-Roberta-Large-Vi-L-14': ('ViT-L-14', 'openai'),
            'XLM-Roberta-Large-Vit-B-16Plus': ('ViT-B-16-plus-240', 'laion400m_e31'),
        }
        clip_name, clip_pretrained = corresponding_clip_models[name]
        # use open_clip to load clip model for img encoding
        model._model, _, vision_preprocessor = open_clip.create_model_and_transforms(clip_name,
                                                                                     pretrained=clip_pretrained,
                                                                                     device=device, jit=jit)
        return model, tokenizer, vision_preprocessor

    def encode_text(self, batch_data):
        return self._mclip_model.encode_text(batch_data)

    def preproc_txt(self, docs, return_np):
        batch_data = self._tokenizer(
            docs.texts,
            max_length=77,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        batch_data = {k: v.to(self._device) for k, v in batch_data.items()}
        return docs, batch_data

