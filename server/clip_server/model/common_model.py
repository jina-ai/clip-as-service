import open_clip
import transformers
import torch

from mclip import MultilingualCLIP
from clip_server.executors.helper import preproc_image, preproc_text


class CLIPModel:
    def __init__(self, model_name, device, jit):
        self._model = self._mclip_model = self._vision_preprocessor = self._tokenizer = None
        self._model_name = model_name
        self._device = device
        self._jit = jit

    @classmethod
    def load(cls, model_name, device, jit):
        model = cls(model_name, device, jit)
        if '::' in model_name:
            name, pretrained = model_name.split('::')
            if pretrained == "M-CLIP":  # MCLIP models need to be loaded by huggingface
                model._mclip_model = MultilingualCLIP.from_pretrained(name)
                # with its corresponding clip model for img encoding
                corresponding_clip_models = {
                    'XLM-Roberta-Large-Vit-B-32': ('ViT-B-32', 'openai'),
                    'XLM-Roberta-Large-Vit-L-14': ('ViT-L-14', 'openai'),
                    'XLM-Roberta-Large-Vit-B-16Plus': ('ViT-B-16-plus-240', 'laion400m_e31'),
                }
                model._tokenizer = transformers.AutoTokenizer.from_pretrained(name)
                name, pretrained = corresponding_clip_models[name]
            # normally we use open_clip loader
            model._model, _, model._vision_preprocessor = open_clip.create_model_and_transforms(name,
                                                                                                pretrained=pretrained,
                                                                                                device=torch.device(
                                                                                                    device), jit=jit)
            # model._model.eval().requires_grad_(False).to(device)  # mark
        else:
            raise ValueError(f'''
        Now, CLIP-as-service depends on `open-clip` which supports more CLIP variants and pretrained weights. 
        The new names is now a string in the format of `<model_name>::<pretrained_weights_name>`, e.g. 
        `ViT-B-32::openai` or `ViT-B-32::laion2b_e16`. 
        ''')
        return model

    def preproc_img(self, docs, return_np):
        return preproc_image(
            docs,
            preprocess_fn=self._vision_preprocessor,
            device=self._device,
            return_np=return_np,
        )

    def preproc_txt(self, docs, return_np):
        if self._mclip_model:
            batch_data = self._tokenizer(
                docs.texts,
                max_length=77,
                padding="longest",
                truncation=True,
                return_tensors="pt",
            )
            batch_data = {k: v.to(self._device) for k, v in batch_data.items()}
            return docs, batch_data
        else:
            return preproc_text(docs, device=self._device, return_np=return_np)

    def encode_text(self, batch_data):
        if self._mclip_model:
            return self._mclip_model.encode_text(batch_data)
        else:
            return self._model.encode_text(batch_data['input_ids'])

    def encode_image(self, batch_data):
        return self._model.encode_image(batch_data['pixel_values'])
